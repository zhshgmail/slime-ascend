import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

SUPPORTED_NPU_PROFILER_CONTENTS = frozenset({"npu", "cpu", "memory", "shapes", "module", "stack"})


@dataclass(frozen=True)
class NPUProfilerContents:
    raw: tuple[str, ...]
    values: frozenset[str] = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "values", frozenset(self.raw))

    @classmethod
    def from_contents(cls, contents: Optional[List[str]]) -> "NPUProfilerContents":
        normalized = tuple(contents or [])
        invalid_contents = set(normalized) - SUPPORTED_NPU_PROFILER_CONTENTS
        if invalid_contents:
            raise ValueError(
                f"Profiler contents only supports {sorted(SUPPORTED_NPU_PROFILER_CONTENTS)}, "
                f"but gets {sorted(invalid_contents)}"
            )
        return cls(raw=normalized)

    def has(self, name: str) -> bool:
        return name in self.values

    @property
    def with_cpu(self) -> bool:
        return self.has("cpu")

    @property
    def with_memory(self) -> bool:
        return self.has("memory")

    @property
    def with_stack(self) -> bool:
        return self.has("stack")

    @property
    def with_modules(self) -> bool:
        return self.has("module")

    @property
    def record_shapes(self) -> bool:
        return self.has("shapes")

    @property
    def include_npu(self) -> bool:
        return self.has("npu")


@dataclass
class NPUProfilerConfig:
    """Configuration for torch_npu.profiler collection."""

    enabled: bool = False
    step_start: int = 0  # First rollout step to collect (inclusive)
    step_end: int = -1  # Last rollout step to collect (exclusive), -1 = until training ends
    ranks: Optional[List[int]] = None  # Ranks to collect on, None/-1 = all ranks
    level: str = "level1"  # level_none / level0 / level1 / level2
    export_type: str = "db"  # text / db
    contents: Optional[List[str]] = None  # npu / cpu / memory / shapes / module / stack
    with_cpu: bool = False
    with_memory: bool = False
    with_stack: bool = False
    with_modules: bool = False
    record_shapes: bool = False
    analysis: bool = False
    save_path: str = "./npu_profile"
    current_rank: int = 0
    _parsed_contents: Optional[NPUProfilerContents] = field(init=False, default=None, repr=False)

    def __post_init__(self):
        if self.ranks is None:
            self.ranks = [-1]
        if self.contents is not None:
            self._parsed_contents = NPUProfilerContents.from_contents(self.contents)
            parsed_contents = self._parsed_contents
            self.with_cpu = parsed_contents.with_cpu
            self.with_memory = parsed_contents.with_memory
            self.with_stack = parsed_contents.with_stack
            self.with_modules = parsed_contents.with_modules
            self.record_shapes = parsed_contents.record_shapes

    def parsed_contents(self) -> NPUProfilerContents:
        return self._parsed_contents or NPUProfilerContents.from_contents(self.contents)

    def uses_content(self, name: str) -> bool:
        return self.parsed_contents().has(name)

    def is_profiling_rank(self) -> bool:
        if not self.enabled:
            return False
        if -1 in self.ranks:
            return True
        return self.current_rank in self.ranks


class NPUProfilerManager:
    """Wrapper of torch_npu.profiler for NPU performance collection.

    A process must have at most one NPUProfilerManager instance (single instance
    per process). It exposes start()/step()/stop() and derives the profiling
    schedule internally from the configured step window, so callers do not need
    to touch torch_npu.profiler details.

    torch_npu is imported lazily so that this module stays importable on GPU
    machines where torch_npu is not installed.
    """

    _instance: Optional["NPUProfilerManager"] = None

    def __init__(self, config: NPUProfilerConfig):
        if NPUProfilerManager._instance is not None:
            raise RuntimeError("NPUProfilerManager must be a single instance per process.")
        NPUProfilerManager._instance = self

        self.config = config
        self.profiler = None
        self._started = False
        self._stopped = False

        if not config.is_profiling_rank():
            return

        Path(config.save_path).mkdir(parents=True, exist_ok=True)

        try:
            import torch_npu
        except ImportError as e:
            raise RuntimeError(
                "`--npu-profile-actor` requires torch_npu, which is not available "
                "in the current environment."
            ) from e

        self._torch_npu = torch_npu
        self.profiler = self._create_profiler()
        self._add_distributed_metadata()

    def _create_profiler(self):
        config = self.config
        torch_npu = self._torch_npu

        level_map = {
            "level_none": torch_npu.profiler.ProfilerLevel.Level_none,
            "level0": torch_npu.profiler.ProfilerLevel.Level0,
            "level1": torch_npu.profiler.ProfilerLevel.Level1,
            "level2": torch_npu.profiler.ProfilerLevel.Level2,
        }
        export_type_map = {
            "text": [torch_npu.profiler.ExportType.Text],
            "db": [torch_npu.profiler.ExportType.Db],
        }
        if config.level not in level_map:
            raise ValueError(f"Invalid npu profile level: {config.level}. Supported: {list(level_map.keys())}")
        if config.export_type not in export_type_map:
            raise ValueError(
                f"Invalid npu profile export type: {config.export_type}. Supported: {list(export_type_map.keys())}"
            )

        # Derive the schedule from the step window.
        if config.step_end == -1:
            active = 1000000
        else:
            active = config.step_end - config.step_start
            if active <= 0:
                raise ValueError(
                    f"npu profile step_end ({config.step_end}) must be > step_start ({config.step_start})."
                )
        skip_first = max(0, config.step_start - 1)
        warmup = 0 if config.step_start == 0 else 1

        # record_shapes / profile_memory / with_stack / with_modules only take effect
        # when the CPU activity is enabled, so enable it implicitly when any of them is requested.
        if config.contents is None or config.uses_content("npu"):
            activities = [torch_npu.profiler.ProfilerActivity.NPU]
        else:
            activities = []
        needs_cpu_activity = self._needs_cpu_activity(config)
        if needs_cpu_activity:
            activities.append(torch_npu.profiler.ProfilerActivity.CPU)

        return torch_npu.profiler.profile(
            activities=activities,
            schedule=torch_npu.profiler.schedule(
                wait=0, warmup=warmup, active=active, repeat=1, skip_first=skip_first
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                config.save_path,
                analyse_flag=config.analysis,
            ),
            record_shapes=config.record_shapes,
            profile_memory=config.with_memory,
            with_stack=config.with_stack,
            with_modules=config.with_modules,
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                profiler_level=level_map[config.level],
                export_type=export_type_map[config.export_type],
            ),
        )

    @staticmethod
    def _needs_cpu_activity(config: NPUProfilerConfig) -> bool:
        cpu_dependent_options_enabled = any(
            (
                config.with_cpu,
                config.with_stack,
                config.with_memory,
                config.record_shapes,
                config.with_modules,
            )
        )
        if config.contents is None:
            return cpu_dependent_options_enabled
        return config.uses_content("cpu") or cpu_dependent_options_enabled

    def _add_distributed_metadata(self):
        if self.profiler is None:
            return

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.profiler.add_metadata_json(
            "distributed_args",
            json.dumps({"rank": self.config.current_rank, "world_size": world_size}),
        )

    def start(self):
        if self.profiler is not None and not self._started:
            self.profiler.start()
            self._started = True
            logger.info(f"[RANK {self.config.current_rank}] NPU profiling started.")

    def step(self):
        if self.profiler is not None:
            self.profiler.step()

    def stop(self):
        if self.profiler is not None and not self._stopped:
            self.profiler.stop()
            self._stopped = True
            logger.info(
                f"[RANK {self.config.current_rank}] NPU profiling stopped. "
                f"Trace saved to {self.config.save_path}"
            )
