# pylint: disable=too-many-lines
"""CPU unit tests for MOPD (Multi-Teacher On-Policy Distillation, PR #116).

Covers the points promised by docs/zh/advanced/multi-teacher-on-policy-distillation.md:

1. ``slime.rollout.mopd._build_payload`` — SGLang request payload for the three
   distill types (token_level / top_k / full_vocab) and multimodal handling.
2. ``_resolve_teacher_urls`` — rm_url priority: teacher config > MOPD_TEACHER_URLS
   env var > args.rm_url.
3. ``_get_sample_domains`` — per-sample domain routing.
4. ``_extract_teacher_data_from_responses`` — SGLang response alignment (skip
   first prompt entry, tail-align to response_length, -inf padding), top-k
   extraction/padding, and the failed-teacher fallback that distinguishes
   "queried but failed" (filled with -inf) from "routed away" (left absent)
   via the ``__target_domains__`` internal key.
5. ``post_process_rewards`` / ``combined_post_process_rewards``.
6. ``slime.utils.ppo_utils.vocab_parallel_reverse_kl`` and
   ``vocab_parallel_topk_reverse_kl`` — numeric agreement with a naive
   reference and gradient correctness (process_group=None single-process path).
7. ``slime.backends.megatron_utils.loss`` — apply_mopd_to_advantages
   (token_level), apply_mopd_full_vocab_to_loss, apply_mopd_topk_to_loss.
8. ``DataIterator.get_next`` recursive slicing of dict[domain, list] fields.
9. ``Sample`` MOPD fields and ``slime_validate_args`` MOPD validation branches.

Everything runs on CPU without megatron / sglang / ray / network. Heavy or
unavailable third-party imports (megatron, sglang_router, wandb, PIL/
transformers via processing_utils) are stubbed in sys.modules *before* the
slime modules are imported; the code under test runs for real.

Run:
    pytest -q tests/tests_npu/ut/test_mopd_npu.py
"""

from __future__ import annotations

import asyncio
import json
import math
from argparse import Namespace

import aiohttp
import pytest
import torch

# Installs the dependency stubs in sys.modules; must precede slime imports.
from _mopd_stub_setup import rm_hub_stub
from slime.backends.megatron_utils.data import DataIterator
from slime.backends.megatron_utils.loss import (
    apply_mopd_full_vocab_to_loss,
    apply_mopd_to_advantages,
    apply_mopd_topk_to_loss,
    compute_advantages_and_returns,
)
from slime.rollout import mopd
from slime.utils.arguments import slime_validate_args
from slime.utils.ppo_utils import (
    vocab_parallel_reverse_kl,
    vocab_parallel_topk_reverse_kl,
)
from slime.utils.types import Sample

NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _mopd_field(rollout_data: dict, key: str):
    """Fetch a MOPD output field, failing loudly when the loss fn skipped it."""
    value = rollout_data.get(key)
    assert value is not None, f"rollout_data missing {key}"
    return value


def _make_rollout_args(**overrides) -> Namespace:
    """Minimal args for slime.rollout.mopd functions."""
    defaults = dict(
        reward_key=None,
        rm_url="default-rm/generate",
        mopd_teachers='[{"name": "math_teacher", "domain": "math"}, {"name": "code_teacher", "domain": "code"}]',
        mopd_distill_type="token_level",
        mopd_topk_k=4,
        mopd_teacher_max_retries=1,
        mopd_teacher_retry_delay=0.0,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _make_sample(**overrides) -> Sample:
    defaults = dict(
        prompt="What is 1+1?",
        response="2",
        tokens=[10, 11, 12, 13, 14],
        response_length=3,
        reward=0.0,
    )
    defaults.update(overrides)
    return Sample(**defaults)


def _sglang_response(logprob_entries, top_logprob_entries=None):
    """Build a fake SGLang /generate response.

    logprob_entries: list of [log_prob, token_id, token_text] (first entry is
    the prompt placeholder, typically None).
    top_logprob_entries: optional list (per position) of list of
    (log_prob, token_id, token_text) tuples.
    """
    meta_info = {"input_token_logprobs": logprob_entries}
    if top_logprob_entries is not None:
        meta_info["input_top_logprobs"] = top_logprob_entries
    return {"meta_info": meta_info}


def _naive_reverse_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """Reference D_KL(pi_student || pi_teacher) = sum_y p_s (log p_s - log p_t)."""
    s_logp = torch.log_softmax(student_logits, dim=-1)
    t_logp = torch.log_softmax(teacher_logits, dim=-1)
    return (s_logp.exp() * (s_logp - t_logp)).sum(dim=-1)


def _naive_topk_reverse_kl(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, topk_indices: torch.Tensor
) -> torch.Tensor:
    """Reference top-k + tail-mass reverse KL with exact teacher probabilities."""
    s_logp = torch.log_softmax(student_logits, dim=-1)
    t_logp = torch.log_softmax(teacher_logits, dim=-1)
    s_p = s_logp.exp()
    t_p = t_logp.exp()

    s_top_p = s_p.gather(-1, topk_indices)
    s_top_lp = s_logp.gather(-1, topk_indices)
    t_top_lp = t_logp.gather(-1, topk_indices)
    t_top_p = t_p.gather(-1, topk_indices)

    kl_topk = (s_top_p * (s_top_lp - t_top_lp)).sum(dim=-1)
    s_tail = (1.0 - s_top_p.sum(dim=-1)).clamp(min=0.0)
    t_tail = (1.0 - t_top_p.sum(dim=-1)).clamp(min=0.0)
    tail_mask = (s_tail > 1e-10) & (t_tail > 1e-10)
    kl_tail = torch.zeros_like(s_tail)
    kl_tail[tail_mask] = s_tail[tail_mask] * (s_tail[tail_mask].log() - t_tail[tail_mask].log())
    return kl_topk + kl_tail


def _mean_reducer(x: torch.Tensor) -> torch.Tensor:
    return x.float().mean()


# ===========================================================================
# 1. _build_payload
# ===========================================================================
@pytest.mark.unit
class TestBuildPayload:
    def test_token_level_payload_has_no_top_logprobs_num(self):
        args = _make_rollout_args(mopd_distill_type="token_level")
        sample = _make_sample()
        payload = mopd._build_payload(sample, args)
        assert payload["return_logprob"] is True
        assert payload["logprob_start_len"] == 0
        assert "top_logprobs_num" not in payload
        assert payload["input_ids"] == sample.tokens
        assert "text" not in payload
        assert payload["sampling_params"]["temperature"] == 0
        assert payload["sampling_params"]["max_new_tokens"] == 0
        assert payload["sampling_params"]["skip_special_tokens"] is False

    def test_top_k_payload_carries_top_logprobs_num(self):
        args = _make_rollout_args(mopd_distill_type="top_k", mopd_topk_k=7)
        payload = mopd._build_payload(_make_sample(), args)
        assert payload["top_logprobs_num"] == 7
        assert payload["input_ids"] == _make_sample().tokens

    def test_full_vocab_raises_for_sglang(self):
        args = _make_rollout_args(mopd_distill_type="full_vocab")
        with pytest.raises(ValueError, match="full_vocab"):
            mopd._build_payload(_make_sample(), args)

    def test_multimodal_uses_text_and_image_data(self):
        args = _make_rollout_args()
        sample = _make_sample(prompt="<image> describe", response="a cat", multimodal_inputs={"images": ["img0"]})
        payload = mopd._build_payload(sample, args)
        assert payload["text"] == "<image> describea cat"
        assert payload["image_data"] == ["encoded:img0"]
        assert "input_ids" not in payload

    def test_multimodal_requires_string_prompt(self):
        args = _make_rollout_args()
        sample = _make_sample(
            prompt=[{"role": "user", "content": "hi"}],
            multimodal_inputs={"images": ["img0"]},
        )
        with pytest.raises(ValueError, match="prompt to be a string"):
            mopd._build_payload(sample, args)


# ===========================================================================
# 2. _resolve_teacher_urls
# ===========================================================================
@pytest.mark.unit
class TestResolveTeacherUrls:
    _CONFIGS = [
        {"name": "math_teacher", "domain": "math", "rm_url": "math/generate"},
        {"name": "code_teacher", "domain": "code"},
    ]

    def test_config_rm_url_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("MOPD_TEACHER_URLS", json.dumps({"math": "env-math/generate"}))
        args = _make_rollout_args()
        url_map = mopd._resolve_teacher_urls(args, self._CONFIGS)
        assert url_map["math"] == "math/generate"

    def test_env_var_used_when_config_lacks_url(self, monkeypatch):
        monkeypatch.setenv("MOPD_TEACHER_URLS", json.dumps({"code": "env-code/generate"}))
        args = _make_rollout_args()
        url_map = mopd._resolve_teacher_urls(args, self._CONFIGS)
        assert url_map["code"] == "env-code/generate"

    def test_args_rm_url_fallback(self, monkeypatch):
        monkeypatch.delenv("MOPD_TEACHER_URLS", raising=False)
        args = _make_rollout_args(rm_url="fallback/generate")
        url_map = mopd._resolve_teacher_urls(args, self._CONFIGS)
        assert url_map["code"] == "fallback/generate"
        assert url_map["math"] == "math/generate"


# ===========================================================================
# 3. _get_sample_domains
# ===========================================================================
@pytest.mark.unit
class TestGetSampleDomains:
    _ALL = ["math", "code"]

    def test_no_metadata_queries_all(self):
        assert mopd._get_sample_domains(_make_sample(metadata={}), self._ALL) is None

    def test_single_teacher_disables_routing(self):
        sample = _make_sample(metadata={"mopd_domains": ["math"]})
        assert mopd._get_sample_domains(sample, ["math"]) is None

    def test_string_domain_becomes_list(self):
        sample = _make_sample(metadata={"mopd_domains": "math"})
        assert mopd._get_sample_domains(sample, self._ALL) == ["math"]

    def test_list_filtered_to_valid_domains(self):
        sample = _make_sample(metadata={"mopd_domains": ["math", "unknown"]})
        assert mopd._get_sample_domains(sample, self._ALL) == ["math"]

    def test_all_unknown_falls_back_to_all(self):
        sample = _make_sample(metadata={"mopd_domains": ["nope"]})
        assert mopd._get_sample_domains(sample, self._ALL) is None

    def test_non_dict_metadata_queries_all(self):
        sample = _make_sample()
        sample.metadata = "not-a-dict"
        assert mopd._get_sample_domains(sample, self._ALL) is None


# ===========================================================================
# 4. _extract_teacher_data_from_responses (token_level)
# ===========================================================================
@pytest.mark.unit
class TestExtractTeacherDataTokenLevel:
    def test_skips_first_prompt_entry(self):
        args = _make_rollout_args()
        sample = _make_sample(
            response_length=2,
            reward={"math": _sglang_response([None, [-0.5, 10, "a"], [-0.6, 11, "b"]])},
        )
        mopd._extract_teacher_data_from_responses(args, [sample])
        assert torch.allclose(sample.mopd_teacher_log_probs["math"], torch.tensor([-0.5, -0.6]))

    def test_tail_alignment_when_sglang_returns_prompt_tokens(self):
        # 1 prompt placeholder + 2 prompt logprobs + 3 response logprobs.
        args = _make_rollout_args()
        sample = _make_sample(
            response_length=3,
            reward={
                "math": _sglang_response(
                    [None, [-9.0, 1, "p"], [-9.1, 2, "p"], [-0.1, 3, "r"], [-0.2, 4, "r"], [-0.3, 5, "r"]]
                )
            },
        )
        mopd._extract_teacher_data_from_responses(args, [sample])
        assert torch.allclose(sample.mopd_teacher_log_probs["math"], torch.tensor([-0.1, -0.2, -0.3]))

    def test_short_response_padded_with_neg_inf(self):
        args = _make_rollout_args()
        sample = _make_sample(
            response_length=4,
            reward={"math": _sglang_response([None, [-0.5, 10, "a"], [-0.6, 11, "b"]])},
        )
        mopd._extract_teacher_data_from_responses(args, [sample])
        lp = sample.mopd_teacher_log_probs["math"]
        assert lp.size(0) == 4
        assert torch.allclose(lp[:2], torch.tensor([-0.5, -0.6]))
        assert torch.isinf(lp[2:]).all() and (lp[2:] < 0).all()

    def test_queried_but_failed_teacher_gets_neg_inf_fallback(self):
        args = _make_rollout_args()
        sample = _make_sample(
            response_length=3,
            reward={
                "math": _sglang_response([None, [-0.5, 10, "a"], [-0.6, 11, "b"], [-0.7, 12, "c"]]),
                # "code" was queried but the request failed -> absent from dict.
                "__target_domains__": ["math", "code"],
            },
        )
        mopd._extract_teacher_data_from_responses(args, [sample])
        assert "code" in sample.mopd_teacher_log_probs
        assert torch.isinf(sample.mopd_teacher_log_probs["code"]).all()
        assert sample.mopd_teacher_log_probs["code"].size(0) == 3

    def test_routed_away_domain_not_filled(self):
        args = _make_rollout_args()
        sample = _make_sample(
            response_length=2,
            reward={
                "math": _sglang_response([None, [-0.5, 10, "a"], [-0.6, 11, "b"]]),
                # "code" was never queried for this sample (domain routing).
                "__target_domains__": ["math"],
            },
        )
        mopd._extract_teacher_data_from_responses(args, [sample])
        assert "math" in sample.mopd_teacher_log_probs
        assert "code" not in sample.mopd_teacher_log_probs

    def test_no_target_domains_marker_fills_all_configured(self):
        # Backward compatibility: without __target_domains__, every configured
        # domain missing from the response gets the -inf fallback.
        args = _make_rollout_args()
        sample = _make_sample(
            response_length=2,
            reward={"math": _sglang_response([None, [-0.5, 10, "a"], [-0.6, 11, "b"]])},
        )
        mopd._extract_teacher_data_from_responses(args, [sample])
        assert "code" in sample.mopd_teacher_log_probs
        assert torch.isinf(sample.mopd_teacher_log_probs["code"]).all()

    def test_non_dict_reward_fills_all_configured_with_neg_inf(self):
        # A non-dict reward (e.g. 0.0) carries no __target_domains__ marker,
        # so the backward-compatible fallback treats every configured domain
        # as "queried but failed" and fills it with -inf.
        args = _make_rollout_args()
        sample = _make_sample(reward=0.0)
        mopd._extract_teacher_data_from_responses(args, [sample])
        assert set(sample.mopd_teacher_log_probs.keys()) == {"math", "code"}
        assert torch.isinf(sample.mopd_teacher_log_probs["math"]).all()
        assert torch.isinf(sample.mopd_teacher_log_probs["code"]).all()

    def test_malformed_response_falls_back_to_neg_inf(self):
        args = _make_rollout_args()
        sample = _make_sample(
            response_length=2,
            reward={"math": {"meta_info": {}}, "__target_domains__": ["math"]},
        )
        mopd._extract_teacher_data_from_responses(args, [sample])
        assert torch.isinf(sample.mopd_teacher_log_probs["math"]).all()


# ===========================================================================
# 5. _extract_teacher_data_from_responses (top_k)
# ===========================================================================
@pytest.mark.unit
class TestExtractTeacherDataTopK:
    def _topk_args(self, k=3):
        return _make_rollout_args(mopd_distill_type="top_k", mopd_topk_k=k)

    def test_topk_extraction_skips_first_and_converts(self):
        args = self._topk_args(k=2)
        sample = _make_sample(
            response_length=2,
            reward={
                "math": _sglang_response(
                    [None, [-0.5, 10, "a"], [-0.6, 11, "b"]],
                    top_logprob_entries=[
                        None,
                        [(-0.5, 10, None), (-1.5, 11, None)],
                        [(-0.6, 12, None), (-1.6, 13, None)],
                    ],
                )
            },
        )
        mopd._extract_teacher_data_from_responses(args, [sample])
        assert sample.mopd_teacher_topk_logits["math"] == [[-0.5, -1.5], [-0.6, -1.6]]
        assert sample.mopd_teacher_topk_indices["math"] == [[10, 11], [12, 13]]
        # token_level log-probs are still extracted (needed for IS weights).
        assert torch.allclose(sample.mopd_teacher_log_probs["math"], torch.tensor([-0.5, -0.6]))

    def test_topk_positions_padded_to_k_with_neg_inf(self):
        args = self._topk_args(k=3)
        sample = _make_sample(
            response_length=2,
            reward={
                "math": _sglang_response(
                    [None, [-0.5, 10, "a"], [-0.6, 11, "b"]],
                    top_logprob_entries=[
                        None,
                        [(-0.5, 10, None)],  # only 1 entry, padded to k=3
                        None,  # missing position -> all -inf / 0
                    ],
                )
            },
        )
        mopd._extract_teacher_data_from_responses(args, [sample])
        logits = sample.mopd_teacher_topk_logits["math"]
        indices = sample.mopd_teacher_topk_indices["math"]
        assert logits[0] == [-0.5, NEG_INF, NEG_INF]
        assert indices[0] == [10, 0, 0]
        assert logits[1] == [NEG_INF, NEG_INF, NEG_INF]
        assert indices[1] == [0, 0, 0]

    def test_topk_missing_input_top_logprobs_gets_fallback(self):
        # Response has token logprobs but no top-k data: log_probs are still
        # extracted, and top-k dicts get the -inf/0 fallback fill.
        args = self._topk_args(k=2)
        sample = _make_sample(
            response_length=2,
            reward={"math": _sglang_response([None, [-0.5, 10, "a"], [-0.6, 11, "b"]])},
        )
        mopd._extract_teacher_data_from_responses(args, [sample])
        assert torch.allclose(sample.mopd_teacher_log_probs["math"], torch.tensor([-0.5, -0.6]))
        assert sample.mopd_teacher_topk_logits["math"] == [[NEG_INF, NEG_INF], [NEG_INF, NEG_INF]]
        assert sample.mopd_teacher_topk_indices["math"] == [[0, 0], [0, 0]]

    def test_topk_tail_alignment(self):
        args = self._topk_args(k=2)
        sample = _make_sample(
            response_length=1,
            reward={
                "math": _sglang_response(
                    [None, [-9.0, 1, "p"], [-0.6, 11, "b"]],
                    top_logprob_entries=[
                        None,
                        [(-9.0, 1, None), (-9.1, 2, None)],
                        [(-0.6, 12, None), (-1.6, 13, None)],
                    ],
                )
            },
        )
        mopd._extract_teacher_data_from_responses(args, [sample])
        assert sample.mopd_teacher_topk_logits["math"] == [[-0.6, -1.6]]
        assert sample.mopd_teacher_topk_indices["math"] == [[12, 13]]

    def test_topk_failed_teacher_gets_fallback(self):
        args = self._topk_args(k=2)
        sample = _make_sample(
            response_length=2,
            reward={
                "math": _sglang_response(
                    [None, [-0.5, 10, "a"], [-0.6, 11, "b"]],
                    top_logprob_entries=[
                        None,
                        [(-0.5, 10, None), (-1.5, 11, None)],
                        [(-0.6, 12, None), (-1.6, 13, None)],
                    ],
                ),
                "__target_domains__": ["math", "code"],
            },
        )
        mopd._extract_teacher_data_from_responses(args, [sample])
        assert sample.mopd_teacher_topk_logits["code"] == [[NEG_INF, NEG_INF], [NEG_INF, NEG_INF]]
        assert torch.isinf(sample.mopd_teacher_log_probs["code"]).all()


# ===========================================================================
# 6. post_process_rewards / combined_post_process_rewards
# ===========================================================================
@pytest.mark.unit
class TestPostProcessRewards:
    def test_zeroes_rewards_and_returns_zero_lists(self):
        args = _make_rollout_args()
        samples = [
            _make_sample(
                response_length=2, reward={"math": _sglang_response([None, [-0.5, 10, "a"], [-0.6, 11, "b"]])}
            ),
            _make_sample(
                response_length=2, reward={"math": _sglang_response([None, [-0.7, 12, "c"], [-0.8, 13, "d"]])}
            ),
        ]
        raw, processed = mopd.post_process_rewards(args, samples)
        assert raw == [0.0, 0.0]
        assert processed == [0.0, 0.0]
        for sample in samples:
            assert sample.reward == 0.0
            assert "math" in sample.mopd_teacher_log_probs

    def test_combined_extracts_from_metadata_and_normalizes(self):
        teacher_responses = {
            "math": _sglang_response([None, [-0.5, 10, "a"], [-0.6, 11, "b"]]),
            "__target_domains__": ["math"],
        }
        samples = [
            _make_sample(response_length=2, reward=1.0, metadata={mopd._MOPD_TEACHER_RESPONSES_KEY: teacher_responses}),
            _make_sample(response_length=2, reward=3.0, metadata={mopd._MOPD_TEACHER_RESPONSES_KEY: teacher_responses}),
        ]
        args = _make_rollout_args(
            advantage_estimator="grpo",
            rewards_normalization=True,
            grpo_std_normalization=True,
            n_samples_per_prompt=2,
            rollout_batch_size=1,
        )
        raw, processed = mopd.combined_post_process_rewards(args, samples)
        assert raw == [1.0, 3.0]
        # Group mean = 2.0 -> [-1, 1]; std (unbiased) = sqrt(2).
        expected = (-1.0) / (math.sqrt(2.0) + 1e-6)
        assert processed[0] == pytest.approx(expected, rel=1e-5)
        assert processed[1] == pytest.approx(-expected, rel=1e-5)
        for sample in samples:
            # Task reward restored; temporary metadata cleaned up.
            assert mopd._MOPD_TEACHER_RESPONSES_KEY not in sample.metadata
            assert "math" in sample.mopd_teacher_log_probs
        assert samples[0].reward == 1.0
        assert samples[1].reward == 3.0

    def test_combined_without_normalization_returns_raw(self):
        samples = [_make_sample(response_length=2, reward=1.5, metadata={})]
        args = _make_rollout_args(
            advantage_estimator="grpo",
            rewards_normalization=False,
            grpo_std_normalization=True,
            n_samples_per_prompt=1,
            rollout_batch_size=1,
        )
        raw, processed = mopd.combined_post_process_rewards(args, samples)
        assert raw == [1.5]
        assert processed == [1.5]


# ===========================================================================
# 7. reward_func with a fake aiohttp session
# ===========================================================================
class _FakeResponse:
    def __init__(self, payload=None, status=200):
        self._payload = payload
        self.status = status

    def raise_for_status(self):
        if self.status >= 400:
            raise aiohttp.ClientResponseError(None, (), status=self.status, message="fake error")

    async def json(self):
        return self._payload


class _FakePostCtx:
    def __init__(self, resp=None, exc=None):
        self._resp = resp
        self._exc = exc

    async def __aenter__(self):
        if self._exc is not None:
            raise self._exc
        return self._resp

    async def __aexit__(self, *args):
        return False


class _FakeSession:
    """Behavior: callable(url, payload, call_index) -> _FakeResponse | Exception."""

    def __init__(self, behavior):
        self.behavior = behavior
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def post(self, url, **kwargs):
        self.calls.append(url)
        outcome = self.behavior(url, kwargs.get("json"), len(self.calls))
        if isinstance(outcome, Exception):
            return _FakePostCtx(exc=outcome)
        return _FakePostCtx(resp=outcome)


@pytest.mark.unit
class TestRewardFuncWithFakeSession:
    def test_failed_teacher_skipped_but_recorded_as_targeted(self, monkeypatch):
        good_response = _sglang_response([None, [-0.5, 10, "a"], [-0.6, 11, "b"]])

        def behavior(url, payload, call_index):
            if "code" in url:
                return aiohttp.ClientConnectionError("boom")
            return _FakeResponse(good_response)

        session = _FakeSession(behavior)
        monkeypatch.setattr(mopd.aiohttp, "ClientSession", lambda *a, **k: session)

        # Distinct per-teacher rm_urls so the fake session can fail only the
        # code teacher by matching on the URL.
        args = _make_rollout_args(
            mopd_teachers=json.dumps(
                [
                    {"name": "math_teacher", "domain": "math", "rm_url": "math-teacher/generate"},
                    {"name": "code_teacher", "domain": "code", "rm_url": "code-teacher/generate"},
                ]
            )
        )
        sample = _make_sample(response_length=2)
        results = asyncio.run(mopd.reward_func(args, sample))

        assert results["math"] == good_response
        assert "code" not in results
        # Both domains were targeted; "code" failed and must later be filled
        # with the -inf fallback by _extract_teacher_data_from_responses.
        assert results["__target_domains__"] == ["math", "code"]

        # End-to-end: extraction fills the failed domain with -inf.
        sample.reward = results
        mopd._extract_teacher_data_from_responses(args, [sample])
        assert torch.isinf(sample.mopd_teacher_log_probs["code"]).all()
        assert torch.allclose(sample.mopd_teacher_log_probs["math"], torch.tensor([-0.5, -0.6]))

    def test_batch_mode_returns_one_dict_per_sample(self, monkeypatch):
        good_response = _sglang_response([None, [-0.5, 10, "a"]])

        def behavior(url, payload, call_index):
            return _FakeResponse(good_response)

        session = _FakeSession(behavior)
        monkeypatch.setattr(mopd.aiohttp, "ClientSession", lambda *a, **k: session)

        args = _make_rollout_args(
            mopd_teachers='[{"name": "math_teacher", "domain": "math"}]',
        )
        samples = [_make_sample(response_length=1), _make_sample(response_length=1)]
        results = asyncio.run(mopd.reward_func(args, samples))
        assert isinstance(results, list) and len(results) == 2
        assert all(r["math"] == good_response for r in results)

    def test_fetch_retries_on_transient_error(self, monkeypatch):
        good_response = _sglang_response([None, [-0.5, 10, "a"]])
        attempts = []

        def behavior(url, payload, call_index):
            attempts.append(call_index)
            if call_index == 1:
                return aiohttp.ClientConnectionError("transient")
            return _FakeResponse(good_response)

        session = _FakeSession(behavior)

        async def _no_sleep(delay):
            return None

        monkeypatch.setattr(asyncio, "sleep", _no_sleep)
        result = asyncio.run(mopd._fetch_teacher_logprobs(session, "math/generate", {}, max_retries=2, retry_delay=0.0))
        assert result == good_response
        assert len(attempts) == 2

    def test_fetch_does_not_retry_on_4xx(self, monkeypatch):
        attempts = []

        def behavior(url, payload, call_index):
            attempts.append(call_index)
            return _FakeResponse(status=400)

        session = _FakeSession(behavior)

        async def _no_sleep(delay):
            return None

        monkeypatch.setattr(asyncio, "sleep", _no_sleep)
        with pytest.raises(aiohttp.ClientResponseError):
            asyncio.run(mopd._fetch_teacher_logprobs(session, "math/generate", {}, max_retries=3, retry_delay=0.0))
        assert len(attempts) == 1


# ===========================================================================
# 8. combined_reward_func
# ===========================================================================
@pytest.mark.unit
class TestCombinedRewardFunc:
    def test_single_sample_collects_teacher_data_and_task_reward(self, monkeypatch):
        teacher_responses = {"math": {"meta_info": {}}, "__target_domains__": ["math"]}
        observed = {}

        async def fake_reward_func(args, sample, **kwargs):
            return teacher_responses

        async def fake_async_rm(args, sample, **kwargs):
            observed["custom_rm_path_during_rm"] = args.custom_rm_path
            return 0.75

        monkeypatch.setattr(mopd, "reward_func", fake_reward_func)
        monkeypatch.setattr(rm_hub_stub, "async_rm", fake_async_rm)

        args = _make_rollout_args(custom_rm_path="slime.rollout.mopd.combined_reward_func")
        sample = _make_sample()
        task_reward = asyncio.run(mopd.combined_reward_func(args, sample))

        assert task_reward == 0.75
        # custom_rm_path was temporarily cleared while the standard rm ran...
        assert observed["custom_rm_path_during_rm"] is None
        # ...and restored afterwards.
        assert args.custom_rm_path == "slime.rollout.mopd.combined_reward_func"
        assert sample.metadata[mopd._MOPD_TEACHER_RESPONSES_KEY] == teacher_responses

    def test_batch_mode(self, monkeypatch):
        async def fake_reward_func(args, samples, **kwargs):
            return [{"math": {}} for _ in samples]

        async def fake_batched_async_rm(args, samples, **kwargs):
            return [0.1 * i for i in range(len(samples))]

        monkeypatch.setattr(mopd, "reward_func", fake_reward_func)
        monkeypatch.setattr(rm_hub_stub, "batched_async_rm", fake_batched_async_rm)

        args = _make_rollout_args(custom_rm_path="slime.rollout.mopd.combined_reward_func")
        samples = [_make_sample(), _make_sample()]
        rewards = asyncio.run(mopd.combined_reward_func(args, samples))
        assert rewards == [0.0, 0.1]
        assert all(s.metadata[mopd._MOPD_TEACHER_RESPONSES_KEY] == {"math": {}} for s in samples)
        assert args.custom_rm_path == "slime.rollout.mopd.combined_reward_func"


# ===========================================================================
# 9. vocab_parallel_reverse_kl (process_group=None single-process path)
# ===========================================================================
@pytest.mark.unit
class TestVocabParallelReverseKL:
    def test_matches_naive_reference(self):
        torch.manual_seed(0)
        student = torch.randn(5, 17, dtype=torch.float64)
        teacher = torch.randn(5, 17, dtype=torch.float64)
        kl = vocab_parallel_reverse_kl(student, teacher, None)
        expected = _naive_reverse_kl(student, teacher)
        assert torch.allclose(kl, expected, atol=1e-10)

    def test_identical_distributions_give_zero(self):
        logits = torch.randn(4, 9, dtype=torch.float64)
        kl = vocab_parallel_reverse_kl(logits, logits.clone(), None)
        assert torch.allclose(kl, torch.zeros(4, dtype=torch.float64), atol=1e-12)

    def test_gradient_matches_analytic_formula(self):
        # d KL / d z_j = p_s(j) * (log p_s(j) - log p_t(j) - KL)
        torch.manual_seed(1)
        student = torch.randn(3, 11, dtype=torch.float64, requires_grad=True)
        teacher = torch.randn(3, 11, dtype=torch.float64)
        kl = vocab_parallel_reverse_kl(student, teacher, None)
        kl.sum().backward()

        s_logp = torch.log_softmax(student.detach(), dim=-1)
        t_logp = torch.log_softmax(teacher, dim=-1)
        p_s = s_logp.exp()
        expected_grad = p_s * (s_logp - t_logp - kl.detach().unsqueeze(-1))
        assert torch.allclose(student.grad, expected_grad, atol=1e-10)

    def test_teacher_receives_no_gradient(self):
        student = torch.randn(3, 7, requires_grad=True)
        teacher = torch.randn(3, 7, requires_grad=True)
        kl = vocab_parallel_reverse_kl(student, teacher, None)
        kl.sum().backward()
        assert student.grad is not None
        assert teacher.grad is None

    def test_gradcheck(self):
        torch.manual_seed(2)
        student = torch.randn(2, 6, dtype=torch.float64, requires_grad=True)
        teacher = torch.randn(2, 6, dtype=torch.float64)
        assert torch.autograd.gradcheck(lambda s: vocab_parallel_reverse_kl(s, teacher, None), (student,))


# ===========================================================================
# 10. vocab_parallel_topk_reverse_kl (process_group=None)
# ===========================================================================
@pytest.mark.unit
class TestVocabParallelTopKReverseKL:
    def _rand_inputs(self, R=4, V=13, k=5, seed=0):
        g = torch.Generator().manual_seed(seed)
        student = torch.randn(R, V, generator=g, dtype=torch.float64)
        teacher = torch.randn(R, V, generator=g, dtype=torch.float64)
        topk_indices = torch.topk(teacher, k, dim=-1).indices
        return student, teacher, topk_indices

    def test_full_coverage_degenerates_to_exact_kl_sglang_mode(self):
        # k == vocab_size with exact log-softmax values -> exact reverse KL.
        student, teacher, _ = self._rand_inputs(R=3, V=8, k=8)
        indices = torch.arange(8).unsqueeze(0).expand(3, 8)
        teacher_log_probs = torch.log_softmax(teacher, dim=-1)
        kl = vocab_parallel_topk_reverse_kl(student, teacher_log_probs, indices, 8, None, is_log_probs=True)
        expected = _naive_reverse_kl(student, teacher)
        assert torch.allclose(kl, expected, atol=1e-10)

    def test_full_coverage_degenerates_to_exact_kl_megatron_mode(self):
        # k == vocab_size with raw logits + exact full-vocab log_sum_exp.
        student, teacher, _ = self._rand_inputs(R=3, V=8, k=8)
        indices = torch.arange(8).unsqueeze(0).expand(3, 8)
        lse = torch.logsumexp(teacher, dim=-1)
        kl = vocab_parallel_topk_reverse_kl(
            student, teacher, indices, 8, None, is_log_probs=False, teacher_log_sum_exp=lse
        )
        expected = _naive_reverse_kl(student, teacher)
        assert torch.allclose(kl, expected, atol=1e-10)

    def test_megatron_mode_matches_reference_with_tail(self):
        # k < V: raw logits + exact teacher log_sum_exp -> exact per-token
        # teacher log-probs and exact tail mass -> matches the reference
        # top-k + tail decomposition of the true KL.
        student, teacher, topk_indices = self._rand_inputs(R=4, V=13, k=5)
        topk_logits = teacher.gather(-1, topk_indices)
        lse = torch.logsumexp(teacher, dim=-1)
        kl = vocab_parallel_topk_reverse_kl(
            student, topk_logits, topk_indices, 13, None, is_log_probs=False, teacher_log_sum_exp=lse
        )
        expected = _naive_topk_reverse_kl(student, teacher, topk_indices)
        assert torch.allclose(kl, expected, atol=1e-9)

    def test_sglang_mode_matches_reference_with_tail(self):
        # k < V: teacher top-k entries are exact log-probs from the full-vocab
        # softmax -> same reference decomposition.
        student, teacher, topk_indices = self._rand_inputs(R=4, V=13, k=5)
        t_logp = torch.log_softmax(teacher, dim=-1)
        topk_log_probs = t_logp.gather(-1, topk_indices)
        kl = vocab_parallel_topk_reverse_kl(student, topk_log_probs, topk_indices, 13, None, is_log_probs=True)
        expected = _naive_topk_reverse_kl(student, teacher, topk_indices)
        assert torch.allclose(kl, expected, atol=1e-9)

    def test_neg_inf_padding_is_masked_out(self):
        # Padding entries (-inf logit, index 0) must not contribute to the KL:
        # result equals the computation with only the valid entries.
        student, teacher, topk_indices = self._rand_inputs(R=4, V=13, k=3)
        t_logp = torch.log_softmax(teacher, dim=-1)
        valid_logits = t_logp.gather(-1, topk_indices)  # [R, 3]

        k_pad = 5
        padded_logits = torch.full((4, k_pad), NEG_INF, dtype=torch.float64)
        padded_indices = torch.zeros(4, k_pad, dtype=torch.long)
        padded_logits[:, :3] = valid_logits
        padded_indices[:, :3] = topk_indices

        kl_valid = vocab_parallel_topk_reverse_kl(student, valid_logits, topk_indices, 13, None, is_log_probs=True)
        kl_padded = vocab_parallel_topk_reverse_kl(student, padded_logits, padded_indices, 13, None, is_log_probs=True)
        assert torch.isfinite(kl_padded).all()
        assert torch.allclose(kl_padded, kl_valid, atol=1e-10)

    def test_explicit_valid_topk_mask(self):
        student, teacher, topk_indices = self._rand_inputs(R=4, V=13, k=3)
        valid_logits = torch.log_softmax(teacher, dim=-1).gather(-1, topk_indices)

        k_pad = 5
        padded_logits = torch.full((4, k_pad), NEG_INF, dtype=torch.float64)
        padded_indices = torch.zeros(4, k_pad, dtype=torch.long)
        padded_logits[:, :3] = valid_logits
        padded_indices[:, :3] = topk_indices

        mask = torch.zeros(4, k_pad, dtype=torch.bool)
        mask[:, :3] = True
        kl_masked = vocab_parallel_topk_reverse_kl(
            student, padded_logits, padded_indices, 13, None, valid_topk_mask=mask, is_log_probs=True
        )
        kl_valid = vocab_parallel_topk_reverse_kl(student, valid_logits, topk_indices, 13, None, is_log_probs=True)
        assert torch.allclose(kl_masked, kl_valid, atol=1e-10)

    def test_all_neg_inf_sglang_fallback_gives_zero_kl(self):
        # The -inf fallback fill produced by _extract_teacher_data_from_responses
        # for a failed teacher must contribute zero KL (documented NCCL-safety
        # behavior for the sglang path).
        student = torch.randn(3, 6, dtype=torch.float64)
        topk_logits = torch.full((3, 4), NEG_INF, dtype=torch.float64)
        topk_indices = torch.zeros(3, 4, dtype=torch.long)
        kl = vocab_parallel_topk_reverse_kl(student, topk_logits, topk_indices, 6, None, is_log_probs=True)
        assert torch.isfinite(kl).all()
        assert torch.allclose(kl, torch.zeros(3, dtype=torch.float64), atol=1e-10)

    def test_teacher_inputs_receive_no_gradient(self):
        student = torch.randn(2, 7, requires_grad=True)
        topk_logits = torch.randn(2, 3, requires_grad=True)
        topk_indices = torch.tensor([[0, 1, 2], [4, 5, 6]])
        kl = vocab_parallel_topk_reverse_kl(student, topk_logits, topk_indices, 7, None, is_log_probs=True)
        kl.sum().backward()
        assert student.grad is not None
        assert topk_logits.grad is None

    @pytest.mark.parametrize("is_log_probs", [True, False])
    def test_gradcheck(self, is_log_probs):
        torch.manual_seed(3)
        R, V, k = 2, 9, 4
        student = torch.randn(R, V, dtype=torch.float64, requires_grad=True)
        teacher = torch.randn(R, V, dtype=torch.float64)
        topk_indices = torch.topk(teacher, k, dim=-1).indices
        if is_log_probs:
            topk_logits = torch.log_softmax(teacher, dim=-1).gather(-1, topk_indices)
            lse = None
        else:
            topk_logits = teacher.gather(-1, topk_indices)
            lse = torch.logsumexp(teacher, dim=-1)

        def fn(s):
            return vocab_parallel_topk_reverse_kl(
                s, topk_logits, topk_indices, V, None, is_log_probs=is_log_probs, teacher_log_sum_exp=lse
            )

        assert torch.autograd.gradcheck(fn, (student,))


# ===========================================================================
# 11. apply_mopd_to_advantages (token_level)
# ===========================================================================
def _make_loss_args(**overrides) -> Namespace:
    defaults = dict(
        use_mopd=True,
        mopd_alpha=0.0,
        mopd_eps_low=0.2,
        mopd_eps_high=5.0,
        mopd_sampling_logprobs_key="rollout_log_probs",
        mopd_distill_type="token_level",
    )
    defaults.update(overrides)
    return Namespace(**defaults)


@pytest.mark.unit
class TestApplyMopdToAdvantages:
    def test_basic_reverse_kl_advantage(self):
        args = _make_loss_args(mopd_eps_low=0.0, mopd_eps_high=1000.0)
        student_log_probs = [torch.tensor([0.1, -0.2, 0.3])]
        teacher_log_probs = [torch.tensor([0.2, -0.1, 0.4])]
        rollout_log_probs = [torch.tensor([0.1, -0.2, 0.3])]  # == student -> IS weight 1
        advantages = [torch.tensor([1.0, 2.0, 3.0])]
        rollout_data = {
            "mopd_teacher_log_probs": {"math": teacher_log_probs},
            "rollout_log_probs": rollout_log_probs,
        }

        apply_mopd_to_advantages(args, rollout_data, advantages, student_log_probs)

        # reverse_kl = teacher - student = [0.1, 0.1, 0.1]; alpha=0 -> mopd_adv = reverse_kl
        assert torch.allclose(_mopd_field(rollout_data, "mopd_advantages")[0], torch.tensor([0.1, 0.1, 0.1]), atol=1e-6)
        assert torch.allclose(_mopd_field(rollout_data, "mopd_is_weights")[0], torch.ones(3), atol=1e-6)
        # mopd_reverse_kl is the pure reverse KL, without alpha * ORM advantage
        assert torch.allclose(
            _mopd_field(rollout_data, "mopd_reverse_kl").get("math")[0], torch.tensor([0.1, 0.1, 0.1]), atol=1e-6
        )
        # Base advantages are not modified in-place
        assert torch.allclose(advantages[0], torch.tensor([1.0, 2.0, 3.0]))

    def test_alpha_combines_orm_advantage(self):
        args = _make_loss_args(mopd_alpha=1.0, mopd_eps_low=0.0, mopd_eps_high=1000.0)
        student_log_probs = [torch.tensor([0.0, 0.0])]
        teacher_log_probs = [torch.tensor([1.0, 1.0])]
        rollout_data = {
            "mopd_teacher_log_probs": {"math": teacher_log_probs},
            "rollout_log_probs": [torch.tensor([0.0, 0.0])],
        }
        advantages = [torch.tensor([2.0, 3.0])]

        apply_mopd_to_advantages(args, rollout_data, advantages, student_log_probs)

        # mopd_adv = reverse_kl + alpha * ORM = 1.0 + [2.0, 3.0]
        assert torch.allclose(_mopd_field(rollout_data, "mopd_advantages")[0], torch.tensor([3.0, 4.0]), atol=1e-6)
        assert torch.allclose(
            _mopd_field(rollout_data, "mopd_reverse_kl").get("math")[0], torch.tensor([1.0, 1.0]), atol=1e-6
        )

    def test_is_weight_zeroed_below_eps_low(self):
        args = _make_loss_args(mopd_eps_low=0.5, mopd_eps_high=100.0)
        student_log_probs = [torch.tensor([-5.0, 0.0])]
        rollout_data = {
            "mopd_teacher_log_probs": {"math": [torch.tensor([0.0, 0.0])]},
            "rollout_log_probs": [torch.tensor([0.0, 0.0])],
        }
        advantages = [torch.tensor([1.0, 1.0])]

        apply_mopd_to_advantages(args, rollout_data, advantages, student_log_probs)

        is_weights = _mopd_field(rollout_data, "mopd_is_weights")[0]
        assert is_weights[0].item() == 0.0  # exp(-5) < eps_low
        assert is_weights[1].item() == 1.0

    def test_is_weight_zeroed_above_eps_high(self):
        args = _make_loss_args(mopd_eps_low=0.0, mopd_eps_high=2.0)
        student_log_probs = [torch.tensor([5.0, 0.0])]
        rollout_data = {
            "mopd_teacher_log_probs": {"math": [torch.tensor([0.0, 0.0])]},
            "rollout_log_probs": [torch.tensor([0.0, 0.0])],
        }
        advantages = [torch.tensor([1.0, 1.0])]

        apply_mopd_to_advantages(args, rollout_data, advantages, student_log_probs)

        is_weights = _mopd_field(rollout_data, "mopd_is_weights")[0]
        assert is_weights[0].item() == 0.0  # exp(5) > eps_high
        assert is_weights[1].item() == 1.0

    def test_multiple_teachers_averaged(self):
        args = _make_loss_args(mopd_eps_low=0.0, mopd_eps_high=1000.0)
        rollout_data = {
            "mopd_teacher_log_probs": {
                "math": [torch.tensor([1.0, 1.0])],
                "code": [torch.tensor([2.0, 2.0])],
            },
            "rollout_log_probs": [torch.tensor([0.0, 0.0])],
        }
        student_log_probs = [torch.tensor([0.0, 0.0])]
        advantages = [torch.tensor([0.0, 0.0])]

        apply_mopd_to_advantages(args, rollout_data, advantages, student_log_probs)

        assert torch.allclose(_mopd_field(rollout_data, "mopd_advantages")[0], torch.tensor([1.5, 1.5]), atol=1e-6)

    def test_per_sample_domain_routing_with_none(self):
        args = _make_loss_args(mopd_eps_low=0.0, mopd_eps_high=1000.0)
        rollout_data = {
            "mopd_teacher_log_probs": {
                "math": [torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0])],
                "code": [None, torch.tensor([2.0, 2.0])],  # sample 0 not routed to code
            },
            "rollout_log_probs": [torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])],
        }
        student_log_probs = [torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])]
        advantages = [torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])]

        apply_mopd_to_advantages(args, rollout_data, advantages, student_log_probs)

        assert torch.allclose(_mopd_field(rollout_data, "mopd_advantages")[0], torch.tensor([1.0, 1.0]), atol=1e-6)
        assert torch.allclose(_mopd_field(rollout_data, "mopd_advantages")[1], torch.tensor([1.5, 1.5]), atol=1e-6)
        # Per-teacher reverse KL logs zeros for the unrouted sample.
        assert torch.allclose(_mopd_field(rollout_data, "mopd_reverse_kl").get("code")[0], torch.zeros(2))

    def test_all_domains_none_gives_zeros(self):
        args = _make_loss_args()
        rollout_data = {
            "mopd_teacher_log_probs": {"math": [None], "code": [None]},
            "rollout_log_probs": [torch.tensor([0.0, 0.0])],
        }
        student_log_probs = [torch.tensor([0.0, 0.0])]
        advantages = [torch.tensor([1.0, 1.0])]

        apply_mopd_to_advantages(args, rollout_data, advantages, student_log_probs)

        assert torch.allclose(_mopd_field(rollout_data, "mopd_advantages")[0], torch.zeros(2))
        assert torch.allclose(_mopd_field(rollout_data, "mopd_is_weights")[0], torch.zeros(2))

    def test_neg_inf_fallback_treated_as_missing(self):
        # The -inf sentinel inserted by post_process_rewards for a failed teacher
        # must be excluded from the average, not contaminate it with -inf.
        args = _make_loss_args(mopd_eps_low=0.0, mopd_eps_high=1000.0)
        rollout_data = {
            "mopd_teacher_log_probs": {
                "math": [torch.tensor([1.0, 1.0])],
                "code": [torch.full((2,), NEG_INF)],  # failed teacher fallback
            },
            "rollout_log_probs": [torch.tensor([0.0, 0.0])],
        }
        student_log_probs = [torch.tensor([0.0, 0.0])]
        advantages = [torch.tensor([0.0, 0.0])]

        apply_mopd_to_advantages(args, rollout_data, advantages, student_log_probs)

        assert torch.isfinite(_mopd_field(rollout_data, "mopd_advantages")[0]).all()
        assert torch.allclose(_mopd_field(rollout_data, "mopd_advantages")[0], torch.tensor([1.0, 1.0]), atol=1e-6)

    def test_only_neg_inf_teacher_gives_zeros(self):
        args = _make_loss_args()
        rollout_data = {
            "mopd_teacher_log_probs": {"math": [torch.full((2,), NEG_INF)]},
            "rollout_log_probs": [torch.tensor([0.0, 0.0])],
        }
        student_log_probs = [torch.tensor([0.0, 0.0])]
        advantages = [torch.tensor([1.0, 1.0])]

        apply_mopd_to_advantages(args, rollout_data, advantages, student_log_probs)

        assert torch.allclose(_mopd_field(rollout_data, "mopd_advantages")[0], torch.zeros(2))
        assert torch.allclose(_mopd_field(rollout_data, "mopd_is_weights")[0], torch.zeros(2))

    def test_student_log_probs_none_returns_early(self):
        args = _make_loss_args()
        rollout_data = {"mopd_teacher_log_probs": {"math": [torch.tensor([1.0])]}}
        apply_mopd_to_advantages(args, rollout_data, [torch.tensor([1.0])], None)
        assert "mopd_advantages" not in rollout_data

    def test_missing_teacher_log_probs_raises(self):
        args = _make_loss_args()
        with pytest.raises(ValueError, match="mopd_teacher_log_probs"):
            apply_mopd_to_advantages(args, {}, [torch.tensor([1.0])], [torch.tensor([0.0])])

    def test_missing_sampling_log_probs_raises(self):
        args = _make_loss_args()
        rollout_data = {"mopd_teacher_log_probs": {"math": [torch.tensor([1.0])]}}
        with pytest.raises(ValueError, match="rollout_log_probs"):
            apply_mopd_to_advantages(args, rollout_data, [torch.tensor([1.0])], [torch.tensor([0.0])])

    def test_sampling_log_probs_falls_back_to_log_probs(self):
        args = _make_loss_args(mopd_eps_low=0.0, mopd_eps_high=1000.0)
        rollout_data = {
            "mopd_teacher_log_probs": {"math": [torch.tensor([1.0])]},
            "log_probs": [torch.tensor([0.0])],  # used when rollout_log_probs is absent
        }
        student_log_probs = [torch.tensor([0.0])]
        advantages = [torch.tensor([0.0])]

        apply_mopd_to_advantages(args, rollout_data, advantages, student_log_probs)
        assert torch.allclose(_mopd_field(rollout_data, "mopd_is_weights")[0], torch.ones(1))

    def test_custom_sampling_logprobs_key(self):
        args = _make_loss_args(mopd_sampling_logprobs_key="log_probs", mopd_eps_low=0.0, mopd_eps_high=1000.0)
        rollout_data = {
            "mopd_teacher_log_probs": {"math": [torch.tensor([1.0])]},
            "log_probs": [torch.tensor([-1.0])],
        }
        student_log_probs = [torch.tensor([0.0])]
        advantages = [torch.tensor([0.0])]

        apply_mopd_to_advantages(args, rollout_data, advantages, student_log_probs)
        assert torch.allclose(_mopd_field(rollout_data, "mopd_is_weights")[0], torch.tensor([math.exp(1.0)]), atol=1e-6)


# ===========================================================================
# 12. apply_mopd_full_vocab_to_loss
# ===========================================================================
def _make_fv_args(**overrides) -> Namespace:
    defaults = dict(
        use_mopd=True,
        mopd_distill_type="full_vocab",
        mopd_teacher_mode="megatron",
        mopd_alpha=0.0,
        mopd_eps_low=0.2,
        mopd_eps_high=5.0,
        mopd_sampling_logprobs_key="rollout_log_probs",
        vocab_size=8,
        mopd_topk_k=4,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


@pytest.mark.unit
class TestApplyMopdFullVocabToLoss:
    def test_kl_loss_matches_naive_full_vocab_kl(self):
        torch.manual_seed(0)
        R, V = 4, 8
        student_logits = [torch.randn(R, V)]
        teacher_logits = [torch.randn(R, V)]
        # sampling == current -> IS weight 1 for every token
        student_lp = [torch.zeros(R)]
        batch = {"rollout_log_probs": [torch.zeros(R)]}
        loss_masks = [torch.ones(R)]
        args = _make_fv_args(vocab_size=V)

        kl_loss, metrics = apply_mopd_full_vocab_to_loss(
            args,
            batch,
            student_logits,
            {"math": teacher_logits},
            loss_masks,
            _mean_reducer,
            current_log_probs=student_lp,
        )

        expected = _naive_reverse_kl(student_logits[0], teacher_logits[0]).mean()
        assert torch.allclose(kl_loss, expected, atol=1e-6)
        assert set(metrics) >= {"mopd_fv_kl", "mopd_is_weight_mean", "mopd_is_nonzero_frac", "mopd_fv_kl/math"}
        assert torch.allclose(metrics["mopd_is_nonzero_frac"], torch.tensor(1.0))

    def test_multiple_domains_averaged(self):
        torch.manual_seed(1)
        R, V = 3, 8
        student_logits = [torch.randn(R, V)]
        t_math = torch.randn(R, V)
        t_code = torch.randn(R, V)
        batch = {"rollout_log_probs": [torch.zeros(R)]}
        loss_masks = [torch.ones(R)]
        args = _make_fv_args(vocab_size=V)

        kl_loss, metrics = apply_mopd_full_vocab_to_loss(
            args,
            batch,
            student_logits,
            {"math": [t_math], "code": [t_code]},
            loss_masks,
            _mean_reducer,
            current_log_probs=[torch.zeros(R)],
        )

        kl_math = _naive_reverse_kl(student_logits[0], t_math)
        kl_code = _naive_reverse_kl(student_logits[0], t_code)
        expected = ((kl_math + kl_code) / 2).mean()
        assert torch.allclose(kl_loss, expected, atol=1e-6)
        assert torch.allclose(metrics["mopd_fv_kl/math"], kl_math.mean(), atol=1e-6)
        assert torch.allclose(metrics["mopd_fv_kl/code"], kl_code.mean(), atol=1e-6)

    def test_none_domain_excluded_and_zero_metric(self):
        torch.manual_seed(2)
        R, V = 3, 8
        student_logits = [torch.randn(R, V)]
        t_math = torch.randn(R, V)
        batch = {"rollout_log_probs": [torch.zeros(R)]}
        loss_masks = [torch.ones(R)]
        args = _make_fv_args(vocab_size=V)

        kl_loss, metrics = apply_mopd_full_vocab_to_loss(
            args,
            batch,
            student_logits,
            {"math": [t_math], "code": [None]},
            loss_masks,
            _mean_reducer,
            current_log_probs=[torch.zeros(R)],
        )

        expected = _naive_reverse_kl(student_logits[0], t_math).mean()
        assert torch.allclose(kl_loss, expected, atol=1e-6)
        assert metrics["mopd_fv_kl/code"].item() == 0.0

    def test_is_gating_zeroes_out_of_range_tokens(self):
        torch.manual_seed(3)
        R, V = 3, 8
        student_logits = [torch.randn(R, V)]
        teacher_logits = [torch.randn(R, V)]
        # token 0: exp(log(10)) = 10 > eps_high -> zeroed
        # token 1: exp(log(0.1)) = 0.1 < eps_low -> zeroed
        # token 2: weight 1 -> kept
        delta = torch.tensor([math.log(10.0), math.log(0.1), 0.0])
        batch = {"rollout_log_probs": [torch.zeros(R)]}
        current_log_probs = [delta]
        loss_masks = [torch.ones(R)]
        args = _make_fv_args(vocab_size=V)

        kl_loss, metrics = apply_mopd_full_vocab_to_loss(
            args,
            batch,
            student_logits,
            {"math": teacher_logits},
            loss_masks,
            _mean_reducer,
            current_log_probs=current_log_probs,
        )

        kl = _naive_reverse_kl(student_logits[0], teacher_logits[0])
        expected = kl[2:].sum() / 3  # divided by mask sum, not by kept-token count
        assert torch.allclose(kl_loss, expected, atol=1e-6)
        assert torch.allclose(metrics["mopd_is_nonzero_frac"], torch.tensor(1.0 / 3.0), atol=1e-6)

    def test_is_weight_falls_back_to_batch_log_probs(self):
        torch.manual_seed(4)
        R, V = 2, 8
        student_logits = [torch.randn(R, V)]
        teacher_logits = [torch.randn(R, V)]
        batch = {"log_probs": [torch.zeros(R)]}  # no rollout_log_probs, no current_log_probs
        loss_masks = [torch.ones(R)]
        args = _make_fv_args(vocab_size=V)

        kl_loss, _ = apply_mopd_full_vocab_to_loss(
            args, batch, student_logits, {"math": teacher_logits}, loss_masks, _mean_reducer
        )
        expected = _naive_reverse_kl(student_logits[0], teacher_logits[0]).mean()
        assert torch.allclose(kl_loss, expected, atol=1e-6)

    def test_missing_sampling_log_probs_raises(self):
        args = _make_fv_args()
        with pytest.raises(ValueError, match="importance sampling"):
            apply_mopd_full_vocab_to_loss(
                args, {}, [torch.randn(2, 8)], {"math": [torch.randn(2, 8)]}, [torch.ones(2)], _mean_reducer
            )

    def test_gradient_flows_to_student_only(self):
        torch.manual_seed(5)
        R, V = 3, 8
        student_logits = [torch.randn(R, V, requires_grad=True)]
        teacher_logits = [torch.randn(R, V, requires_grad=True)]
        batch = {"rollout_log_probs": [torch.zeros(R)]}
        loss_masks = [torch.ones(R)]
        args = _make_fv_args(vocab_size=V)

        kl_loss, _ = apply_mopd_full_vocab_to_loss(
            args,
            batch,
            student_logits,
            {"math": teacher_logits},
            loss_masks,
            _mean_reducer,
            current_log_probs=[torch.zeros(R)],
        )
        kl_loss.backward()
        assert student_logits[0].grad is not None
        assert torch.isfinite(student_logits[0].grad).all()
        assert teacher_logits[0].grad is None


# ===========================================================================
# 13. apply_mopd_topk_to_loss
# ===========================================================================
@pytest.mark.unit
class TestApplyMopdTopKToLoss:
    def _identity_topk(self, teacher_logits):
        R, V = teacher_logits.shape
        indices = torch.arange(V).unsqueeze(0).expand(R, V).contiguous()
        return indices

    def test_full_coverage_matches_exact_kl_megatron_mode(self):
        torch.manual_seed(10)
        R, V = 4, 8
        student_logits = [torch.randn(R, V)]
        teacher = torch.randn(R, V)
        topk_indices = self._identity_topk(teacher)
        batch = {"rollout_log_probs": [torch.zeros(R)]}
        loss_masks = [torch.ones(R)]
        args = _make_fv_args(vocab_size=V, mopd_distill_type="top_k", mopd_teacher_mode="megatron", mopd_topk_k=V)

        kl_loss, metrics = apply_mopd_topk_to_loss(
            args,
            batch,
            student_logits,
            {"math": [teacher]},
            {"math": [topk_indices]},
            loss_masks,
            _mean_reducer,
            current_log_probs=[torch.zeros(R)],
            teacher_topk_log_sum_exp_per_domain={"math": [torch.logsumexp(teacher, dim=-1)]},
        )

        expected = _naive_reverse_kl(student_logits[0], teacher).mean()
        assert torch.allclose(kl_loss, expected, atol=1e-6)
        assert "mopd_topk_kl/math" in metrics

    def test_full_coverage_matches_exact_kl_sglang_mode(self):
        torch.manual_seed(11)
        R, V = 4, 8
        student_logits = [torch.randn(R, V)]
        teacher = torch.randn(R, V)
        topk_indices = self._identity_topk(teacher)
        topk_log_probs = torch.log_softmax(teacher, dim=-1)
        batch = {"rollout_log_probs": [torch.zeros(R)]}
        loss_masks = [torch.ones(R)]
        args = _make_fv_args(vocab_size=V, mopd_distill_type="top_k", mopd_teacher_mode="sglang", mopd_topk_k=V)

        kl_loss, _ = apply_mopd_topk_to_loss(
            args,
            batch,
            student_logits,
            {"math": [topk_log_probs]},
            {"math": [topk_indices]},
            loss_masks,
            _mean_reducer,
            current_log_probs=[torch.zeros(R)],
        )

        expected = _naive_reverse_kl(student_logits[0], teacher).mean()
        assert torch.allclose(kl_loss, expected, atol=1e-6)

    def test_partial_topk_matches_reference(self):
        torch.manual_seed(12)
        R, V, k = 3, 10, 4
        student_logits = [torch.randn(R, V)]
        teacher = torch.randn(R, V)
        topk_indices = torch.topk(teacher, k, dim=-1).indices
        topk_logits = teacher.gather(-1, topk_indices)
        batch = {"rollout_log_probs": [torch.zeros(R)]}
        loss_masks = [torch.ones(R)]
        args = _make_fv_args(vocab_size=V, mopd_distill_type="top_k", mopd_teacher_mode="megatron", mopd_topk_k=k)

        kl_loss, _ = apply_mopd_topk_to_loss(
            args,
            batch,
            student_logits,
            {"math": [topk_logits]},
            {"math": [topk_indices]},
            loss_masks,
            _mean_reducer,
            current_log_probs=[torch.zeros(R)],
            teacher_topk_log_sum_exp_per_domain={"math": [torch.logsumexp(teacher, dim=-1)]},
        )

        expected = _naive_topk_reverse_kl(student_logits[0], teacher, topk_indices).mean()
        assert torch.allclose(kl_loss, expected, atol=1e-6)

    def test_neg_inf_padding_entries_masked(self):
        torch.manual_seed(13)
        R, V, k = 3, 10, 4
        student_logits = [torch.randn(R, V)]
        teacher = torch.randn(R, V)
        topk_indices = torch.topk(teacher, k, dim=-1).indices
        t_logp = torch.log_softmax(teacher, dim=-1)
        valid_logits = t_logp.gather(-1, topk_indices)

        # Pad to k+2 with -inf / index 0 entries (SGLang short-position fill).
        padded_logits = torch.full((R, k + 2), NEG_INF)
        padded_indices = torch.zeros(R, k + 2, dtype=torch.long)
        padded_logits[:, :k] = valid_logits
        padded_indices[:, :k] = topk_indices

        batch = {"rollout_log_probs": [torch.zeros(R)]}
        loss_masks = [torch.ones(R)]
        args = _make_fv_args(vocab_size=V, mopd_distill_type="top_k", mopd_teacher_mode="sglang", mopd_topk_k=k + 2)

        kl_loss, _ = apply_mopd_topk_to_loss(
            args,
            batch,
            student_logits,
            {"math": [padded_logits]},
            {"math": [padded_indices]},
            loss_masks,
            _mean_reducer,
            current_log_probs=[torch.zeros(R)],
        )

        expected = _naive_topk_reverse_kl(student_logits[0], teacher, topk_indices).mean()
        assert torch.isfinite(kl_loss)
        assert torch.allclose(kl_loss, expected, atol=1e-6)

    def test_all_neg_inf_teacher_contributes_zero_kl(self):
        # Failed-teacher fallback (-inf fill) must not poison the loss.
        torch.manual_seed(14)
        R, V, k = 3, 8, 4
        student_logits = [torch.randn(R, V)]
        batch = {"rollout_log_probs": [torch.zeros(R)]}
        loss_masks = [torch.ones(R)]
        args = _make_fv_args(vocab_size=V, mopd_distill_type="top_k", mopd_teacher_mode="sglang", mopd_topk_k=k)

        kl_loss, metrics = apply_mopd_topk_to_loss(
            args,
            batch,
            student_logits,
            {"math": [torch.full((R, k), NEG_INF)]},
            {"math": [torch.zeros(R, k, dtype=torch.long)]},
            loss_masks,
            _mean_reducer,
            current_log_probs=[torch.zeros(R)],
        )
        assert torch.isfinite(kl_loss)
        assert torch.allclose(kl_loss, torch.tensor(0.0), atol=1e-8)

    def test_is_gating_zeroes_out_of_range_tokens(self):
        torch.manual_seed(15)
        R, V = 3, 8
        student_logits = [torch.randn(R, V)]
        teacher = torch.randn(R, V)
        topk_indices = self._identity_topk(teacher)
        delta = torch.tensor([math.log(10.0), 0.0, 0.0])  # token 0 zeroed (weight 10 > eps_high)
        batch = {"rollout_log_probs": [torch.zeros(R)]}
        loss_masks = [torch.ones(R)]
        args = _make_fv_args(vocab_size=V, mopd_distill_type="top_k", mopd_teacher_mode="megatron", mopd_topk_k=V)

        kl_loss, metrics = apply_mopd_topk_to_loss(
            args,
            batch,
            student_logits,
            {"math": [teacher]},
            {"math": [topk_indices]},
            loss_masks,
            _mean_reducer,
            current_log_probs=[delta],
            teacher_topk_log_sum_exp_per_domain={"math": [torch.logsumexp(teacher, dim=-1)]},
        )

        kl = _naive_reverse_kl(student_logits[0], teacher)
        expected = kl[1:].sum() / 3
        assert torch.allclose(kl_loss, expected, atol=1e-6)
        assert torch.allclose(metrics["mopd_is_nonzero_frac"], torch.tensor(2.0 / 3.0), atol=1e-6)

    def test_missing_sampling_log_probs_raises(self):
        args = _make_fv_args(mopd_distill_type="top_k")
        with pytest.raises(ValueError, match="importance sampling"):
            apply_mopd_topk_to_loss(
                args,
                {},
                [torch.randn(2, 8)],
                {"math": [torch.randn(2, 4)]},
                {"math": [torch.zeros(2, 4, dtype=torch.long)]},
                [torch.ones(2)],
                _mean_reducer,
            )

    def test_gradient_flows_to_student(self):
        torch.manual_seed(16)
        R, V = 3, 8
        student_logits = [torch.randn(R, V, requires_grad=True)]
        teacher = torch.randn(R, V)
        topk_indices = self._identity_topk(teacher)
        batch = {"rollout_log_probs": [torch.zeros(R)]}
        loss_masks = [torch.ones(R)]
        args = _make_fv_args(vocab_size=V, mopd_distill_type="top_k", mopd_teacher_mode="megatron", mopd_topk_k=V)

        kl_loss, _ = apply_mopd_topk_to_loss(
            args,
            batch,
            student_logits,
            {"math": [teacher]},
            {"math": [topk_indices]},
            loss_masks,
            _mean_reducer,
            current_log_probs=[torch.zeros(R)],
            teacher_topk_log_sum_exp_per_domain={"math": [torch.logsumexp(teacher, dim=-1)]},
        )
        kl_loss.backward()
        assert student_logits[0].grad is not None
        assert torch.isfinite(student_logits[0].grad).all()
        assert student_logits[0].grad.abs().sum() > 0


# ===========================================================================
# 14. compute_advantages_and_returns MOPD integration
# ===========================================================================
@pytest.mark.unit
class TestComputeAdvantagesMopdIntegration:
    def _base_rollout_data(self):
        return {
            "log_probs": [torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])],
            "rollout_log_probs": [torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])],
            "ref_log_probs": None,
            "rewards": [1.0, 0.0],
            "values": None,
            "response_lengths": [2, 2],
            "loss_masks": [torch.ones(2), torch.ones(2)],
            "total_lengths": [4, 4],
            "mopd_teacher_log_probs": {
                "math": [torch.tensor([0.5, 0.5]), torch.tensor([0.5, 0.5])],
            },
        }

    def _args(self, distill_type="token_level"):
        return Namespace(
            kl_coef=0.0,
            use_rollout_logprobs=False,
            custom_advantage_function_path=None,
            advantage_estimator="grpo",
            use_opd=False,
            use_mopd=True,
            mopd_distill_type=distill_type,
            mopd_alpha=0.0,
            mopd_eps_low=0.0,
            mopd_eps_high=1000.0,
            mopd_sampling_logprobs_key="rollout_log_probs",
            normalize_advantages=False,
        )

    def test_token_level_populates_mopd_advantages(self):
        rollout_data = self._base_rollout_data()
        args = self._args("token_level")
        compute_advantages_and_returns(args, rollout_data)
        # reverse_kl = teacher - student = 0.5 everywhere
        assert torch.allclose(_mopd_field(rollout_data, "mopd_advantages")[0], torch.tensor([0.5, 0.5]), atol=1e-6)
        assert "mopd_is_weights" in rollout_data
        assert "advantages" in rollout_data  # GRPO returns still computed

    def test_logits_based_distill_types_skip_token_level_mopd(self):
        for distill_type in ("full_vocab", "top_k"):
            rollout_data = self._base_rollout_data()
            args = self._args(distill_type)
            compute_advantages_and_returns(args, rollout_data)
            assert "mopd_advantages" not in rollout_data


# ===========================================================================
# 15. DataIterator recursive dict slicing
# ===========================================================================
@pytest.mark.unit
class TestDataIteratorNestedDict:
    def test_nested_dict_fields_sliced_per_domain(self):
        # MOPD fields are dict[domain, list[sample]]: slicing them like a plain
        # list (vals[i]) would raise KeyError: 0 on the domain string keys.
        rollout_data = {
            "mopd_teacher_topk_logits": {
                "math": [f"m{i}" for i in range(4)],
                "code": [f"c{i}" for i in range(4)],
            },
            "log_probs": [f"lp{i}" for i in range(4)],
            "optional_field": None,
        }
        iterator = DataIterator(rollout_data, micro_batch_indices=[[0, 2], [1, 3]])

        batch = iterator.get_next(["mopd_teacher_topk_logits", "log_probs", "optional_field", "missing"])
        assert batch["mopd_teacher_topk_logits"]["math"] == ["m0", "m2"]
        assert batch["mopd_teacher_topk_logits"]["code"] == ["c0", "c2"]
        assert batch["log_probs"] == ["lp0", "lp2"]
        assert batch["optional_field"] is None
        assert batch["missing"] is None

        batch2 = iterator.get_next(["mopd_teacher_topk_logits", "log_probs"])
        assert batch2["mopd_teacher_topk_logits"]["math"] == ["m1", "m3"]
        assert batch2["log_probs"] == ["lp1", "lp3"]

    def test_reset_restarts_iteration(self):
        rollout_data = {"x": [1, 2, 3]}
        iterator = DataIterator(rollout_data, micro_batch_indices=[[0], [1]])
        iterator.get_next(["x"])
        iterator.get_next(["x"])
        assert iterator.offset == 2
        iterator.reset()
        assert iterator.offset == 0
        assert iterator.get_next(["x"])["x"] == [1]

    def test_deeply_nested_dict(self):
        rollout_data = {"nested": {"a": {"b": [10, 20]}}}
        iterator = DataIterator(rollout_data, micro_batch_indices=[[1]])
        batch = iterator.get_next(["nested"])
        assert batch["nested"]["a"]["b"] == [20]


# ===========================================================================
# 16. Sample MOPD fields
# ===========================================================================
@pytest.mark.unit
class TestSampleMopdFields:
    def test_mopd_fields_default_to_none(self):
        sample = Sample()
        assert sample.mopd_teacher_log_probs is None
        assert sample.mopd_teacher_topk_logits is None
        assert sample.mopd_teacher_topk_indices is None
        assert sample.mopd_teacher_fv_logits is None

    def test_mopd_fields_roundtrip_through_dict(self):
        sample = Sample(response="hello", response_length=1)
        sample.mopd_teacher_log_probs = {"math": [0.1, 0.2]}
        sample.mopd_teacher_topk_logits = {"math": [[-0.1, -1.0]]}
        sample.mopd_teacher_topk_indices = {"math": [[5, 7]]}

        data = sample.to_dict()
        assert data["mopd_teacher_log_probs"] == {"math": [0.1, 0.2]}
        assert data["mopd_teacher_topk_logits"] == {"math": [[-0.1, -1.0]]}
        assert data["mopd_teacher_topk_indices"] == {"math": [[5, 7]]}

        restored = Sample.from_dict(data)
        assert restored.mopd_teacher_log_probs == {"math": [0.1, 0.2]}
        assert restored.mopd_teacher_topk_logits == {"math": [[-0.1, -1.0]]}
        assert restored.mopd_teacher_topk_indices == {"math": [[5, 7]]}


# ===========================================================================
# 17. slime_validate_args MOPD validation
# ===========================================================================
def _make_validate_args(**overrides) -> Namespace:
    """Minimal args namespace that passes all non-MOPD validation branches."""
    defaults = dict(
        # eval
        eval_config=None,
        eval_prompt_data=None,
        eval_interval=None,
        eval_reward_key=None,
        eval_function_path=None,
        eval_max_context_len=None,
        # router / profiler
        use_slime_router=False,
        npu_profile_actor_global_step_start=0,
        npu_profile_actor_global_step_end=-1,
        npu_profile_actor_ranks=None,
        npu_profile_rollout_global_step_start=0,
        npu_profile_rollout_global_step_end=-1,
        npu_profile_rollout_ranks=None,
        npu_profile_rollout_token_start=None,
        npu_profile_rollout_token_end=None,
        # kl / ref
        kl_coef=0.0,
        use_kl_loss=False,
        kl_loss_coef=0.0,
        ref_load=None,
        ref_ckpt_step=None,
        # opd
        use_opd=False,
        opd_type=None,
        opd_teacher_load=None,
        # mopd
        use_mopd=False,
        mopd_teachers=None,
        mopd_teacher_loads=None,
        mopd_teacher_ckpt_steps=None,
        mopd_alpha=0.0,
        mopd_eps_low=0.2,
        mopd_eps_high=5.0,
        mopd_distill_type="token_level",
        mopd_topk_k=1024,
        mopd_teacher_mode=None,
        # reward
        rm_type=None,
        custom_rm_path=None,
        custom_reward_post_process_path=None,
        reward_key=None,
        # checkpoint / save
        megatron_to_hf_mode="raw",
        load=None,
        hf_checkpoint=None,
        save=None,
        save_interval=None,
        # algo
        advantage_estimator="grpo",
        normalize_advantages=False,
        use_rollout_logprobs=False,
        use_tis=False,
        get_mismatch_metrics=False,
        custom_tis_function_path=None,
        eps_clip=0.2,
        eps_clip_high=None,
        # batch / data
        use_dynamic_batch_size=False,
        max_tokens_per_gpu=None,
        log_probs_max_tokens_per_gpu=None,
        rollout_batch_size=8,
        n_samples_per_prompt=1,
        global_batch_size=None,
        num_steps_per_rollout=None,
        over_sampling_batch_size=None,
        num_epoch=None,
        num_rollout=1,
        rollout_global_dataset=True,
        # cluster / debug
        actor_num_gpus_per_node=8,
        actor_num_nodes=1,
        offload=False,
        debug_rollout_only=False,
        debug_train_only=False,
        colocate=False,
        rollout_num_gpus=None,
        train_memory_margin_bytes=0,
        offload_train=None,
        offload_rollout=None,
        dump_details=None,
        load_debug_rollout_data=None,
        rollout_function_path="slime.rollout.sglang_rollout.generate_rollout",
        # misc
        enable_mtp_training=False,
        use_rollout_routing_replay=False,
        custom_config_path=None,
        rollout_max_context_len=None,
        rollout_max_prompt_len=None,
        qkv_format="thd",
        train_backend="megatron",
        only_train_params_name_list=None,
        freeze_params_name_list=None,
        update_weight_mode="full",
        update_weight_transport="nccl",
        update_weight_delta_dir=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


_TEACHERS_ONE = '[{"name": "math_teacher", "domain": "math"}]'
_TEACHERS_TWO = '[{"name": "math_teacher", "domain": "math"}, {"name": "code_teacher", "domain": "code"}]'


@pytest.mark.unit
class TestMopdArgValidation:
    def test_mopd_and_opd_mutually_exclusive(self):
        # opd_type must be set so the earlier OPD validation branch passes and
        # the MOPD mutual-exclusion check is reached.
        args = _make_validate_args(use_mopd=True, use_opd=True, opd_type="sglang", mopd_teachers=_TEACHERS_ONE)
        with pytest.raises(ValueError, match="mutually exclusive"):
            slime_validate_args(args)

    def test_mopd_requires_teachers(self, monkeypatch):
        monkeypatch.delenv("MOPD_TEACHERS_JSON", raising=False)
        args = _make_validate_args(use_mopd=True, mopd_teachers=None)
        with pytest.raises(ValueError, match="mopd-teachers is required"):
            slime_validate_args(args)

    def test_mopd_teachers_from_env_var(self, monkeypatch):
        monkeypatch.setenv("MOPD_TEACHERS_JSON", _TEACHERS_ONE)
        args = _make_validate_args(use_mopd=True, mopd_teachers=None)
        slime_validate_args(args)
        assert args._mopd_teachers_parsed == [{"name": "math_teacher", "domain": "math"}]

    def test_mopd_teachers_invalid_json(self):
        args = _make_validate_args(use_mopd=True, mopd_teachers="{not json")
        with pytest.raises(ValueError, match="must be valid JSON"):
            slime_validate_args(args)

    def test_mopd_teachers_empty_list(self):
        args = _make_validate_args(use_mopd=True, mopd_teachers="[]")
        with pytest.raises(ValueError, match="non-empty JSON list"):
            slime_validate_args(args)

    def test_mopd_teacher_entry_must_be_dict(self):
        args = _make_validate_args(use_mopd=True, mopd_teachers='["math"]')
        with pytest.raises(ValueError, match="must be a dict"):
            slime_validate_args(args)

    def test_mopd_teacher_requires_domain_key(self):
        args = _make_validate_args(use_mopd=True, mopd_teachers='[{"name": "t1"}]')
        with pytest.raises(ValueError, match="must contain a 'domain' key"):
            slime_validate_args(args)

    def test_mopd_duplicate_domain_raises(self):
        args = _make_validate_args(
            use_mopd=True,
            mopd_teachers='[{"name": "t1", "domain": "math"}, {"name": "t2", "domain": "math"}]',
        )
        with pytest.raises(ValueError, match="duplicate domain"):
            slime_validate_args(args)

    def test_megatron_full_vocab_requires_teacher_loads(self):
        args = _make_validate_args(
            use_mopd=True,
            mopd_teachers=_TEACHERS_ONE,
            mopd_teacher_mode="megatron",
            mopd_distill_type="full_vocab",
            mopd_teacher_loads=None,
        )
        with pytest.raises(ValueError, match="requires --mopd-teacher-loads"):
            slime_validate_args(args)

    def test_megatron_top_k_requires_teacher_loads(self):
        args = _make_validate_args(
            use_mopd=True,
            mopd_teachers=_TEACHERS_ONE,
            mopd_teacher_mode="megatron",
            mopd_distill_type="top_k",
            mopd_teacher_loads=None,
        )
        with pytest.raises(ValueError, match="requires --mopd-teacher-loads"):
            slime_validate_args(args)

    def test_sglang_full_vocab_not_supported(self):
        args = _make_validate_args(
            use_mopd=True,
            mopd_teachers=_TEACHERS_ONE,
            mopd_teacher_mode="sglang",
            mopd_distill_type="full_vocab",
        )
        with pytest.raises(ValueError, match="not supported with --mopd-teacher-mode=sglang"):
            slime_validate_args(args)

    def test_sglang_mode_rejects_teacher_loads(self, tmp_path):
        args = _make_validate_args(
            use_mopd=True,
            mopd_teachers=_TEACHERS_ONE,
            mopd_teacher_mode="sglang",
            mopd_teacher_loads=[str(tmp_path)],
        )
        with pytest.raises(ValueError, match="incompatible with --mopd-teacher-loads"):
            slime_validate_args(args)

    def test_teacher_loads_count_mismatch(self, tmp_path):
        args = _make_validate_args(
            use_mopd=True,
            mopd_teachers=_TEACHERS_TWO,
            mopd_teacher_loads=[str(tmp_path)],  # 1 path vs 2 teachers
        )
        with pytest.raises(ValueError, match="must match"):
            slime_validate_args(args)

    def test_teacher_loads_missing_path_raises(self):
        args = _make_validate_args(
            use_mopd=True,
            mopd_teachers=_TEACHERS_ONE,
            mopd_teacher_loads=["/nonexistent/mopd-teacher-path"],
        )
        with pytest.raises(FileNotFoundError, match="does not exist"):
            slime_validate_args(args)

    def test_teacher_ckpt_steps_require_loads(self):
        args = _make_validate_args(
            use_mopd=True,
            mopd_teachers=_TEACHERS_ONE,
            mopd_teacher_ckpt_steps=[3],
        )
        with pytest.raises(ValueError, match="requires --mopd-teacher-loads"):
            slime_validate_args(args)

    def test_negative_eps_low_raises(self):
        args = _make_validate_args(use_mopd=True, mopd_teachers=_TEACHERS_ONE, mopd_eps_low=-0.1)
        with pytest.raises(ValueError, match="mopd-eps-low"):
            slime_validate_args(args)

    def test_eps_high_must_exceed_eps_low(self):
        args = _make_validate_args(use_mopd=True, mopd_teachers=_TEACHERS_ONE, mopd_eps_low=5.0, mopd_eps_high=5.0)
        with pytest.raises(ValueError, match="mopd-eps-high"):
            slime_validate_args(args)

    def test_top_k_requires_positive_k(self):
        args = _make_validate_args(
            use_mopd=True,
            mopd_teachers=_TEACHERS_ONE,
            mopd_teacher_mode="sglang",
            mopd_distill_type="top_k",
            mopd_topk_k=0,
        )
        with pytest.raises(ValueError, match="mopd-topk-k"):
            slime_validate_args(args)

    def test_teacher_mode_defaults_from_loads(self):
        # No teacher loads -> default sglang
        args = _make_validate_args(use_mopd=True, mopd_teachers=_TEACHERS_ONE)
        slime_validate_args(args)
        assert args.mopd_teacher_mode == "sglang"

    def test_teacher_mode_defaults_to_megatron_when_loads_set(self, tmp_path):
        ckpt = tmp_path / "teacher"
        ckpt.mkdir()
        (ckpt / "latest_checkpointed_iteration.txt").write_text("1")
        args = _make_validate_args(use_mopd=True, mopd_teachers=_TEACHERS_ONE, mopd_teacher_loads=[str(ckpt)])
        slime_validate_args(args)
        assert args.mopd_teacher_mode == "megatron"

    def test_sglang_pure_distillation_autoconfigures_rm_paths(self):
        args = _make_validate_args(
            use_mopd=True,
            mopd_teachers=_TEACHERS_ONE,
            mopd_teacher_mode="sglang",
            mopd_alpha=0.0,
        )
        slime_validate_args(args)
        assert args.custom_rm_path == "slime.rollout.mopd.reward_func"
        assert args.custom_reward_post_process_path == "slime.rollout.mopd.post_process_rewards"

    def test_sglang_combined_mode_autoconfigures_and_requires_rm_type(self):
        args = _make_validate_args(
            use_mopd=True,
            mopd_teachers=_TEACHERS_ONE,
            mopd_teacher_mode="sglang",
            mopd_alpha=0.5,
            rm_type=None,
        )
        with pytest.raises(ValueError, match="requires --rm-type"):
            slime_validate_args(args)

    def test_sglang_combined_mode_with_rm_type_ok(self):
        args = _make_validate_args(
            use_mopd=True,
            mopd_teachers=_TEACHERS_ONE,
            mopd_teacher_mode="sglang",
            mopd_alpha=0.5,
            rm_type="math",
        )
        slime_validate_args(args)
        assert args.custom_rm_path == "slime.rollout.mopd.combined_reward_func"
        assert args.custom_reward_post_process_path == "slime.rollout.mopd.combined_post_process_rewards"

    def test_alpha_zero_megatron_defaults_rm_type_to_zero(self, tmp_path):
        ckpt = tmp_path / "teacher"
        ckpt.mkdir()
        (ckpt / "latest_checkpointed_iteration.txt").write_text("1")
        args = _make_validate_args(
            use_mopd=True,
            mopd_teachers=_TEACHERS_ONE,
            mopd_teacher_loads=[str(ckpt)],
            mopd_alpha=0.0,
            rm_type=None,
        )
        slime_validate_args(args)
        assert args.rm_type == "zero"

    def test_megatron_full_vocab_happy_path(self, tmp_path):
        ckpt = tmp_path / "teacher"
        ckpt.mkdir()
        (ckpt / "latest_checkpointed_iteration.txt").write_text("1")
        args = _make_validate_args(
            use_mopd=True,
            mopd_teachers=_TEACHERS_TWO,
            mopd_teacher_loads=[str(ckpt), str(ckpt)],
            mopd_teacher_mode="megatron",
            mopd_distill_type="full_vocab",
        )
        slime_validate_args(args)
        assert [t["domain"] for t in args._mopd_teachers_parsed] == ["math", "code"]

    def test_sglang_top_k_happy_path(self):
        args = _make_validate_args(
            use_mopd=True,
            mopd_teachers=_TEACHERS_TWO,
            mopd_teacher_mode="sglang",
            mopd_distill_type="top_k",
            mopd_topk_k=128,
        )
        slime_validate_args(args)
        assert [t["domain"] for t in args._mopd_teachers_parsed] == ["math", "code"]

    def test_teacher_loads_without_use_mopd_raises(self):
        args = _make_validate_args(use_mopd=False, mopd_teacher_loads=["/tmp/fake"])
        with pytest.raises(ValueError, match="use-mopd is not enabled"):
            slime_validate_args(args)

    def test_teacher_ckpt_steps_without_use_mopd_raises(self):
        args = _make_validate_args(use_mopd=False, mopd_teacher_ckpt_steps=[1])
        with pytest.raises(ValueError, match="use-mopd is not enabled"):
            slime_validate_args(args)
