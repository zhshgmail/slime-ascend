# Original Slime TP4 cold-restore preparation

These tools add observers/readers around the selected original Slime train.py. They do not implement a substitute model, optimizer, RNG restore, generator, or training loop. Native hook invocation and full cold restore are NOT_RUN until the real target consumer runs. Keep the accepted source/runtime/TP4/16 samples/4096/graphs/offload/backup settings. Current full-project acceptance remains open.

## A valid paired execution

1. Use `prepare_entries.py --original-argv ORIGINAL_TARGET_ARGV.json --checkpoint-root /restore/split-checkpoint-output --output UNIQUE_OUTPUT --contract CONTRACT.json --contract-sha256 SHA256` to prepare the explicit `control_argv.json` and `resume_argv.json`. Each JSON is the complete `sys.argv` for the normal original `train.py`; keep the existing original runpy entry and bind the selected JSON in place of its argument file. Add only that profile's recorded environment. Stage these tools at the Slime source root so the original hook's normal module import resolves.
2. Control runs original rollouts 0 and 1 from the same canonical HF model and dataset. Both calls preserve actual rollout Samples and trainer inputs through original debug-save switches; they do not replay or fabricate responses. The hook returns without touching state outside rollout1/step0, avoiding original lazy optimizer state before the first update. Before original training step 1, the observer hashes each TP rank's full local model, original optimizer DP-zero master/moment buffers, scheduler, and original RNG getter. Optimizer `state_dict()` alone omits parameter and moment buffers and is insufficient. The reader synchronizes and copies state to CPU; this run cannot serve as an uninstrumented performance measurement.
3. At that same point, rank0 preserves the actual tracker byte while it still selects iteration0. Resume needs a shallow read-only checkpoint view: the SAME control's canonical `iter_0000000` directory, its canonical `rollout` directory, and this captured `split_latest_checkpointed_iteration.txt` bound as `latest_checkpointed_iteration.txt`. Do not copy 44GB. Do not pass the later control root unchanged: after control finishes its tracker points at1, and selected MCore uses a truthiness test so `--ckpt-step 0` does not override it. Do not substitute a different fresh trajectory's checkpoint0.
4. Retire the control unit completely, then start a fresh process/container with the resume argv (`start_rollout_id=1`, `num_rollout=2`, real split checkpoint root). All four ranks must pass the original model/optimizer/scheduler/RNG loader. The existing original data-source load is also required. Resource admission, target paths and bounded physical lifetime remain Root's separate authority; this preparation grants none.
5. Run `compare_step_digests.py --control CONTROL_STATE_DIR --resume RESUME_STATE_DIR --output RESULT.json --contract CONTRACT.json --contract-sha256 SHA256`. It requires all four actual rank files for rollout1/step0 and compares every recorded section. Existing author and independent fresh-run checkpoints are not a paired control.
6. Compare the ACTUAL original rollout/training-input dumps before attributing a next-step parameter, gradient, loss or metric difference to restoration. Original deterministic-inference is false and trainer checkpoints do not restore SGLang's server sampling trajectory. Restarting its initial random seed is not the post-rollout0 generator state. Do not silently enable deterministic inference, forge replay, skip generation, or invent tolerances. The independently observed followup token-count discrepancy is a separate unresolved input-interface issue. Exact restored state is useful evidence even if subsequent generated inputs differ; that mismatch must remain explicit and is not a numerical-continuity pass.

## Persisted-state reader

`compare_states.py --left ROOT/iter_0000000 --right OTHER/iter_0000000 --output RESULT.json` uses actual Torch DCP readers/planners, reconstructs one logical tensor/object at a time on CPU, and compares full logical state plus `common.pt`. It includes all four RNG objects and every master/moment key. CPU tensor cap defaults to4GiB for one tensor, object cap512MiB; loading one side and hashing before the other bounds residency. Explicit exact-value common startup differences can be declared with `--allow-common-differences`; no wildcard, blanket args exclusion or automatic normalization occurs. `--metadata-only` compares key/type/dtype/shape/chunks ONLY, never tensor values. Package SHA, metadata equality, CPU object decode and these utilities' tests are not device RNG restore, model construction or full native cold-restore proof.

The selected CUDA-named RNG API is redirected by the actual target Adaptor; native validation must bind that loaded provider and real NPU generator/tracker behavior. CPU tests do not execute the observer or its native optimizer getter. Source/hash fixtures and test failures remain in the evidence packet.


## Complete-original contract and admission

Generic persisted comparison remains useful for q_norm-only or other small operands and never reports complete original state. Add `--complete-original --contract CONTRACT.json --contract-sha256 SHA256` for a complete persisted-state verdict. Metadata-only mode can never yield that verdict. The externally retained SHA must authenticate the contract before comparing both sides; matching incomplete files are insufficient.

Build the contract inventory using `geometry(FileSystemReader(CANONICAL_ITER0).read_metadata())` from this reader and the actual canonical metadata bytes bound by the original input receipt. It must have the original522 entries, four RNG objects and all four ranks' actual param/exp_avg/exp_avg_sq keys. Do not invent model architecture keys. Contract shape: `{"schema":"slime-original-tp4-v1","inventory": GEOMETRY,"metadata":{"path":"CANONICAL_ITER0/.metadata","sha256":"CANONICAL_SHA256"},"identity":{"source":{"files":{"ACTUAL_SOURCE_PATH":"SHA256"}},"input":{"files":{"ACTUAL_INPUT_PATH":"SHA256"}},"runtime":{"files":{"ACTUAL_RUNTIME_PATH":"SHA256"},"python":"MAJOR.MINOR","torch":"EXACT_VERSION"}}}`. Source, input and runtime files must exist and match in the actual consumer namespace. Root must bind accepted whole source/runtime/input receipts and actual assets; unknown identity prevents qualification. The contract SHA authenticates these selected expectations, not the truth of an unreviewed identity declaration.

The observer requires `SLIME_COLD_CONTRACT` and `SLIME_COLD_CONTRACT_SHA256`, checks actual identity files/runtime, and reads the original sharded_state_dict model inventory, requires exact canonical model key/global geometry and four-rank shard coverage, and validates nonempty model, scheduler, per-rank original master/moment geometry and Python/NumPy/CPU/device/tracker RNG before writing. Set ALL profile env_add values BEFORE starting the original Ray head or explicitly propagate them to train_env_vars; a late driver-only export does not deliver new keys to original actors. Do not modify the original actor implementation to hide this condition. Only original DP1TP4 rollout1/step0 is observed; CPU malformed-JSON diagnostics are utility checks and never native invocation evidence.

Selected original distrib_optimizer.py line1121 allocates torch.cat while world3 + local3 + receive1 coexist: potential8P, and a previous concat may survive RHS assignment, potential9P. Actual canonical P=13,395,099,648B gives107,160,797,184B and120,555,896,832B respectively across ranks. These are potential source-derived buffer occupancy, not measured RSS or a complete host bound. Admission must additionally cover original training/offload/backups, bucket padding, tensor readback, finite-mask allocations and allocator/runtime headroom; measure target peak. The old7P estimate is incomplete and old64GiB floor cannot qualify this observer.

## Withdrawn complete-state verdict
Fresh independent normal-CLI counterexamples still make64f accept incomplete model values despite full separate inventory, unrelated optimizer metadata and tiny RNG state. Complete-state acceptance is therefore withdrawn. The step reader now reports only `reported_sections_equal`, keeps `all_complete_state_equal=false`, and returns2 for equal-but-unqualified input. The DCP reader preserves generic persisted-state equality; requested complete-original value mode returns2 and never complete=true. Errors remain failures. This is a truthful scope restriction, NOT full completeness repair, native qualification or weakened original acceptance. Exact binding of actual model value names/dtypes/shards to inventory and full original optimizer/RNG content remains an open repair.

## Value-bound original save adapter (unqualified native integration)

`sharded_values.collect_original` calls selected original MCore
`generate_state_dict` using the original save metadata, actual model,
optimizer, scheduler and original `torch_dist` RNG object. It then calls original
`save_preprocess` to expand the actual factories. Every tensor record owns both
its actual value hash/dtype/local shape and the SAME original ShardedTensor's
key/global shape/offset. There is no separate model inventory used to validate an
unrelated model digest. Optimizer parameter/master/moment records come from the
original optimizer's own save-state interface, including its actual semantic
keys. Original checkpointing/mapping/preprocess source files must be present in
an externally authenticated source contract and match their actual file hashes.

The new hook path is
`tools.cold_restore.sharded_values.before_train_step`. It preserves the original
iteration-zero split tracker and original RNG object-list serialization as a
small independently decodable file. It never manufactures device RNG bytes.
`verify_tensor_coverage` requires actual value-bearing records for every requested
key/chunk in authenticated DCP metadata, with exact dtype/shape/chunk coordinates
and matching replicas. Its success concerns ONLY the explicitly requested tensor
set, and ALWAYS leaves `all_complete_state_equal=false`.

Native invocation, full original model/optimizer capture, device RNG API/extent
validation and paired cold restore versus uninterrupted control are NOT_RUN.
No complete-state verdict is re-enabled by this adapter. Existing e0e2915
withdrawal remains. Full verification must independently consume actual saved
and restored values, all optimizer semantic state and raw RNG content; matching
capture JSON is insufficient. Existing original-getter peak memory estimates
(including temporary concatenations) remain; this path is not memory-qualified
by its small CPU tests. This is a readback observer and cannot supply performance
results. No training math, model implementation or original argv is changed.
