# Algeon cuDF fork

This repository is forked from NVIDIA [cuDF](https://github.com/rapidsai/cudf) at `v26.08.00`.

This fork keeps the upstream cuDF codebase as its base and carries local changes needed by
`algeon-datafusion` (check [diff](https://github.com/algeon-dev/cudf/compare/v26.08.00...algeon-dev:cudf:algeon-26.08)).

## Fork Changes

The fork is intentionally limited to changes required by `algeon-datafusion`:

- GPU memory planning through retained join and regex state, plus preflight sizing for radix sort
  and fixed-width gather. Algeon-added execution APIs use explicit CUDA streams and RMM memory
  resources.
- Parquet reader fixes for predicate pruning and decimal values, plus ordered and coalesced range
  reads for remote sources.
- Shared RMM runtime identity when libcudf is built with static third-party dependencies.
