# SECOND HW-QAT Quant Scheme Report

- Scope: VoxelBackBone8x only
- Activation: signed symmetric INT8, zero_point=0
- Weight: signed symmetric INT8, zero_point=0, per_channel
- BN: folded offline into sparse conv weight/bias for export
- Accumulator: INT32
- Requant: shift-only nearest power-of-two
- Bias bits checked: 32
- Shift bits checked: 5
- Hard failures: 0
- Warnings: 0

## Hard Failures
- None

## Warnings
- None
