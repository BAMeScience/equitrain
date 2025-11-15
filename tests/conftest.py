"""
Pytest configuration shared across the suite.

We mask CUDA devices early so that deterministic CPU-only reference tests
(`test_finetune_mace_jax.py`, `test_jax_model_equivalence.py`,
`test_train_mace_jax.py`) execute even when the host has GPUs. Those tests make
bitwise comparisons between Torch and JAX outputs, which would otherwise be
skipped when ``torch.cuda.is_available()`` is ``True``. Setting the environment
variable here guarantees it is applied before any other module imports Torch or
JAX, so the skip markers (which evaluate at import time) always see a CPU-only
environment.
"""

import os

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
