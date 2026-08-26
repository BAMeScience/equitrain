# API Reference

This reference is curated around the stable public objects that are useful when
embedding Equitrain in Python code. The narrative pages remain the primary
documentation for end-to-end workflows.

## Torch Wrapper Interface

::: equitrain.backends.torch_wrappers.base.AbstractWrapper

## Torch Model Wrappers

::: equitrain.backends.torch_wrappers.mace.MaceWrapper

::: equitrain.backends.torch_wrappers.ani.AniWrapper

::: equitrain.backends.torch_wrappers.orb.OrbWrapper

::: equitrain.backends.torch_wrappers.sevennet.SevennetWrapper

M3GNet is covered in [Model Wrappers](model-wrappers.md#m3gnet). Its generated
API reference is omitted because the MatGL wrapper requires the optional MatGL
runtime package at import time.

## Fine-Tuning Wrappers

::: equitrain.finetune.delta_torch.DeltaFineTuneWrapper

::: equitrain.finetune.freeze_torch.FreezeFineTuneWrapper

::: equitrain.finetune.lora_torch.LoRAFineTuneWrapper

::: equitrain.finetune.lora_torch.LoRASpec

JAX fine-tuning wrappers are described in [Fine-Tuning](fine-tuning.md). They
are kept out of this generated page so the documentation build does not require
the optional JAX runtime.
