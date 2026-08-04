# Model Notes

Model support is not interchangeable. Read the matching example config and model implementation before copying a training setup.

## Qwen Image

- Architecture: `qwen_image`.
- Uses flow matching and the Qwen Image VAE.
- Resolution bucket dimensions must be divisible by 32.
- The 24 GB example uses BF16, text-embedding caching, 3-bit transformer quantization with an accuracy-recovery adapter, float8 text-encoder quantization, and `low_vram`.
- The example uses `adamw8bit`, `lr: 1e-4`, rank 16, and multiple resolution buckets.
- Training the text encoder is not recommended in the example.
- Base Qwen Image generation does not support control images. Use the Qwen Image Edit architecture for control-image workflows.

## Qwen Image Edit

- Architecture: `qwen_image_edit` or the corresponding Plus variant.
- Training data uses `control_path` for the reference/control image.
- Samples use `ctrl_img`.
- Keep the control-image distribution and edit prompts representative of inference.
- Do not apply the base Qwen Image control-image limitation to the Edit architecture.

## Krea 2

- Architecture: `krea2`.
- Uses a single-stream MMDiT with Qwen3-VL-4B-Instruct conditioning and the Qwen Image VAE.
- Default maximum text length is 512; adjust `model.model_kwargs.max_text_length` only when captions require it.
- `model.model_kwargs.edit: true` enables reference-image/edit conditioning.
- `model.model_kwargs.kv_cache: true` enables asymmetric reference-token attention for KV-cached inference. A LoRA trained without the intended KV-cache mode should not be assumed compatible with KV-cached inference.
- Reference and VLM image sizes are controlled by `control_image_max_pixels` and `vlm_max_pixels`.
- There is no dedicated Krea 2 training example in this repository, so treat values as implementation guidance and validate carefully.

## Flux 1 Dev

- Use `arch` and model flags matching the Flux example, including `is_flux: true` where required by the process.
- Flow matching, BF16, gradient checkpointing, and `adamw8bit` at `1e-4` are the example baseline.
- Multiple resolution buckets are generally useful.
- Flux-only options include attention masking and splitting the model across GPUs.
- Use `network_kwargs.only_if_contains` and `ignore_if_contains` to reduce the target layer set when memory or overfitting is a concern.

## Flux Schnell

- Use the Schnell training adapter specified by the example.
- Schnell does not use classifier-free guidance; sample guidance is `1`.
- Sampling normally uses 1 to 4 steps.
- Validate with the same short-step sampler you intend to use in production.

## Flux Kontext

- Training is paired: use `control_path` for the control image.
- Samples reference control images with `--ctrl_img`.
- Kontext uses substantially more latent memory; 1024 resolution may OOM on 24 GB cards.
- Test the paired data path before increasing resolution or batch size.

## Wan 2.1

- The example uses `arch: wan21`, flow matching, BF16, checkpointing, `adamw8bit`, and `lr: 1e-4`.
- Text-encoder training is disabled in the example.
- Image-oriented training settings may not teach temporal motion well. Use appropriate frame counts, captions, and validation clips for video work.
- `wan21_i2v` requires first-frame/control-image conditioning and a matching `control_path`/sample setup.

## Wan 2.2 14B

- The model has high-noise and low-noise transformer stages.
- At least one of `train_high_noise` and `train_low_noise` must be enabled.
- The example trains both stages and uses `switch_boundary_every: 10`.
- 4-bit quantization, accuracy-recovery adapters, text-embedding caching, and `low_vram` are used in the 24 GB example.
- Keep high- and low-noise checkpoints distinct when training only one stage. Do not assume a single-stage LoRA works identically across both regimes.

## General model advice

- Use the model's example as the first working configuration.
- Treat resolution, frame count, text length, and quantization as coupled VRAM decisions.
- Match `train.noise_scheduler` and `sample.sampler` where the example says they must match.
- Preserve the model's expected caption and control-image format.
- Save a baseline sample before training so changes are attributable.
