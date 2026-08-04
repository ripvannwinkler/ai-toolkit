# Configuration

## Top-level structure

The standard configuration contains `job`, `config.name`, and one or more `config.process` entries:

```yaml
job: extension
config:
  name: run_name
  process:
    - type: sd_trainer
      # process-specific sections
```

Environment variables can be expanded with `${NAME}`. An unset variable is an error. The config loader is permissive, so a misspelled key may not fail until a later stage; copy a nearby example and change it incrementally.

## Network

```yaml
network:
  type: lora              # lora, locon, lorm, or lokr
  linear: 16
  linear_alpha: 16
  conv: 0
  conv_alpha: 1
  dropout: 0.0
  transformer_only: true
  network_kwargs: {}
```

`rank` is accepted as a backward-compatible alias for `linear`. The code default is rank 4 and alpha 1. A rank of 16 is a practical starting point for many modern transformer LoRAs, but rank should match the complexity and amount of training data. Higher rank increases capacity and overfitting risk.

Use `network_kwargs.only_if_contains` and `ignore_if_contains` to target layers. Exclusions take priority.

## Dataset

```yaml
datasets:
  - folder_path: /path/to/images
    caption_ext: txt
    resolution: [512, 768, 1024]
    caption_dropout_rate: 0.05
    shuffle_tokens: false
    cache_latents_to_disk: true
    cache_text_embeddings: false
```

Images and captions normally share a basename, for example `image-01.png` and `image-01.txt`. The trainer buckets and downsizes images; it does not upscale them. Use JPG, JPEG, or PNG for the safest path.

Latent caching saves repeated VAE work. Disk caching is useful for large datasets. Augmentations can disable latent caching. Text-embedding caching saves memory and compute for supported models, but it also means caption changes and trigger-word substitution cannot happen dynamically.

For paired or edit workflows, use `control_path` as documented by the model example.

## Train

Common fields and their code defaults:

| Field | Default | Purpose |
| --- | --- | --- |
| `steps` | `1000` | Number of optimizer steps |
| `batch_size` | `1` | Per-device batch size |
| `gradient_accumulation` | `1` | Number of backward passes grouped by the trainer |
| `gradient_accumulation_steps` | `1` | Alternate accumulation control; do not combine with the field above |
| `lr` | `1e-6` | Base or starting LR, depending on optimizer |
| `optimizer` | `adamw` | Optimizer name |
| `optimizer_params` | `{}` | Extra constructor parameters |
| `lr_scheduler` | `constant` | LR scheduler name |
| `lr_scheduler_params` | `{}` | Scheduler constructor parameters |
| `dtype` | `fp32` | Training parameter dtype |
| `gradient_checkpointing` | `false` | Lower VRAM at the cost of compute |
| `train_unet` | `true` | Train the main denoiser/transformer |
| `train_text_encoder` | `false` | Train the text encoder when supported |

Separate rates are available for some parameter groups, including `unet_lr`, `text_encoder_lr`, `refiner_lr`, `embedding_lr`, and `adapter_lr`.

## Schedulers

The supported built-in scheduler names are `constant`, `cosine`, `cosine_with_restarts`, `step`, `linear`, and `constant_with_warmup`. The default `constant` scheduler keeps the scheduler-controlled base LR constant. Adaptive optimizers can still change their internal effective step size.

For an actual decay schedule:

```yaml
lr_scheduler: cosine
```

The trainer supplies the total step count when the scheduler parameters do not specify it. Scheduler behavior can be irrelevant or conflicting for optimizers that manage their own LR; see the optimizer page.

## Save and sample

```yaml
save:
  dtype: float16
  save_every: 250
  max_step_saves_to_keep: 4
  save_format: safetensors

sample:
  sampler: flowmatch
  sample_every: 250
  sample_start_step: 0
  width: 1024
  height: 1024
  seed: 42
  prompts:
    - "a useful validation prompt"
```

Sampling is a diagnostic, not a training target. Keep prompts fixed when comparing checkpoints.
