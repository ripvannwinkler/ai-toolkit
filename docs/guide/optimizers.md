# Optimizers

There is no universal best optimizer. The values below are conservative starting points for LoRA training, not guarantees. Tune against fixed validation prompts and save checkpoints frequently.

## Quick recommendations

| Optimizer | Starting `lr` | Starting `optimizer_params` | Notes |
| --- | ---: | --- | --- |
| `adamw8bit` | `1e-4` | `weight_decay: 1e-4` | Best general-purpose memory-saving baseline |
| `adamw` | `1e-4` | `weight_decay: 1e-4` | Use when memory is available or bitsandbytes is unavailable |
| `adam` / `adam8bit` | `1e-4` | `weight_decay: 0` | Stable baseline; use AdamW-style decay when regularization is wanted |
| `lion8bit` | `1e-5` | `weight_decay: 1e-4` | More sensitive to LR; start lower |
| `lion` | `1e-5` | `weight_decay: 1e-4` | Same tuning caution as Lion8bit |
| `adafactor` | `1e-4` | `relative_step: false`, `scale_parameter: false`, `warmup_init: false` | Useful when optimizer state memory matters |
| `prodigy8bit` | `1.0` | See Prodigy section | Adaptive D estimate; do not use Adam-scale LR values |
| `prodigy` | `1.0` | See Prodigy section | Full-precision Prodigy |
| `dadaptationadam` | `1.0` | Usually defaults | Adaptive optimizer; its useful LR range differs from Adam |
| `dadaptationlion` | `1.0` | Usually defaults | Adaptive Lion variant |
| `automagic` | `1e-4` | Usually defaults | Adaptive per-parameter LR; high values are clamped by the implementation |
| `automagic2` | `1e-4` | Usually defaults | Hook-based and not accumulation-friendly |
| `automagic3` | `1e-4` | `fused: false` when accumulating | Adaptive group LR; experimental |
| `automagicexperiment` | `1e-4` | `fused: false` when accumulating | Experimental; do not use as a first baseline |

The toolkit passes `optimizer_params` directly to the selected optimizer. Unsupported parameter names fail at optimizer construction. Some optimizer types also force their own `eps` value in `toolkit/optimizer.py`, so avoid overriding `eps` unless the implementation accepts it without a duplicate argument.

## Recommended baselines

### AdamW8bit

```yaml
optimizer: adamw8bit
lr: 1e-4
optimizer_params:
  weight_decay: 1e-4
```

This is the safest first comparison for most LoRA experiments. Use `cosine` if you have evidence that late-training updates are harming validation quality; otherwise the default constant scheduler is simple and predictable.

### Prodigy8bit

```yaml
optimizer: prodigy8bit
lr: 1.0
optimizer_params:
  d_coef: 1.0
  growth_rate: 1.02
  safeguard_warmup: true
  decouple: true
  weight_decay: 1e-4
  use_bias_correction: false
```

The project changes configured Prodigy LR values below `0.1` to `1.0`. `d0` starts at `1e-6` as an internal scale estimate; it is not the LR to tune. `d_coef` changes the adaptive scale and is the first parameter to try when the run is too slow or too aggressive. `growth_rate` limits upward growth but does not create decay. Use `lr_scheduler: cosine` when you explicitly want the scheduler-controlled LR to decline.

### Automagic v3

```yaml
optimizer: automagic3
lr: 1e-4
optimizer_params:
  fused: false
  clip_threshold: 1.0
  weight_decay: 1e-4
```

The starting LR is a launch point; Automagic v3 adapts it from direction consistency. Fused mode updates parameters from backward hooks and bypasses normal trainer gradient clipping and NaN-skip handling. It is not compatible with multi-backward gradient accumulation. Use `fused: false` unless you specifically need the fused memory behavior and are not accumulating.

## Scheduler interaction

`constant` is the project default. It leaves the scheduler-controlled base LR unchanged. This does not mean every optimizer has a fixed effective step size:

- Prodigy adapts its internal `d` scale.
- Automagic adapts LR values internally.
- Adam, AdamW, Lion, and Adafactor still change update magnitude through their moment normalization.

Use `cosine` or `linear` when you want explicit late-training decay. Do not assume that a scheduler can undo an optimizer's internal scale growth; validate the resulting update behavior.

## Weight decay and `decouple`

`weight_decay` regularizes the trainable weights; it is not LR decay. With `decouple: true`, supported optimizers apply AdamW-style decay separately from gradient normalization. This is generally the easier behavior to reason about for LoRA. A value of zero disables it.

## Common pitfalls

- Do not use `lr: 1e-4` for Prodigy or D-Adaptation and assume it means the same thing as Adam; this project promotes small values to `1.0`.
- Do not use fused Automagic with `gradient_accumulation > 1`.
- Do not add `betas` to an optimizer whose constructor does not accept them.
- Do not compare optimizers using only the final checkpoint; adaptive optimizers can peak earlier.
- Start with one baseline optimizer before tuning rank, scheduler, data, and optimizer simultaneously.
