# Training Process

## Lifecycle

An `sd_trainer` job follows this sequence:

1. Parse the config and expand environment variables.
2. Build typed process, model, network, dataset, save, sample, and train settings.
3. Load the base model, VAE, tokenizer, and text encoders.
4. Create the trainable network or full-model parameter groups.
5. Build the optimizer and LR scheduler.
6. Load or cache latents and text embeddings as configured.
7. Optionally produce baseline samples.
8. Iterate over batches, select noise and timesteps, run the model, and compute loss.
9. Accumulate gradients if configured, then update the optimizer.
10. Log metrics, save checkpoints, and generate samples at configured intervals.
11. Save the final network and optionally upload it.

## What one training step does

For each batch, the trainer prepares the model input and a noise/timestep target. Flow-matching models generally train toward the noise-minus-clean-latent direction. The model loss is reduced, gradients are optionally clipped, and the optimizer is stepped according to the configured accumulation mode.

The scheduler is stepped separately. This distinction matters for adaptive optimizers: the scheduler changes the optimizer parameter-group LR, while an optimizer may maintain its own per-group or per-parameter scale.

## Gradient accumulation

The defaults are effectively no accumulation:

```yaml
gradient_accumulation: 1
gradient_accumulation_steps: 1
```

Use only one accumulation setting. A value greater than 1 means multiple backward passes contribute to an optimizer update. This changes the effective batch size and usually requires a proportional reduction in the number of optimizer steps for fair comparisons.

Hook-based optimizers such as Automagic v2 and Automagic v3 fused update parameters during backward. They are not compatible with multi-backward accumulation. Set `fused: false` for Automagic v3 when accumulating.

## Checkpointing and validation

Save checkpoints before the point where you expect the model to peak. Compare fixed validation prompts across checkpoints rather than judging only the latest sample. Training loss can continue to improve after identity, composition, caption following, or edit fidelity has started to degrade.

Keep the best checkpoint by observed behavior, not necessarily the final checkpoint. Resume both model and optimizer state when continuing a run; resetting the optimizer can change adaptive LR behavior.

## Practical training loop

1. Verify one batch and one sample before committing to a long run.
2. Start with 500 to 1000 steps for a small experiment.
3. Save every 100 to 250 steps while tuning.
4. Change one variable at a time: rank, LR, captions, resolution, or optimizer.
5. Watch for NaNs, sudden loss spikes, washed-out samples, prompt leakage, and memorization.
6. Remove or reduce the cause before increasing steps.
