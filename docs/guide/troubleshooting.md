# Troubleshooting

## Out of memory

Reduce memory in this order:

1. Lower resolution or number of resolution buckets.
2. Enable gradient checkpointing.
3. Enable latent or text-embedding caching when supported.
4. Use the model example's quantization and `low_vram` settings.
5. Keep batch size at 1 and use accumulation only when the optimizer supports it.
6. Reduce LoRA rank or target fewer layers.

Do not assume a smaller batch automatically makes every model fit. Text encoders, VAE caching, control images, and video frames can dominate memory independently.

## Loss spikes or NaNs

- Confirm the model dtype and PyTorch build are supported.
- Reduce LR or adaptive optimizer aggressiveness.
- For Automagic fused mode, try `fused: false` so the normal trainer safety path applies.
- Check captions, image files, and control images for corrupt or unexpected inputs.
- Disable one optional optimization at a time to isolate the cause.

## Samples get worse while loss improves

This is usually a checkpoint-selection or overfitting problem, not proof that the run needs more steps. Compare fixed validation prompts, reduce steps, lower rank, add mild weight decay, improve caption diversity, or use an explicit decaying scheduler.

## Trigger words do not work

Trigger-word replacement does not work dynamically when text embeddings are cached. Put the trigger in captions before caching, or disable text-embedding caching for that workflow.

## Control-image failures

Check all three pieces together:

- the model architecture supports the control/edit mode;
- each dataset item has the expected `control_path`;
- sample prompts include the expected control-image argument.

Base Qwen Image and Qwen Image Edit are different paths. Flux Kontext, Wan I2V, and other edit workflows also have model-specific control handling.

## Environment checks

Run the manager doctor command:

```powershell
python -m manager doctor
```

It checks the venv, Torch trio, GPU visibility, local runtime tools, and dependency state. Run training with the venv interpreter, not a different global Python:

```powershell
venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
