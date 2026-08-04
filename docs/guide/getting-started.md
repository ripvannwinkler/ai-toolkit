# Getting Started

## Install the environment

The repository manager is the preferred setup path. On Windows, run `run_windows.bat`, or invoke it directly:

```powershell
python -m manager install
```

The manager creates `venv/` or uses an existing `.venv/`, installs the matching PyTorch build and requirements, and adds platform extras where wheels exist. Start the UI with:

```powershell
python -m manager launch
```

For a manual setup, create and activate a venv before installing PyTorch and `requirements.txt`:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install torch torchvision torchaudio
python -m pip install -r requirements.txt
```

Use the repository manager when possible because it pins the Torch trio and platform-specific accelerator wheels together.

## Run a config

Training configs are YAML, JSON, or JSONC. A normal extension job has this shape:

```yaml
job: extension
config:
  name: my_lora
  process:
    - type: sd_trainer
      training_folder: output
      device: cuda:0
      network:
        type: lora
        linear: 16
        linear_alpha: 16
      datasets:
        - folder_path: /path/to/images
          caption_ext: txt
          resolution: [512, 768, 1024]
      train:
        steps: 2000
        batch_size: 1
        optimizer: adamw8bit
        lr: 1e-4
```

Run it with the venv Python:

```powershell
venv\Scripts\python.exe flux_train_ui.py --config config.yaml
```

The exact launch command can vary with the job entry point. The UI can generate and launch the same config.

## Before the first run

- Use captions with the same base filename as each image.
- Start with a small step count and inspect samples at regular intervals.
- Save checkpoints frequently enough to compare training stages.
- Keep a validation prompt set separate from training captions.
- Confirm that the model architecture, dataset format, and sample sampler match.
- If VRAM is tight, enable gradient checkpointing, latent caching, text-embedding caching where supported, or model quantization from the relevant example.
