---
layout: home
hero:
  name: AI Toolkit
  text: Training guide
  tagline: Configure, run, and troubleshoot diffusion-model training jobs.
  actions:
    - theme: brand
      text: Get started
      link: /guide/getting-started
    - theme: alt
      text: Optimizer guide
      link: /guide/optimizers
features:
  - title: Configuration first
    details: Understand the YAML shape, defaults, datasets, networks, saves, and samples.
    link: /guide/configuration
  - title: Optimizer guidance
    details: Compare fixed-rate, adaptive, memory-efficient, and experimental optimizers.
    link: /guide/optimizers
  - title: Model-aware training
    details: Read the constraints and practical notes for Qwen Image, Krea 2, Flux, Wan, and more.
    link: /guide/models
---

## What this guide covers

These pages document the behavior implemented in this repository. Settings copied from example YAML files are labeled as starting points, not universal requirements. When a code default and an example disagree, the code default wins unless the example explicitly sets the value.

## Quick links

- [Install and run a first job](/guide/getting-started)
- [Learn the complete config shape](/guide/configuration)
- [Understand what happens during training](/guide/training-process)
- [Choose an optimizer and starting parameters](/guide/optimizers)
- [Check model-specific constraints](/guide/models)
- [Diagnose common failures](/guide/troubleshooting)
