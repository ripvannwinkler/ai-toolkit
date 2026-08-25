#!/usr/bin/env python
"""Generate paired pole images for the Krea 2 areola/nipple DETAIL slider.

For each scene variation it renders TWO images at the SAME seed:
    pos/<i>.png  =  BASE + VARY[i] + POS_SUFFIX   (textured / natural bumps)
    neg/<i>.png  =  BASE + VARY[i] + NEG_SUFFIX   (smooth / ghost-like)
plus a NEUTRAL caption .txt next to each (BASE + VARY[i], no texture words).

Why this shape:
  * PairedImageDataset pairs files by MATCHING BASENAME, so pos/01 <-> neg/01.
  * Captions fall back to the uniform target_class unless a sibling .txt exists;
    we write a NEUTRAL .txt so both poles share an identical text condition and
    the LoRA learns the pure visual delta (that's what makes it a slider).

Usage:
    uv run python scripts/gen_nipple_poles.py --dry-run      # preview prompts (no GPU)
    uv run python scripts/gen_nipple_poles.py                # generate (needs GPU)
    uv run python scripts/gen_nipple_poles.py --count 8 --base-seed 7
"""
import argparse
import os
import sys
import gc

# Ensure the repo root (home of the `toolkit` package) is importable no matter
# how/where this script is launched.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---- prompt building blocks (edit here) ------------------------------------
BASE = ("Photograph tightly cropped to a bare female chest and cleavage, "
        "both areolas and nipples centered in frame, collarbones along the top edge, "
        "chin and neck cut off, no face, torso-only crop, 85mm lens, "
        "soft natural light, photorealistic, high detail")

# Vary only skin tone / lighting / angle. Keep the framing constant.
VARY = [
    "fair skin, soft diffused studio light, frontal",
    "fair-to-medium skin, warm window light, three-quarter angle",
    "medium skin tone, golden-hour backlight, rim light",
    "medium-deep skin tone, soft overcast daylight, frontal",
    "deep skin tone, dramatic low-key lighting",
    "deep skin tone, bright even studio light, slightly elevated angle",
]

POS_SUFFIX = (", highly detailed realistic skin, clearly visible Montgomery glands, "
              "naturally textured areola and nipple, fine pores and surface micro-detail, "
              "natural pigmentation, crisp sharp focus, macro detail")

NEG_SUFFIX = (", smooth soft airbrushed skin, ghost-like translucent areola and nipple, "
              "faint softly blurred edges, low contrast, matte porcelain finish, "
              "dreamy ethereal, minimal surface detail")

# Face/head exclusions. Only bite once guidance > 1 (Krea 2 applies CFG as
# guidance-1), so pair with --guidance ~5.5.
DEFAULT_NEGATIVE = ("face, head, eyes, nose, mouth, lips, teeth, hair, forehead, eyebrows, "
                    "portrait, full body, waist down, text, watermark, logo")
# ----------------------------------------------------------------------------


def build_items(count: int):
    """List of {idx, neutral, pos, neg}. Cycles VARY if count > len(VARY)."""
    items = []
    for i in range(count):
        vary = VARY[i % len(VARY)]
        neutral = f"{BASE}, {vary}"
        items.append({"idx": i, "neutral": neutral,
                      "pos": neutral + POS_SUFFIX,
                      "neg": neutral + NEG_SUFFIX})
    return items


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="krea/Krea-2-Turbo",
                   help="HF id or local .safetensors path (default: Turbo)")
    p.add_argument("--arch", default="krea2")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--qtype", default="convrot8",
                   help="DiT quant: convrot8 / qfloat8 / uint3 / none(bf16)")
    p.add_argument("--qtype-te", dest="qtype_te", default="qfloat8")
    p.add_argument("--no-low-vram", dest="low_vram", action="store_false",
                   default=True)
    p.add_argument("--out-dir", default="datasets/krea2_nipple")
    p.add_argument("--count", type=int, default=6,
                   help="number of scene variations (= pairs)")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--steps", type=int, default=9,
                   help="Turbo is distilled/few-step (proven configs use ~9); raise for more detail")
    p.add_argument("--guidance", type=float, default=5.5,
                   help="Krea2 applies CFG as guidance-1; 5.5 -> 4.5 effective (needed for steering + negative prompt)")
    p.add_argument("--sampler", default="flowmatch")
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--neg-prompt", default=DEFAULT_NEGATIVE, help="global negative prompt (face exclusions)")
    p.add_argument("--lora",
                   default="D:/Models/loras/krea2/krea2_snofs_v1_3D.safetensors",
                   help="LoRA/LoKr .safetensors to merge in (pass '' to disable)")
    p.add_argument("--lora-strength", dest="lora_strength", type=float, default=1.0,
                   help="merge-in strength for --lora (1.0 = full strength)")
    p.add_argument("--dry-run", action="store_true",
                   help="print prompts without loading the model")
    return p.parse_args()


def main():
    args = parse_args()
    items = build_items(args.count)
    if args.count > len(VARY):
        print(f"note: count={args.count} > {len(VARY)} variations; cycling VARY.")

    if args.dry_run:
        print(f"[dry-run] {len(items)} pairs -> {args.out_dir}/pos , {args.out_dir}/neg\n")
        for it in items:
            print(f"--- scene {it['idx']:02d}  (seed={args.base_seed + it['idx']}) ---")
            print("NEUTRAL:", it["neutral"])
            print("POS     :", it["pos"])
            print("NEG     :", it["neg"])
            print()
        return

    import torch
    from toolkit.config_modules import ModelConfig, GenerateImageConfig
    from toolkit.util.get_model import get_model_class
    from toolkit.train_tools import get_torch_dtype

    pos_dir = os.path.join(args.out_dir, "pos")
    neg_dir = os.path.join(args.out_dir, "neg")
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    model_config = ModelConfig(
        name_or_path=args.model,
        arch=args.arch,
        quantize=args.qtype != "none",
        qtype=args.qtype,
        quantize_te=True,
        qtype_te=args.qtype_te,
        low_vram=args.low_vram,
        dtype=args.dtype,
    )

    # Krea 2 is a registered extension model -> resolve the correct class (NOT the
    # legacy StableDiffusion) and use its flow-matching scheduler.
    ModelClass = get_model_class(model_config)
    noise_scheduler = (
        ModelClass.get_train_scheduler()
        if hasattr(ModelClass, "get_train_scheduler")
        else None
    )

    print("Loading model...")
    sd = ModelClass(
        device=args.device,
        model_config=model_config,
        dtype=model_config.dtype,
        noise_scheduler=noise_scheduler,
    )
    torch_dtype = get_torch_dtype(args.dtype)
    with torch.no_grad():
        sd.load_model()
        sd.pipeline.to(torch.device(args.device), torch_dtype)

    # ---- optional LoRA / LoKr merge-in (bakes the network into the DiT) ----
    if args.lora:
        from safetensors.torch import load_file
        from toolkit.config_modules import NetworkConfig
        from toolkit.lora_special import LoRASpecialNetwork
        from toolkit.accelerator import unwrap_model

        print(f"Loading LoRA/LoKr: {args.lora}")
        lora_sd = load_file(args.lora)
        # diffusers-style prefix -> the transformer prefix krea2's loader expects
        lora_sd = {k.replace("diffusion_model.", "transformer."): v
                   for k, v in lora_sd.items()}
        dit = unwrap_model(sd.get_model_to_train())

        is_lokr = any("lokr_w1" in k for k in lora_sd)
        if is_lokr:
            # full-rank LoKr: NetworkConfig defaults auto-detect rank/factor
            network_config = NetworkConfig(type="lokr", transformer_only=True)
            ntype = "lokr"
        else:
            dim_key = next((k for k in lora_sd if k.endswith("lora_A.weight")), None)
            dim = int(lora_sd[dim_key].shape[0]) if dim_key else 4
            network_config = NetworkConfig(
                type="lora", linear=dim, linear_alpha=dim, transformer_only=True)
            ntype = "lora"

        LoRASpecialNetwork.LORA_PREFIX_UNET = "lora_transformer"
        network = LoRASpecialNetwork(
            text_encoder=None,
            unet=dit,
            lora_dim=network_config.linear,
            multiplier=1.0,
            alpha=network_config.linear_alpha,
            train_unet=True,
            train_text_encoder=False,
            network_config=network_config,
            network_type=ntype,
            transformer_only=True,
            is_transformer=True,
            target_lin_modules=getattr(sd, "target_lora_modules", ["SingleStreamDiT"]),
        )
        network.apply_to(None, dit, apply_text_encoder=False, apply_unet=True)
        network.force_to(torch.device(args.device), torch_dtype)
        network._update_torch_multiplier()
        network.load_weights(lora_sd)
        network.merge_in(merge_weight=args.lora_strength)
        del network
        gc.collect()
        print(f"Merged {ntype.upper()} into DiT at strength {args.lora_strength}")

    gen_configs = []
    for it in items:
        seed = args.base_seed + it["idx"]
        stem_pos = os.path.join(pos_dir, f"{it['idx']:02d}")
        stem_neg = os.path.join(neg_dir, f"{it['idx']:02d}")
        gen_configs.append(GenerateImageConfig(
            prompt=it["pos"], width=args.width, height=args.height,
            negative_prompt=args.neg_prompt, seed=seed,
            guidance_scale=args.guidance, num_inference_steps=args.steps,
            output_path=stem_pos + ".png", output_ext="png"))
        gen_configs.append(GenerateImageConfig(
            prompt=it["neg"], width=args.width, height=args.height,
            negative_prompt=args.neg_prompt, seed=seed,
            guidance_scale=args.guidance, num_inference_steps=args.steps,
            output_path=stem_neg + ".png", output_ext="png"))
        # Neutral caption next to BOTH (pairing is by basename; keep text identical).
        with open(stem_pos + ".txt", "w", encoding="utf-8") as f:
            f.write(it["neutral"])
        with open(stem_neg + ".txt", "w", encoding="utf-8") as f:
            f.write(it["neutral"])

    print(f"Generating {len(gen_configs)} images ({len(items)} pairs)...")
    sd.generate_images(gen_configs, sampler=args.sampler)

    del sd
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\nDone. Pairs written to:\n  {pos_dir}\n  {neg_dir}")
    print("Review pos vs neg side-by-side; if too similar, raise --guidance or push the suffixes.")
    print("Then train: uv run run.py --config config/examples/train_slider_krea2.yaml")


if __name__ == "__main__":
    main()
