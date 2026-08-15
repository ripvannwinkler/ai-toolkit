#!/usr/bin/env python3
"""Caption every image in a directory using a local vision model.

Talks to an OpenAI-compatible server (default http://127.0.0.1:8080/v1),
lets you pick a model from /v1/models, and writes each caption to a .txt
file next to the image with the same base name.

Stdlib only — no dependencies.
"""

import argparse
import base64
import json
import mimetypes
import sys
import urllib.request
from pathlib import Path

DEFAULT_API_BASE = "http://127.0.0.1:8080/v1"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_PROMPT = "Describe this image in detail. Write a single, clear caption."


def api_get(api_base: str, path: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(f"{api_base}{path}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def api_post(api_base: str, path: str, payload: dict, timeout: int = 600) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{api_base}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_models(api_base: str) -> list[tuple[str, bool]]:
    """Return (model_id, accepts_images) pairs from the /models endpoint."""
    try:
        body = api_get(api_base, "/models")
    except Exception as e:
        sys.exit(f"Could not reach {api_base}/models — is the server running? ({e})")
    models = []
    for m in body.get("data", []):
        model_id = m.get("id")
        if not model_id:
            continue
        inputs = (m.get("architecture") or {}).get("input_modalities") or []
        models.append((model_id, "image" in inputs))
    if not models:
        sys.exit("The /models endpoint returned no models.")
    return models


def choose_model(models: list[tuple[str, bool]]) -> str:
    print("Available models:")
    for i, (model_id, vision) in enumerate(models, 1):
        tag = " [vision]" if vision else ""
        print(f"  [{i}] {model_id}{tag}")
    if not any(v for _, v in models):
        print("Warning: no model advertises image input — captioning may fail.")
    ids = [m for m, _ in models]
    while True:
        choice = input(f"Choose a model (1-{len(models)} or type the exact id): ").strip()
        if choice in ids:
            return choice
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            model_id, vision = models[int(choice) - 1]
            if not vision and input(f"{model_id} does not advertise image input. Use anyway? [y/N] ").strip().lower() != "y":
                continue
            return model_id
        print("Invalid choice, try again.")


def get_prompt(cli_prompt: str | None) -> str:
    """Use the --prompt value if given, else let the user type one interactively."""
    if cli_prompt:
        return cli_prompt
    print(f"Default prompt: {DEFAULT_PROMPT}")
    custom = input("Type a custom caption prompt (or press Enter for default): ").strip()
    return custom or DEFAULT_PROMPT


def find_images(directory: Path) -> list[Path]:
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def caption_image(api_base: str, path: Path, model: str, prompt: str) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                ],
            }
        ],
    }
    body = api_post(api_base, "/chat/completions", payload)
    return body["choices"][0]["message"]["content"].strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Caption every image in a directory with a local vision model."
    )
    parser.add_argument("directory", type=Path, help="Directory containing images")
    parser.add_argument(
        "--api-base", default=DEFAULT_API_BASE, help=f"OpenAI-compatible API base (default: {DEFAULT_API_BASE})"
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Caption prompt (if omitted, you will be asked to type one)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-caption images that already have a .txt file",
    )
    args = parser.parse_args()

    directory = args.directory.expanduser().resolve()
    if not directory.is_dir():
        sys.exit(f"Not a directory: {directory}")

    images = find_images(directory)
    if not images:
        sys.exit(f"No images found in {directory}")
    if not args.overwrite:
        images = [p for p in images if not p.with_suffix(".txt").exists()]
        if not images:
            sys.exit("All images already have captions (use --overwrite to redo).")

    model = choose_model(fetch_models(args.api_base))
    prompt = get_prompt(args.prompt)

    print(f"\nCaptioning {len(images)} image(s) with {model}...")
    print(f"Prompt: {prompt}\n")
    failures = 0
    for i, path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {path.name} ... ", end="", flush=True)
        try:
            caption = caption_image(args.api_base, path, model, prompt)
        except Exception as e:
            failures += 1
            print(f"FAILED ({e})")
            continue
        path.with_suffix(".txt").write_text(caption, encoding="utf-8")
        print("done")

    print(f"\nFinished: {len(images) - failures} captioned, {failures} failed.")


if __name__ == "__main__":
    main()
