#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from aura_sr import AuraSR
from huggingface_hub import snapshot_download
from PIL import Image
from safetensors.torch import load_file


MODEL_ID = "fal/AuraSR-v2"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch x4 upscale images with fal/AuraSR-v2."
    )
    parser.add_argument("--input-dir", default="samples", help="Directory with source images.")
    parser.add_argument(
        "--output-dir",
        default="samples-aurasr-upscaled",
        help="Directory for AuraSR x4 output images.",
    )
    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
        help="Hugging Face model id or a local checkpoint file.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cpu"),
        default="auto",
        help="Execution device.",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=4,
        help="Tile batch size. Lower this if memory is tight.",
    )
    parser.add_argument(
        "--weight-type",
        choices=("checkboard", "constant"),
        default="constant",
        help="Overlap blending mode used by AuraSR.",
    )
    parser.add_argument(
        "--method",
        choices=("overlapped", "plain"),
        default="overlapped",
        help="Use overlapped tile blending or plain 4x inference.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Optional list of file names to process from the input directory.",
    )
    return parser.parse_args()


def resolve_device(device_name):
    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return "mps"
    if device_name == "cpu":
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def iter_input_images(input_dir):
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def filter_input_images(image_paths, file_names):
    if not file_names:
        return image_paths
    wanted = set(file_names)
    return [path for path in image_paths if path.name in wanted]


def load_model(model_id, device):
    model_path = Path(snapshot_download(model_id))
    config = json.loads((model_path / "config.json").read_text())
    model = AuraSR(config, device=device)
    checkpoint = load_file(model_path / "model.safetensors")
    model.upsampler.load_state_dict(checkpoint, strict=True)
    model.upsampler.eval()
    return model


def upscale_image(model, image_path, output_path, max_batch_size, weight_type, method):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        if method == "plain":
            result = model.upscale_4x(
                image,
                max_batch_size=max_batch_size,
            )
        else:
            result = model.upscale_4x_overlapped(
                image,
                max_batch_size=max_batch_size,
                weight_type=weight_type,
            )
    result.save(output_path)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
      raise SystemExit(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Loading AuraSR on {device}...")
    model = load_model(args.model_id, device)

    image_paths = filter_input_images(list(iter_input_images(input_dir)), args.files)
    if not image_paths:
        raise SystemExit(f"No supported images found in {input_dir}")

    for image_path in image_paths:
        output_path = output_dir / image_path.name
        if output_path.exists() and not args.overwrite:
            print(f"Skipping existing {output_path.name}")
            continue
        print(f"Upscaling {image_path.name} -> {output_path.name}")
        upscale_image(model, image_path, output_path, args.max_batch_size, args.weight_type, args.method)

    print("Done.")


if __name__ == "__main__":
    main()