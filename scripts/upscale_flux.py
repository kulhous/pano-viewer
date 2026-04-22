#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from diffusers import FluxControlNetModel, FluxControlNetPipeline


BASE_MODEL = "black-forest-labs/FLUX.1-dev"
CONTROLNET_MODEL = "jasperai/Flux.1-dev-Controlnet-Upscaler"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch x4 upscale images with Jasper's FLUX.1-dev ControlNet upscaler."
    )
    parser.add_argument("--input-dir", default="samples", help="Directory with source images.")
    parser.add_argument(
        "--output-dir",
        default="samples-flux-upscaled",
        help="Directory for x4 upscaled output images.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        help="Hugging Face token with access to black-forest-labs/FLUX.1-dev.",
    )
    parser.add_argument("--prompt", default="", help="Optional prompt bias for the upscaler.")
    parser.add_argument(
        "--negative-prompt",
        default="",
        help="Optional negative prompt.",
    )
    parser.add_argument("--steps", type=int, default=28, help="Inference steps per image.")
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=3.5,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--conditioning-scale",
        type=float,
        default=0.6,
        help="ControlNet conditioning scale.",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=256,
        help="Reduce this if memory is tight.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cpu"),
        default="auto",
        help="Execution device. Apple Silicon should use mps when available.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
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


def resolve_dtype(device):
    if device == "mps":
        return torch.float16
    return torch.float32


def load_pipeline(device, dtype, token):
    controlnet = FluxControlNetModel.from_pretrained(
        CONTROLNET_MODEL,
        torch_dtype=dtype,
        token=token,
    )
    pipe = FluxControlNetPipeline.from_pretrained(
        BASE_MODEL,
        controlnet=controlnet,
        torch_dtype=dtype,
        token=token,
    )
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe.to(device)
    return pipe


def iter_input_images(input_dir):
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def upscale_image(pipe, image_path, output_path, args):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        control_image = image.resize((width * 4, height * 4), Image.Resampling.LANCZOS)

    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        control_image=control_image,
        controlnet_conditioning_scale=args.conditioning_scale,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        max_sequence_length=args.max_sequence_length,
        width=control_image.size[0],
        height=control_image.size[1],
    ).images[0]
    result.save(output_path)


def main():
    args = parse_args()
    if not args.token:
        raise SystemExit(
            "Missing Hugging Face token. Set HF_TOKEN or pass --token after accepting access to black-forest-labs/FLUX.1-dev."
        )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    dtype = resolve_dtype(device)

    print(f"Loading pipeline on {device} with {dtype}...")
    pipe = load_pipeline(device, dtype, args.token)

    image_paths = list(iter_input_images(input_dir))
    if not image_paths:
        raise SystemExit(f"No supported images found in {input_dir}")

    for image_path in image_paths:
        output_path = output_dir / image_path.name
        if output_path.exists() and not args.overwrite:
            print(f"Skipping existing {output_path.name}")
            continue
        print(f"Upscaling {image_path.name} -> {output_path.name}")
        upscale_image(pipe, image_path, output_path, args)

    print("Done.")


if __name__ == "__main__":
    main()