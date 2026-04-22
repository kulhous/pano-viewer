#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image


MODEL_REPO = "Kim2091/UltraSharpV2"
MODEL_FILES = {
    "full": "4x-UltraSharpV2_fp32_op17.onnx",
    "lite": "4x-UltraSharpV2_Lite_fp32_op17.onnx",
}
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch x4 upscale images with Kim2091/UltraSharpV2 via ONNX Runtime."
    )
    parser.add_argument("--input-dir", default="samples", help="Directory with source images.")
    parser.add_argument(
        "--output-dir",
        default="samples-ultrasharpv2-upscaled",
        help="Directory for UltraSharpV2 x4 output images.",
    )
    parser.add_argument(
        "--variant",
        choices=("full", "lite"),
        default="full",
        help="Model variant to use. 'full' is higher quality, 'lite' is faster.",
    )
    parser.add_argument(
        "--providers",
        nargs="*",
        default=None,
        help="Optional ONNX Runtime providers to use in priority order.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=256,
        help="Tile size for large-image inference. Use 0 to disable tiling.",
    )
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=16,
        help="Overlap in source pixels between neighboring tiles.",
    )
    return parser.parse_args()


def iter_input_images(input_dir):
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def resolve_providers(requested_providers):
    available = ort.get_available_providers()
    if requested_providers:
        selected = [provider for provider in requested_providers if provider in available]
        if not selected:
            raise SystemExit(
                "None of the requested providers are available. Available providers: " + ", ".join(available)
            )
        return selected

    preferred = []
    if "CoreMLExecutionProvider" in available:
        preferred.append("CoreMLExecutionProvider")
    if "CPUExecutionProvider" in available:
        preferred.append("CPUExecutionProvider")
    return preferred or available


def load_session(variant, providers):
    model_file = MODEL_FILES[variant]
    model_path = hf_hub_download(MODEL_REPO, model_file)
    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session, input_name, output_name, model_path


def image_to_tensor(image):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))[None, ...]


def tensor_to_image(tensor):
    tensor = np.clip(tensor[0], 0.0, 1.0)
    array = np.transpose(tensor, (1, 2, 0))
    array = (array * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def run_model(session, input_name, output_name, image):
    return session.run([output_name], {input_name: image_to_tensor(image)})[0]


def upscale_tiled(session, input_name, output_name, image, tile_size, tile_overlap):
    width, height = image.size
    scale = 4
    result = np.zeros((height * scale, width * scale, 3), dtype=np.float32)
    weight = np.zeros((height * scale, width * scale, 1), dtype=np.float32)

    stride = max(tile_size - tile_overlap * 2, 1)

    for top in range(0, height, stride):
        for left in range(0, width, stride):
            bottom = min(top + tile_size, height)
            right = min(left + tile_size, width)
            tile = image.crop((left, top, right, bottom))
            tile_output = run_model(session, input_name, output_name, tile)[0]
            tile_output = np.clip(np.transpose(tile_output, (1, 2, 0)), 0.0, 1.0)

            out_top = top * scale
            out_left = left * scale
            out_bottom = bottom * scale
            out_right = right * scale

            output_height = out_bottom - out_top
            output_width = out_right - out_left

            mask = np.ones((output_height, output_width, 1), dtype=np.float32)
            overlap_scaled = tile_overlap * scale

            if out_top > 0:
                ramp = np.linspace(0.0, 1.0, min(overlap_scaled, output_height), dtype=np.float32)[:, None, None]
                mask[: ramp.shape[0], :, :] *= ramp
            if out_bottom < height * scale:
                ramp = np.linspace(1.0, 0.0, min(overlap_scaled, output_height), dtype=np.float32)[:, None, None]
                mask[-ramp.shape[0] :, :, :] *= ramp
            if out_left > 0:
                ramp = np.linspace(0.0, 1.0, min(overlap_scaled, output_width), dtype=np.float32)[None, :, None]
                mask[:, : ramp.shape[1], :] *= ramp
            if out_right < width * scale:
                ramp = np.linspace(1.0, 0.0, min(overlap_scaled, output_width), dtype=np.float32)[None, :, None]
                mask[:, -ramp.shape[1] :, :] *= ramp

            result[out_top:out_bottom, out_left:out_right, :] += tile_output[:output_height, :output_width, :] * mask
            weight[out_top:out_bottom, out_left:out_right, :] += mask

    result /= np.maximum(weight, 1e-6)
    result = (result * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(result)


def upscale_image(session, input_name, output_name, image_path, output_path, tile_size, tile_overlap):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        if tile_size and max(image.size) > tile_size:
            output_image = upscale_tiled(session, input_name, output_name, image, tile_size, tile_overlap)
        else:
            output = run_model(session, input_name, output_name, image)
            output_image = tensor_to_image(output)
    output_image.save(output_path)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    providers = resolve_providers(args.providers)
    print("Using providers:", ", ".join(providers))
    session, input_name, output_name, model_path = load_session(args.variant, providers)
    print(f"Loaded {args.variant} model from {model_path}")

    image_paths = list(iter_input_images(input_dir))
    if not image_paths:
        raise SystemExit(f"No supported images found in {input_dir}")

    for image_path in image_paths:
        output_path = output_dir / image_path.name
        if output_path.exists() and not args.overwrite:
            print(f"Skipping existing {output_path.name}")
            continue
        print(f"Upscaling {image_path.name} -> {output_path.name}")
        upscale_image(session, input_name, output_name, image_path, output_path, args.tile_size, args.tile_overlap)

    print("Done.")


if __name__ == "__main__":
    main()