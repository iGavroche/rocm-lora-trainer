#!/usr/bin/env python3
import argparse
import base64
import json
import math
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

from musubi_tuner.dataset import image_video_dataset
from musubi_tuner.qwen_image.qwen_image_utils import load_qwen2_5_vl

import logging

try:
    import toml
except ImportError:
    try:
        import tomli as toml
    except ImportError:
        toml = None

try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


IMAGE_FACTOR = 28  # The image size must be divisible by this factor
DEFAULT_MAX_SIZE = 1280

DEFAULT_PROMPT = """# Image Annotator
You are a professional image annotator. Please complete the following task based on the input image.
## Create Image Caption
1. Write the caption using natural, descriptive text without structured formats or rich text.
2. Enrich caption details by including: object attributes, vision relations between objects, and environmental details.
3. Identify the text visible in the image, without translation or explanation, and highlight it in the caption with quotation marks.
4. Maintain authenticity and accuracy, avoid generalizations."""

# Default config values for LM Studio
DEFAULT_CONFIG = {
    "api_url": "http://127.0.0.1:1234/v1/chat/completions",
    "api_key": "lm-studio",  # LM Studio doesn't require a real key, but some clients expect it
    "model": "qwen/qwen3-vl-4b",
    "max_new_tokens": 1024,
    "temperature": 0.7,
    "use_api": True,
    "image_dir": "myface",
    "output_format": "text",
    "max_size": 1280,
    "prompt": DEFAULT_PROMPT,
}


def load_config(config_path: str) -> dict:
    """Load configuration from TOML file"""
    if toml is None:
        raise ImportError("toml or tomli library is required for config file support. Install with: pip install toml or pip install tomli")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = toml.load(f)
    
    # Merge with defaults
    merged_config = DEFAULT_CONFIG.copy()
    if "captioning" in config:
        merged_config.update(config["captioning"])
    
    return merged_config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate captions for images using Qwen2.5-VL or LM Studio API")

    parser.add_argument("--config", type=str, help="Path to TOML config file (overrides defaults)")
    parser.add_argument("--image_dir", type=str, help="Path to directory containing images")
    parser.add_argument("--model_path", type=str, help="Path to Qwen2.5-VL model (required if not using API)")
    parser.add_argument("--output_file", type=str, help="Output JSONL file path (required for 'jsonl' format)")
    parser.add_argument("--max_new_tokens", type=int, help="Maximum number of new tokens to generate")
    parser.add_argument(
        "--prompt", type=str, help="Custom prompt for caption generation (supports \\n for newlines)"
    )
    parser.add_argument(
        "--max_size",
        type=int,
        help=f"Maximum image size. The images are resized to fit the total pixel area within (max_size x max_size)",
    )
    parser.add_argument("--fp8_vl", action="store_true", help="Load Qwen2.5-VL model in fp8 precision")
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["jsonl", "text"],
        help="Output format: 'jsonl' for JSONL file or 'text' for individual text files",
    )
    # API options
    parser.add_argument("--use_api", action="store_true", help="Use LM Studio API instead of local model")
    parser.add_argument("--api_url", type=str, help="LM Studio API URL (default: http://127.0.0.1:1234/v1/chat/completions)")
    parser.add_argument("--api_key", type=str, help="API key (optional for LM Studio)")
    parser.add_argument("--api_model", type=str, help="Model name for API (default: qwen/qwen3-vl-4b)")
    parser.add_argument("--temperature", type=float, help="Temperature for API generation")

    args = parser.parse_args()
    
    # Load config file if provided
    config = None
    if args.config:
        config = load_config(args.config)
        # Override args with config values if not explicitly set
        if args.image_dir is None and "image_dir" in config:
            args.image_dir = config["image_dir"]
        if args.output_format is None and "output_format" in config:
            args.output_format = config["output_format"]
        if args.max_new_tokens is None and "max_new_tokens" in config:
            args.max_new_tokens = config["max_new_tokens"]
        if args.prompt is None and "prompt" in config:
            args.prompt = config["prompt"]
        if args.max_size is None and "max_size" in config:
            args.max_size = config["max_size"]
        if args.api_url is None and "api_url" in config:
            args.api_url = config["api_url"]
        if args.api_key is None and "api_key" in config:
            args.api_key = config["api_key"]
        if args.api_model is None and "model" in config:
            args.api_model = config["model"]
        if args.temperature is None and "temperature" in config:
            args.temperature = config["temperature"]
        # use_api is set by action="store_true", so check if it was explicitly set
        if not args.use_api and "use_api" in config:
            args.use_api = config["use_api"]
    
    # Apply defaults if not set
    if args.image_dir is None:
        args.image_dir = DEFAULT_CONFIG["image_dir"]
    if args.output_format is None:
        args.output_format = DEFAULT_CONFIG["output_format"]
    if args.max_new_tokens is None:
        args.max_new_tokens = DEFAULT_CONFIG["max_new_tokens"]
    if args.prompt is None:
        args.prompt = DEFAULT_CONFIG["prompt"]
    if args.max_size is None:
        args.max_size = DEFAULT_CONFIG["max_size"]
    if args.api_url is None:
        args.api_url = DEFAULT_CONFIG["api_url"]
    if args.api_key is None:
        args.api_key = DEFAULT_CONFIG["api_key"]
    if args.api_model is None:
        args.api_model = DEFAULT_CONFIG["model"]
    if args.temperature is None:
        args.temperature = DEFAULT_CONFIG.get("temperature", 0.7)
    # use_api defaults to True (API mode) unless explicitly disabled
    if not args.use_api and args.model_path is None:
        args.use_api = DEFAULT_CONFIG["use_api"]
    
    # Validate arguments
    if not args.use_api and args.model_path is None:
        raise ValueError("--model_path is required when not using API (or set --use_api)")
    
    return args


def load_model_and_processor(model_path: str, device: torch.device, max_size: int = DEFAULT_MAX_SIZE, fp8_vl: bool = False):
    """Load Qwen2.5-VL model and processor"""
    logger.info(f"Loading model from: {model_path}")

    min_pixels = 256 * IMAGE_FACTOR * IMAGE_FACTOR  # this means 256x256 is the minimum input size
    max_pixels = max_size * IMAGE_FACTOR * IMAGE_FACTOR

    # We don't have configs in model_path, so we use defaults from Hugging Face
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

    # Use load_qwen2_5_vl function from qwen_image_utils
    dtype = torch.float8_e4m3fn if fp8_vl else torch.bfloat16
    _, model = load_qwen2_5_vl(model_path, dtype=dtype, device=device, disable_mmap=True)

    model.eval()

    logger.info(f"Model loaded successfully on device: {model.device}")
    return processor, model


def resize_image(image: Image.Image, max_size: int = DEFAULT_MAX_SIZE) -> Image.Image:
    """Resize image to a suitable resolution"""
    min_area = 256 * 256
    max_area = max_size * max_size
    width, height = image.size
    width_rounded = int((width / IMAGE_FACTOR) + 0.5) * IMAGE_FACTOR
    height_rounded = int((height / IMAGE_FACTOR) + 0.5) * IMAGE_FACTOR

    bucket_resos = []
    if width_rounded * height_rounded < min_area:
        # Scale up to min area
        scale_factor = math.sqrt(min_area / (width_rounded * height_rounded))
        new_width = math.ceil(width * scale_factor / IMAGE_FACTOR) * IMAGE_FACTOR
        new_height = math.ceil(height * scale_factor / IMAGE_FACTOR) * IMAGE_FACTOR

        # Add to bucket resolutions: default and slight variations for keeping aspect ratio
        bucket_resos.append((new_width, new_height))
        bucket_resos.append((new_width + IMAGE_FACTOR, new_height))
        bucket_resos.append((new_width, new_height + IMAGE_FACTOR))
    elif width_rounded * height_rounded > max_area:
        # Scale down to max area
        scale_factor = math.sqrt(max_area / (width_rounded * height_rounded))
        new_width = math.floor(width * scale_factor / IMAGE_FACTOR) * IMAGE_FACTOR
        new_height = math.floor(height * scale_factor / IMAGE_FACTOR) * IMAGE_FACTOR

        # Add to bucket resolutions: default and slight variations for keeping aspect ratio
        bucket_resos.append((new_width, new_height))
        bucket_resos.append((new_width - IMAGE_FACTOR, new_height))
        bucket_resos.append((new_width, new_height - IMAGE_FACTOR))
    else:
        # Keep original resolution, but add slight variations for keeping aspect ratio
        bucket_resos.append((width_rounded, height_rounded))
        bucket_resos.append((width_rounded - IMAGE_FACTOR, height_rounded))
        bucket_resos.append((width_rounded, height_rounded - IMAGE_FACTOR))
        bucket_resos.append((width_rounded + IMAGE_FACTOR, height_rounded))
        bucket_resos.append((width_rounded, height_rounded + IMAGE_FACTOR))

    # Min/max area filtering
    bucket_resos = [(w, h) for w, h in bucket_resos if w * h >= min_area and w * h <= max_area]

    # Select bucket which has a nearest aspect ratio
    aspect_ratio = width / height
    bucket_resos.sort(key=lambda x: abs((x[0] / x[1]) - aspect_ratio))
    bucket_reso = bucket_resos[0]

    # Resize to bucket
    image_np = image_video_dataset.resize_image_to_bucket(image, bucket_reso)

    # Convert back to PIL
    image = Image.fromarray(image_np)
    return image


def image_to_base64(image: Image.Image, max_size: int = DEFAULT_MAX_SIZE) -> str:
    """Convert PIL Image to base64 string, resizing if needed"""
    # Resize image if needed
    image_resized = resize_image(image, max_size=max_size)
    
    # Convert to base64
    buffered = BytesIO()
    image_resized.save(buffered, format="JPEG", quality=95)
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_base64


def generate_caption_api(
    image_path: str,
    api_url: str,
    api_key: str,
    api_model: str,
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    max_size: int = DEFAULT_MAX_SIZE,
) -> str:
    """Generate caption using LM Studio API (OpenAI-compatible)"""
    if requests is None:
        raise ImportError("requests library is required for API mode. Install with: pip install requests")
    
    # Load and convert image to base64
    image = Image.open(image_path).convert("RGB")
    img_base64 = image_to_base64(image, max_size=max_size)
    
    # Prepare API request
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    # LM Studio uses OpenAI-compatible format
    payload = {
        "model": api_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
    }
    
    # Make API request
    response = requests.post(api_url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    
    result = response.json()
    
    # Extract caption from response
    if "choices" in result and len(result["choices"]) > 0:
        caption = result["choices"][0]["message"]["content"]
        return caption.strip()
    else:
        raise ValueError(f"Unexpected API response format: {result}")
    
    return ""


def generate_caption(
    processor,
    model,
    image_path: str,
    device: torch.device,
    max_new_tokens: int,
    prompt: str = DEFAULT_PROMPT,
    max_size: int = DEFAULT_MAX_SIZE,
    fp8_vl: bool = False,
) -> str:
    """Generate caption for a single image using local model"""
    # Load and process image
    image = Image.open(image_path).convert("RGB")

    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs = resize_image(image, max_size=max_size)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(device)

    # Generate caption with fp8 support
    if fp8_vl:
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=processor.tokenizer.eos_token_id)
    else:
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=processor.tokenizer.eos_token_id)

    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    caption = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    # Return as string instead of list
    return caption[0] if caption else ""


def process_images(args):
    """Main processing function"""
    # Validate arguments
    if args.output_format == "jsonl" and not args.output_file:
        raise ValueError("--output_file is required when --output_format is 'jsonl'")

    # Process custom prompt - replace \n with actual newlines
    if args.prompt:
        args.prompt = args.prompt.replace("\\n", "\n")
        prompt = args.prompt
    else:
        prompt = DEFAULT_PROMPT

    # Get image files
    image_files = image_video_dataset.glob_images(args.image_dir)
    logger.info(f"Found {len(image_files)} image files")
    
    # Determine mode
    use_api = args.use_api
    logger.info(f"Mode: {'API (LM Studio)' if use_api else 'Local Model'}")
    logger.info(f"Output format: {args.output_format}")
    
    if use_api:
        logger.info(f"API URL: {args.api_url}")
        logger.info(f"API Model: {args.api_model}")
        processor = None
        model = None
        device = None
    else:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        if args.fp8_vl:
            logger.info("Using fp8 precision for model")
        
        # Load model and processor
        processor, model = load_model_and_processor(args.model_path, device, args.max_size, args.fp8_vl)

    # Create output directory if needed for JSONL format
    if args.output_format == "jsonl":
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process images and write results
    if args.output_format == "jsonl":
        # JSONL output format
        with open(args.output_file, "w", encoding="utf-8") as f:
            for image_path in tqdm(image_files, desc="Generating captions"):
                if use_api:
                    caption = generate_caption_api(
                        image_path, args.api_url, args.api_key, args.api_model,
                        prompt, args.max_new_tokens, args.temperature, args.max_size
                    )
                else:
                    caption = generate_caption(
                        processor, model, image_path, device, args.max_new_tokens, prompt, args.max_size, args.fp8_vl
                    )

                # Create JSONL entry
                entry = {"image_path": image_path, "caption": caption}

                # Write to file
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()  # Ensure data is written immediately

        logger.info(f"Caption generation completed. Results saved to: {args.output_file}")

    else:
        # Text file output format
        for image_path in tqdm(image_files, desc="Generating captions"):
            if use_api:
                caption = generate_caption_api(
                    image_path, args.api_url, args.api_key, args.api_model,
                    prompt, args.max_new_tokens, args.temperature, args.max_size
                )
            else:
                caption = generate_caption(
                    processor, model, image_path, device, args.max_new_tokens, prompt, args.max_size, args.fp8_vl
                )

            # Generate text file path: same directory as image, with .txt extension
            image_path_obj = Path(image_path)
            text_file_path = image_path_obj.with_suffix(".txt")

            # Write caption to text file
            with open(text_file_path, "w", encoding="utf-8") as f:
                f.write(caption)

        logger.info("Caption generation completed. Text files saved alongside each image.")


def main():
    """Main function"""
    args = parse_args()
    process_images(args)


if __name__ == "__main__":
    main()
