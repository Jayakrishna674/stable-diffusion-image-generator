import random
import sys

import torch
from diffusers import StableDiffusionPipeline
from IPython.display import display


MODEL_ID = "stabilityai/stable-diffusion-2-1"


def load_pipeline():
    """Load Stable Diffusion and use CUDA when available."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # float16 is best suited to supported GPUs; use float32 on CPU.
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    )
    pipe = pipe.to(device)
    return pipe


def generate_image(pipe, prompt):
    """Generate an image from a text prompt and return it with its seed."""
    seed = random.randint(0, sys.maxsize)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(seed)

    with torch.no_grad():
        image = pipe(
            prompt,
            num_inference_steps=50,
            generator=generator,
        ).images[0]

    return image, seed


def main():
    print(f"Loading {MODEL_ID}...")
    pipe = load_pipeline()

    prompt = input("Enter your prompt: ").strip()
    if not prompt:
        print("Please enter a non-empty prompt.")
        return

    image, seed = generate_image(pipe, prompt)
    print(f"Image generated. Seed: {seed}")
    display(image)


if __name__ == "__main__":
    main()
