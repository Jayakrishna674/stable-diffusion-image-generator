# Stable Diffusion Image Generator

Generate images from text prompts using the Stable Diffusion 2.1 model and Hugging Face Diffusers.

## Features

- Text-to-image generation from a user-provided prompt
- Random seed generated for each image
- Uses CUDA when a compatible GPU is available
- Displays the generated image in a Jupyter Notebook or Google Colab

## Project files

- `main.py` — loads the model and generates an image
- `requirements.txt` — Python dependencies
- `.gitignore` — excludes virtual environments and temporary files

## Requirements

- Python 3.10 or 3.11 is recommended
- Internet connection for the initial model download
- A CUDA-compatible GPU with sufficient VRAM is recommended; CPU execution can be very slow
- A Hugging Face account or access approval may be needed if the model repository's terms require it

## Setup

Create and activate a virtual environment if desired, then install dependencies:

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install packages:

```bash
pip install -r requirements.txt
```

## Run

This script uses `IPython.display`, so run it in Jupyter Notebook or Google Colab. In a notebook cell, use:

```python
%run main.py
```

Enter a descriptive prompt when asked. The first run downloads the model files, which can take time and disk space.

## Model

`stabilityai/stable-diffusion-2-1`

Check the model card and license before using or distributing the model or generated outputs:
https://huggingface.co/stabilityai/stable-diffusion-2-1

## Notes

- The script uses float16 on CUDA and float32 on CPU to improve compatibility.
- Model weights are not included in this repository; they are downloaded by Diffusers when needed.
- Do not commit virtual environments, model weight files, access tokens, or other secrets.
