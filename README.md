![Fractal Preview](https://github.com/anttiluode/fractal_forest/fractal.png

# Fractal Art Creator

A simple Tkinter GUI that lets you:

1. **Generate** a procedurally-fractal “forest” scene (trees, ferns, stones, clouds, rivers)  
2. **Transform** it on the right via Stable Diffusion Img2Img with configurable prompt, strength, guidance, latent- and memory-blending  

## Prerequisites

- Python 3.8+  
- A CUDA-enabled GPU (optional; CPU will work but much slower)  
- Webcam **not** required (the script uses a matplotlib figure as its “canvas”)

## Installation

```bash
git clone https://github.com/anttiluode/fractal_forest.git
cd fractal_forest
pip install -r requirements.txt
```

## Usage

```bash
python fractal_forest.py
```

1. Click **Generate Fractal** to redraw the left pane.  
2. Edit your SD prompt & sliders, then click **Generate SD** to run Img2Img on that fractal.  
3. Repeat as desired.  

## Files

- `fractal_forest.py` — main GUI script  
- `requirements.txt` — Python dependencies  
- `README.md` — this file  

Feel free to tweak the procedural parameters in `fractal_forest.py` or swap in your own Stable Diffusion weights!
