#!/usr/bin/env python3
"""
fractal_world_sd.py

A Fractal World Generator + Stable Diffusion Img2Img wrapper in one Tkinter GUI:
 - left pane: live fractal world (trees, ferns, stones, clouds, rivers)
 - right pane: SD Img2Img output of whatever is in the left pane
 - controls for prompt, strength, guidance, latent & memory blending
"""

import sys, types, io, threading, time
# -- dummy-triton autotuner patch (to satisfy diffusers/triton) --
try:
    import triton.runtime
except ImportError:
    sys.modules["triton"] = types.ModuleType("triton")
    sys.modules["triton.runtime"] = types.ModuleType("triton.runtime")
import triton.runtime
if not hasattr(triton.runtime, "Autotuner"):
    class DummyAutotuner:
        def __init__(self,*a,**k): pass
        def tune(self,*a,**k): return None
    triton.runtime.Autotuner = DummyAutotuner

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2, numpy as np, random
import torch
from diffusers import StableDiffusionImg2ImgPipeline
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# === Procedural fractal world functions ===
def midpoint_displacement(n=512, roughness=1.0):
    arr = np.zeros(n+1)
    arr[0], arr[-1] = random.uniform(0.2,0.8), random.uniform(0.2,0.8)
    step, scale = n, roughness
    while step > 1:
        half = step//2
        for i in range(0,n,step):
            mid = (arr[i]+arr[i+step])/2 + random.uniform(-scale,scale)
            arr[i+half] = mid
        scale *= 0.6; step//=2
    return arr

fern_rules = [
    (0.85,0.04,-0.04,0.85,0,1.6),
    (0.20,-0.26,0.23,0.22,0,1.6),
    (-0.15,0.28,0.26,0.24,0,0.44),
    (0.00,0.00,0.00,0.16,0,0)
]
def draw_fern(ax, x0, y0, scale=50, n=1000):
    x,y = 0,0; pts=[]
    for _ in range(n):
        a,b,c,d,e,f = random.choice(fern_rules)
        x,y = a*x + b*y + e, c*x + d*y + f
        pts.append((x0 + x*scale, y0 + y*scale))
    arr = np.array(pts)
    ax.scatter(arr[:,0], arr[:,1], s=0.1, c='#228B22', alpha=0.8)

def draw_tree(ax, x,y, angle, depth, length, color_trunk='saddlebrown'):
    if depth==0: return
    rad = np.deg2rad(angle)
    x2 = x + np.cos(rad)*length
    y2 = y + np.sin(rad)*length
    ax.plot([x,x2],[y,y2],c=color_trunk,lw=depth)
    draw_tree(ax, x2,y2, angle+random.uniform(15,30), depth-1, length*0.7)
    draw_tree(ax, x2,y2, angle-random.uniform(15,30), depth-1, length*0.7)

def draw_river(ax, x0,x1,y0,y1, roughness=0.02, gen=5):
    pts=[(x0,y0),(x1,y1)]
    def subdiv(points,g):
        if g==0: return points
        new=[]
        for p0,p1 in zip(points,points[1:]):
            x,y=p0; x2,y2=p1
            mx,my=(x+x2)/2,(y+y2)/2
            dx,dy=x2-x,y2-y
            perp=np.array([-dy,dx]); perp/=np.linalg.norm(perp)
            d=random.uniform(-roughness,roughness)
            new.append((x,y))
            new.append((mx+perp[0]*d, my+perp[1]*d))
        new.append(points[-1])
        return subdiv(new,g-1)
    path=subdiv(pts,gen)
    xs,ys=zip(*path)
    ax.plot(xs,ys,c='steelblue',lw=2,alpha=0.6)

# === Main GUI ===
class FractalWorldSD:
    def __init__(self, root):
        self.root = root
        root.title("Fractal World + Stable Diffusion")
        self._build_controls()
        self._build_canvases()
        # load SDim2im once
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            torch_dtype=torch.float16 if device=="cuda" else torch.float32
        ).to(device)
        self.memory = None

    def _build_controls(self):
        ctrl = ttk.Frame(self.root); ctrl.pack(fill=tk.X, pady=4)
        # fractal params not exposed here (use generate button)
        ttk.Button(ctrl, text="Generate Fractal", command=self._on_generate).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="Generate SD", command=self._on_sd).pack(side=tk.LEFT, padx=4)

        # SD sliders
        def mk_scale(label, var, frm, **kwargs):
            ttk.Label(frm, text=label).pack(side=tk.LEFT, padx=4)
            ttk.Scale(frm, **kwargs, variable=var, orient=tk.HORIZONTAL, length=120).pack(side=tk.LEFT)
        self.prompt = tk.StringVar(value="A fantasy landscape, ethereal and misty")
        ttk.Label(ctrl, text="Prompt:").pack(side=tk.LEFT, padx=4)
        ttk.Entry(ctrl, textvariable=self.prompt, width=30).pack(side=tk.LEFT)
        self.strength = tk.DoubleVar(value=0.75)
        mk_scale("Strength", self.strength, ctrl, from_=0.1, to=1.0)
        self.guidance = tk.DoubleVar(value=7.5)
        mk_scale("Guidance", self.guidance, ctrl, from_=1.0, to=20.0)
        self.latent_blend = tk.DoubleVar(value=0.7)
        mk_scale("LatentBlend", self.latent_blend, ctrl, from_=0.0, to=1.0)
        self.memory_blend = tk.DoubleVar(value=0.5)
        mk_scale("MemoryBlend", self.memory_blend, ctrl, from_=0.0, to=1.0)

    def _build_canvases(self):
        pane = ttk.Frame(self.root); pane.pack(fill=tk.BOTH, expand=True)
        # fractal figure
        self.fig = plt.Figure(figsize=(4,4))
        self.ax = self.fig.add_subplot(111); self.ax.axis('off')
        self.canvasF = FigureCanvasTkAgg(self.fig, master=pane)
        self.canvasF.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # SD output
        self.sd_label = ttk.Label(pane)
        self.sd_label.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def _on_generate(self):
        # redraw fractal world
        self.ax.cla(); self.ax.axis('off')
        # sky
        sky = np.linspace(0.6,1,200)
        self.ax.imshow(np.vstack([sky,sky]), extent=[0,1,0.4,1], aspect='auto', cmap='Blues')
        # clouds
        for _ in range(5):
            cx,cy = random.uniform(0.1,0.9), random.uniform(0.7,0.95)
            for _ in range(random.randint(5,10)):
                dx,dy=random.uniform(-.05,.05),random.uniform(-.02,.02)
                r=random.uniform(.03,.07)
                self.ax.add_patch(mpatches.Ellipse((cx+dx,cy+dy), r, r*.6, color='white', alpha=0.5))
        # mountains
        m = midpoint_displacement(512,0.3)
        xs = np.linspace(0,1,len(m))
        self.ax.fill_between(xs, m*0.1+0.4, 0.4, color='grey')
        # rivers
        for _ in range(2):
            px=random.uniform(0.1,0.9)
            idx=int(px*(len(m)-1))
            y0=m[idx]*0.1+0.4
            draw_river(self.ax, px, px+0.02, y0, 0.0)
        # ground
        self.ax.fill_between([0,1],[0,0],[0.4,0.4], color='darkgreen')
        # ferns, stones, trees
        for _ in range(50): draw_fern(self.ax, random.uniform(0,1), random.uniform(0.02,0.3), scale=random.uniform(0.02,0.05))
        for _ in range(200):
            x,y = random.uniform(0,1), random.uniform(0,0.3)
            size=random.uniform(0.005,0.02)
            pts=midpoint_displacement(8,0.2)
            ang=np.linspace(0,2*np.pi,len(pts),endpoint=False)
            xs=x+size*np.cos(ang)*pts; ys=y+size*np.sin(ang)*pts
            self.ax.add_patch(mpatches.Polygon(np.column_stack((xs,ys)), closed=True, color='dimgray'))
        for _ in range(50): draw_tree(self.ax, random.uniform(0,1), 0.4, 90, random.randint(3,6), random.uniform(0.05,0.15))
        self.canvasF.draw()

    def _on_sd(self):
        # grab fractal image from the mpl figure
        buf = io.BytesIO()
        self.fig.savefig(buf, format='png', dpi=256, facecolor=self.fig.get_facecolor())
        buf.seek(0)
        src = Image.open(buf).convert('RGB').resize((512,512))
        prompt = self.prompt.get()
        strength = float(self.strength.get())
        guidance = float(self.guidance.get())
        latent_blend = float(self.latent_blend.get())
        memory_blend = float(self.memory_blend.get())

        def worker():
            # run SD Img2Img
            try:
                out = self.pipe(prompt=prompt, image=src, strength=strength, guidance_scale=guidance).images[0]
            except Exception as e:
                print("SD error:", e)
                out = src
            sd_np = np.array(out).astype(np.float32)
            src_np = np.array(src).astype(np.float32)
            blended = latent_blend*sd_np + (1-latent_blend)*src_np
            blended = np.clip(blended,0,255).astype(np.uint8)
            if self.memory is None:
                self.memory = blended
            else:
                self.memory = (memory_blend*self.memory + (1-memory_blend)*blended).astype(np.uint8)
            final = Image.fromarray(self.memory)
            tkimg = ImageTk.PhotoImage(final)
            # update in main thread
            self.sd_label.after(0, lambda: self.sd_label.configure(image=tkimg) or setattr(self.sd_label,'image',tkimg))

        threading.Thread(target=worker, daemon=True).start()

if __name__=="__main__":
    root = tk.Tk()
    app = FractalWorldSD(root)
    root.protocol("WM_DELETE_WINDOW", root.quit)
    root.mainloop()
