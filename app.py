import os
import io
import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline, CLIPSegProcessor, CLIPSegForImageSegmentation
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

app = FastAPI(title="GeoGuard IEEE Disaster Synthesis Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Loading inference pipeline on {device} ({dtype})...")

# 1. Depth Estimator
depth_estimator = pipeline(
    "depth-estimation", 
    model="Intel/dpt-large", 
    device=0 if device == "cuda" else -1
)

# 2. CLIPSeg Zero-Shot Segmentor
clip_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
clip_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)

# 3. ControlNet + SD Inpainting Pipeline
controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=dtype)
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    controlnet=controlnet,
    torch_dtype=dtype,
    use_safetensors=False
).to(device)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

def extract_depth(image: Image.Image) -> np.ndarray:
    depth = depth_estimator(image)["depth"]
    depth_np = np.array(depth)
    return (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)

def extract_base_mask(image: Image.Image, prompt: str) -> np.ndarray:
    inputs = clip_processor(
        text=[prompt], 
        images=[image], 
        padding="max_length", 
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
    logits = outputs.logits.unsqueeze(1)
    mask_pred = torch.sigmoid(logits)[0][0].cpu().numpy()
    return (mask_pred > 0.28).astype(np.uint8) * 255

def compute_topological_mask(base_mask: np.ndarray, depth_map: np.ndarray, severity: float) -> np.ndarray:
    """Expands mask strictly along lower-elevation gradients (solves Limitation 1)."""
    iterations = int(severity * 12)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated_boundary = cv2.dilate(base_mask, kernel, iterations=iterations)
    
    active_depths = depth_map[base_mask > 0]
    elevation_thresh = (np.percentile(active_depths, 50) if len(active_depths) > 0 else 0.5) + (severity * 0.25)
    
    growth_zone = (dilated_boundary > 0) & (depth_map <= elevation_thresh)
    refined_mask = np.zeros_like(base_mask)
    refined_mask[growth_zone] = 255
    refined_mask[base_mask > 0] = 255
    return refined_mask

def apply_global_atmospheric_weather(image: Image.Image, weather_type: str, severity: float) -> Image.Image:
    """Applies global atmospheric attenuation and weather particles (solves Limitation 2)."""
    img_np = np.array(image).astype(np.float32)
    h, w, _ = img_np.shape

    if weather_type == "heavy_rain":
        # Desaturate and darken sky/ambience
        enhancer = ImageEnhance.Color(image)
        desat = np.array(enhancer.enhance(1.0 - (severity * 0.5))).astype(np.float32)
        
        # Add rain streak layer
        rain_layer = np.zeros((h, w), dtype=np.uint8)
        num_streaks = int(500 * severity)
        for _ in range(num_streaks):
            x = np.random.randint(0, w - 10)
            y = np.random.randint(0, h - 30)
            length = np.random.randint(15, 30)
            cv2.line(rain_layer, (x, y), (x + 4, y + length), 255, 1)
            
        rain_blur = cv2.GaussianBlur(rain_layer, (3, 3), 0)
        for c in range(3):
            desat[:, :, c] = np.clip(desat[:, :, c] + (rain_blur * 0.4), 0, 255)
        return Image.fromarray(desat.astype(np.uint8))

    elif weather_type == "snow_blizzard":
        # Cool blue color cast + haze
        img_np[:, :, 0] = np.clip(img_np[:, :, 0] * 0.85 + (severity * 40), 0, 255)
        img_np[:, :, 1] = np.clip(img_np[:, :, 1] * 0.90 + (severity * 45), 0, 255)
        img_np[:, :, 2] = np.clip(img_np[:, :, 2] * 1.05 + (severity * 60), 0, 255)
        return Image.fromarray(img_np.astype(np.uint8))

    return image

@app.post("/api/generate")
async def generate_disaster(
    image_file: UploadFile = File(...),
    disaster_type: str = Form("flood"),          # flood, collapse, heavy_rain, snow_blizzard
    target_prompt: str = Form("river, road, ground"),
    disaster_prompt: str = Form("submerged flooded terrain muddy turbulent brown water debris"),
    severity: float = Form(0.7)                   # 0.0 to 1.0
):
    init_img = Image.open(io.BytesIO(await image_file.read())).convert("RGB").resize((512, 512))
    
    # 1. Depth Guidance Map
    depth_norm = extract_depth(init_img)
    depth_control_img = Image.fromarray((depth_norm * 255).astype(np.uint8)).convert("RGB")
    
    # 2. Base Segmentation Mask
    base_mask = extract_base_mask(init_img, target_prompt)
    
    # 3. Depth-Regulated Mask Expansion
    regulated_mask = compute_topological_mask(base_mask, depth_norm, severity)
    regulated_mask_img = Image.fromarray(regulated_mask)

    # 4. Latent Inpainting
    generator = torch.Generator(device=device).manual_seed(42)
    result = pipe(
        prompt=disaster_prompt,
        negative_prompt="blurry, distorted buildings, deformed structures, floating artifacts, clean water",
        image=init_img,
        mask_image=regulated_mask_img,
        control_image=depth_control_img,
        num_inference_steps=25,
        guidance_scale=7.5,
        controlnet_conditioning_scale=0.8,
        generator=generator
    ).images[0]

    # 5. Dual-Branch Atmospheric Blend (Limitation 2 Resolution)
    if disaster_type in ["heavy_rain", "snow_blizzard"]:
        result = apply_global_atmospheric_weather(result, disaster_type, severity)

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")

@app.get("/api/health")
def health():
    return {"status": "ready", "device": device, "dtype": str(dtype)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)