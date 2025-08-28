import runpod
import torch
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import os
import sys
import tempfile
import cv2

# WAN2.2 spezifische Imports
sys.path.append('/workspace/Wan2.2')

def download_models():
    """Lädt die WAN2.2 Modelle herunter wenn noch nicht vorhanden"""
    import subprocess
    
    model_dir = "/workspace/models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Check ob Modelle schon da sind
    if not os.path.exists(f"{model_dir}/Wan2.2-I2V-A14B"):
        print("Downloading WAN2.2 I2V model...")
        # Huggingface CLI verwenden zum Download
        subprocess.run([
            "huggingface-cli", "download", 
            "Wan-AI/Wan2.2-I2V-A14B",
            "--local-dir", f"{model_dir}/Wan2.2-I2V-A14B"
        ])
    
    if not os.path.exists(f"{model_dir}/Wan2.2-VAE"):
        print("Downloading WAN2.2 VAE...")
        subprocess.run([
            "huggingface-cli", "download",
            "Wan-AI/Wan2.2-VAE",
            "--local-dir", f"{model_dir}/Wan2.2-VAE"
        ])
    
    return f"{model_dir}/Wan2.2-I2V-A14B"

def load_model():
    """Lädt das WAN2.2 Modell"""
    model_path = download_models()
    
    # Import WAN2.2 Module
    from wan2_2_inference import Wan22I2VPipeline
    
    # Pipeline initialisieren
    pipe = Wan22I2VPipeline(
        model_path=model_path,
        device="cuda",
        dtype=torch.float16
    )
    
    return pipe

# Globale Variable für das Modell
model = None

def handler(job):
    """
    RunPod Handler für WAN2.2 I2V
    
    Input Format:
    {
        "input": {
            "image": "base64_encoded_image",
            "prompt": "description of the video",
            "resolution": "1080p",  # oder "720p", "480p"
            "duration": 5,  # Sekunden
            "fps": 24,
            "seed": 42  # optional
        }
    }
    """
    global model
    
    try:
        # Modell beim ersten Aufruf laden
        if model is None:
            print("Loading WAN2.2 model...")
            model = load_model()
            print("Model loaded successfully!")
        
        # Input Parameter extrahieren
        job_input = job['input']
        
        # Bild decodieren
        image_data = base64.b64decode(job_input['image'])
        image = Image.open(BytesIO(image_data)).convert('RGB')
        
        # Auflösung bestimmen
        resolution = job_input.get('resolution', '720p')
        if resolution == '1080p':
            width, height = 1920, 1080
        elif resolution == '720p':
            width, height = 1280, 720
        else:  # 480p
            width, height = 854, 480
        
        # Bild auf Zielauflösung resizen
        image = image.resize((width, height), Image.LANCZOS)
        
        # Video Parameter
        prompt = job_input.get('prompt', '')
        duration = job_input.get('duration', 5)
        fps = job_input.get('fps', 24)
        seed = job_input.get('seed', -1)
        
        # Anzahl Frames berechnen
        num_frames = duration * fps
        
        # Seed setzen wenn angegeben
        if seed > 0:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        print(f"Generating {duration}s video at {resolution} with {fps}fps...")
        print(f"Prompt: {prompt}")
        
        # Video generieren mit WAN2.2
        with torch.no_grad():
            video_frames = model(
                image=image,
                prompt=prompt,
                num_frames=num_frames,
                height=height,
                width=width,
                fps=fps,
                num_inference_steps=50,  # Qualität vs Speed
                guidance_scale=7.0
            )
        
        # Video zu MP4 konvertieren
        temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        
        # OpenCV VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video.name, fourcc, fps, (width, height))
        
        # Frames schreiben
        for frame in video_frames:
            # PIL Image zu numpy array
            frame_np = np.array(frame)
            # RGB zu BGR für OpenCV
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        
        # Video zu Base64 encodieren
        with open(temp_video.name, 'rb') as f:
            video_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Temp Datei löschen
        os.unlink(temp_video.name)
        
        print("Video generation completed!")
        
        return {
            "video": video_base64,
            "resolution": f"{width}x{height}",
            "duration": duration,
            "fps": fps,
            "frames": num_frames
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e)}

# RunPod Serverless starten
runpod.serverless.start({"handler": handler})
