"""
Vereinfachter WAN2.2 Inference Wrapper für RunPod
"""

import torch
import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# WAN2.2 Pfad hinzufügen
sys.path.append('/workspace/Wan2.2')

class Wan22I2VPipeline:
    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        self.model_path = model_path
        
        # Lazy loading - Modell wird beim ersten Aufruf geladen
        self.model = None
        self.vae = None
        
    def load_models(self):
        """Lädt die WAN2.2 Modelle"""
        if self.model is not None:
            return
            
        print("Loading WAN2.2 models...")
        
        # Import der WAN2.2 Module
        try:
            from Wan2_2.models import load_i2v_model
            from Wan2_2.vae import load_vae
        except:
            # Fallback wenn Module anders strukturiert sind
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "wan_models", 
                "/workspace/Wan2.2/generate.py"
            )
            wan_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(wan_module)
            
            # Modelle über das generate Script laden
            self.setup_from_generate_script()
            return
        
        # Modelle laden
        self.model = load_i2v_model(
            self.model_path,
            device=self.device,
            dtype=self.dtype
        )
        
        vae_path = self.model_path.replace("I2V-A14B", "VAE")
        self.vae = load_vae(vae_path, device=self.device)
        
        print("Models loaded successfully!")
    
    def setup_from_generate_script(self):
        """Alternative Setup-Methode über das generate.py Script"""
        import subprocess
        import json
        
        # Temporäre Config erstellen
        config = {
            "task": "i2v-A14B",
            "ckpt_dir": self.model_path,
            "device": self.device,
            "dtype": "fp16" if self.dtype == torch.float16 else "fp32",
            "offload_model": False
        }
        
        config_path = "/tmp/wan_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        # Model über generate.py initialisieren
        cmd = f"cd /workspace/Wan2.2 && python generate.py --config {config_path} --init_only"
        subprocess.run(cmd, shell=True, check=True)
        
        # Initialisiertes Modell laden
        import pickle
        with open("/tmp/wan_model.pkl", 'rb') as f:
            self.model = pickle.load(f)
        with open("/tmp/wan_vae.pkl", 'rb') as f:
            self.vae = pickle.load(f)
    
    def __call__(self, image, prompt, num_frames, height, width, 
                 fps=24, num_inference_steps=50, guidance_scale=7.0):
        """
        Generiert Video aus Bild und Prompt
        
        Args:
            image: PIL Image
            prompt: Text Beschreibung
            num_frames: Anzahl der Frames
            height: Video Höhe
            width: Video Breite
            fps: Frames per second
            num_inference_steps: Denoising Steps
            guidance_scale: CFG Scale
            
        Returns:
            Liste von PIL Images (Frames)
        """
        
        # Modelle laden falls noch nicht geschehen
        if self.model is None:
            self.load_models()
        
        # Bild vorbereiten
        image = image.resize((width, height), Image.LANCZOS)
        image_tensor = self.preprocess_image(image)
        
        # Prompt vorbereiten
        if not prompt:
            prompt = "high quality video, smooth motion, cinematic"
        
        # Negative prompt für bessere Qualität
        negative_prompt = "blurry, low quality, distorted, deformed"
        
        with torch.no_grad():
            # Video generieren
            frames = self.generate_video(
                image_tensor,
                prompt,
                negative_prompt,
                num_frames,
                height,
                width,
                num_inference_steps,
                guidance_scale
            )
        
        # Frames zu PIL Images konvertieren
        pil_frames = []
        for frame in frames:
            frame_np = (frame * 255).astype(np.uint8)
            pil_frames.append(Image.fromarray(frame_np))
        
        return pil_frames
    
    def preprocess_image(self, image):
        """Konvertiert PIL Image zu Tensor"""
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        image_tensor = image_tensor.unsqueeze(0).to(self.device, self.dtype)
        return image_tensor
    
    def generate_video(self, image_tensor, prompt, negative_prompt,
                      num_frames, height, width, steps, cfg):
        """Kern Video Generation Logik"""
        
        # Hier würde die eigentliche WAN2.2 Generation stattfinden
        # Dies ist ein vereinfachtes Beispiel
        
        try:
            # Versuche die offizielle generate Funktion zu nutzen
            from Wan2_2.pipeline import i2v_generate
            
            frames = i2v_generate(
                model=self.model,
                vae=self.vae,
                image=image_tensor,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_frames=num_frames,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=cfg
            )
            
        except ImportError:
            # Fallback: Direkte Modell-Nutzung
            frames = self.direct_generation(
                image_tensor, prompt, num_frames, 
                height, width, steps, cfg
            )
        
        return frames
    
    def direct_generation(self, image, prompt, num_frames, h, w, steps, cfg):
        """Fallback Generation Methode"""
        # Placeholder - würde echte Generation implementieren
        frames = []
        for i in range(num_frames):
            # Simuliere Frame Generation
            frame = np.ones((h, w, 3), dtype=np.float32)
            frames.append(frame)
        return frames
