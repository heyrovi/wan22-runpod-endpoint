import runpod
import torch
import os
import sys
import base64
from io import BytesIO
from PIL import Image
import subprocess
import json
import tempfile

def handler(job):
    """
    Vereinfachter RunPod Handler für WAN2.2
    Nutzt das offizielle WAN2.2 CLI direkt
    """
    
    try:
        job_input = job['input']
        
        # Base64 Bild decodieren und speichern
        image_data = base64.b64decode(job_input['image'])
        input_image = Image.open(BytesIO(image_data)).convert('RGB')
        
        # Temporäre Dateien
        temp_image = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        
        # Bild speichern
        input_image.save(temp_image.name)
        
        # Parameter
        prompt = job_input.get('prompt', 'high quality video')
        resolution = job_input.get('resolution', '720p')
        duration = job_input.get('duration', 5)
        
        # Auflösung mapping
        if resolution == '1080p':
            size = '1920*1080'
        elif resolution == '720p':
            size = '1280*720'
        else:
            size = '854*480'
        
        # WAN2.2 Command zusammenbauen
        # Nutze das 5B Modell (TI2V) das sowohl Text als auch Image unterstützt
        cmd = [
            'python', '/workspace/Wan2.2/generate.py',
            '--task', 'ti2v-5B',
            '--size', size,
            '--ckpt_dir', '/workspace/models/Wan2.2-TI2V-5B',
            '--image', temp_image.name,
            '--prompt', prompt,
            '--output', temp_video.name,
            '--seed', str(job_input.get('seed', 42)),
            '--offload_model', 'True'  # Wichtig für GPU Memory
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        # WAN2.2 ausführen
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd='/workspace/Wan2.2'
        )
        
        if result.returncode != 0:
            print(f"Error output: {result.stderr}")
            return {"error": f"WAN2.2 generation failed: {result.stderr}"}
        
        # Video zu Base64
        with open(temp_video.name, 'rb') as f:
            video_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Cleanup
        os.unlink(temp_image.name)
        os.unlink(temp_video.name)
        
        return {
            "video": video_base64,
            "resolution": size.replace('*', 'x'),
            "duration": duration,
            "prompt": prompt
        }
        
    except Exception as e:
        print(f"Handler error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# Modelle beim Start herunterladen
def download_models():
    """Download WAN2.2 models if not present"""
    
    model_dir = "/workspace/models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Clone WAN2.2 repo wenn nicht vorhanden
    if not os.path.exists('/workspace/Wan2.2'):
        print("Cloning WAN2.2 repository...")
        subprocess.run([
            'git', 'clone', 
            'https://github.com/Wan-Video/Wan2.2.git',
            '/workspace/Wan2.2'
        ], check=True)
    
    # Download TI2V-5B Model (kleiner und schneller)
    if not os.path.exists(f"{model_dir}/Wan2.2-TI2V-5B"):
        print("Downloading WAN2.2 TI2V-5B model...")
        subprocess.run([
            'huggingface-cli', 'download',
            'Wan-AI/Wan2.2-TI2V-5B',
            '--local-dir', f'{model_dir}/Wan2.2-TI2V-5B',
            '--local-dir-use-symlinks', 'False'
        ], check=True)
    
    # Download VAE
    if not os.path.exists(f"{model_dir}/Wan2.2-VAE"):
        print("Downloading WAN2.2 VAE...")
        subprocess.run([
            'huggingface-cli', 'download',
            'Wan-AI/Wan2.2-VAE',
            '--local-dir', f'{model_dir}/Wan2.2-VAE',
            '--local-dir-use-symlinks', 'False'
        ], check=True)
    
    print("Models ready!")

# Download models on container start
print("Initializing WAN2.2...")
download_models()
print("WAN2.2 initialized!")

# Start RunPod serverless
runpod.serverless.start({"handler": handler})
