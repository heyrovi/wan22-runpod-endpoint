#!/usr/bin/env python3
import runpod
import os
import sys
import torch
import base64
import tempfile
import subprocess
from PIL import Image
from io import BytesIO

sys.path.append('/Wan2.2')
os.chdir('/Wan2.2')

def handler(job):
    """WAN2.2 I2V Handler für 1080p Video Generation"""
    
    job_input = job['input']
    
    # Decode base64 image
    image_data = base64.b64decode(job_input['image'])
    image = Image.open(BytesIO(image_data)).convert('RGB')
    
    # Save input image
    temp_img = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    image.save(temp_img.name)
    
    # Output path
    temp_out = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    
    # Build command for WAN2.2 I2V
    prompt = job_input.get('prompt', 'high quality cinematic video, smooth camera movement')
    
    # WAN2.2 command für 1080p
    cmd = [
        'python', 'generate.py',
        '--task', 'i2v-A14B',
        '--size', '1920*1080',  # 1080p!
        '--ckpt_dir', '/models/Wan2.2-I2V-A14B',
        '--image', temp_img.name,
        '--prompt', prompt,
        '--seed', str(job_input.get('seed', 42)),
        '--save_dir', os.path.dirname(temp_out.name),
        '--filename', os.path.basename(temp_out.name).replace('.mp4', ''),
        '--offload_model', 'True'  # GPU Memory sparen
    ]
    
    # Falls duration angegeben
    duration = job_input.get('duration', 5)
    if duration:
        cmd.extend(['--num_frames', str(duration * 24)])  # 24fps
    
    print(f"Running: {' '.join(cmd)}")
    
    # Execute WAN2.2
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return {"error": result.stderr}
    
    # Read generated video
    video_path = temp_out.name.replace('.mp4', '_0000.mp4')  # WAN2.2 adds _0000
    
    if not os.path.exists(video_path):
        # Fallback paths
        possible_paths = [
            temp_out.name,
            f"/Wan2.2/outputs/{os.path.basename(temp_out.name)}",
            f"/Wan2.2/outputs/i2v/{os.path.basename(temp_out.name).replace('.mp4', '_0000.mp4')}"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                video_path = path
                break
        else:
            return {"error": f"Video not found. Looked in: {possible_paths}"}
    
    # Convert to base64
    with open(video_path, 'rb') as f:
        video_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    # Cleanup
    os.unlink(temp_img.name)
    if os.path.exists(video_path):
        os.unlink(video_path)
    
    return {
        "video": video_base64,
        "resolution": "1920x1080",
        "prompt": prompt,
        "duration": duration
    }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
