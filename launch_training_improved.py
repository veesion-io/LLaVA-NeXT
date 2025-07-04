#!/usr/bin/env python3

import subprocess
import sys
import time
from datetime import datetime

def request_training():
    """Request a new training job with improved video description generation"""
    
    # Generate training name with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    training_name = f"llava_cheater_improved-{timestamp}"
    
    print(f"🚀 Launching improved training: {training_name}")
    
    # Training configuration with improved video description generation
    cmd = [
        "python3", "-m", "veesion_io.request_training",
        "--training-name", training_name,
        "--simone-branch", "local_debug_hang",
        "--terraform-scalable-training-branch", "main",
        "--training-script", "scripts/video/train/SO400M_Qwen2_7B_ov_to_video_am9.sh",
        "--instance-type", "p5.48xlarge",  # H100 instance
        "--spot", "true",
        "--max-price", "10.0",  # $10/hour max
        "--region", "us-east-1",
        "--timeout", "7200",  # 2 hours timeout
        "--description", f"Improved video training with better description generation and 1.5x batch size - {training_name}"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Training request submitted successfully!")
        print(f"Training name: {training_name}")
        print(f"Output: {result.stdout}")
        return training_name
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to submit training request: {e}")
        print(f"Error output: {e.stderr}")
        return None

if __name__ == "__main__":
    training_name = request_training()
    if training_name:
        print(f"\n🎯 Training '{training_name}' launched successfully!")
        print("📊 Monitor with: python3 /opt/wait_and_monitor.py <training_name>")
        print("🛑 Kill with: python3 kill_all_trainings.py")
    else:
        sys.exit(1) 