#!/usr/bin/env python3
"""
Simple LLaVA training launcher using veesion_data_core tools
"""

import time
import sys
import argparse
import veesion_data_core.tools as tools

def launch_training(spot=False):
    """Launch training using veesion_data_core tools"""
    
    training_name = f"llava_s3_bucket_{int(time.time())}"
    script_path = "scripts/video/train/SO400M_Qwen2_7B_ov_to_video_am9.sh"
    branch = "new_development_branch"
    gpu_instances = 1
    instance_type = "h100.24xlarge"
    region = "us-east-1"
    
    print(f"🎯 Launching training: {training_name}")
    print(f"📝 Script: {script_path}")
    print(f"🌿 Branch: {branch}")
    print(f"💻 Instance: {gpu_instances}x {instance_type}")
    print(f"🌍 Region: {region}")
    print(f"💰 Spot instance: {spot}")
    print("=" * 60)
    
    kwargs = {
        "training_name": training_name,
        "dataset_version": 28,
        "cpu_instances_count": 0,
        "gpu_instances_count": gpu_instances,
        "n_epochs": 5,
        "archi": instance_type,
        "simone_branch": branch,
        "terraform_scalable_training_branch": "llava_training",
    }
    
    if spot:
        # Check if spot is supported before adding it
        import inspect
        sig = inspect.signature(tools.request_training)
        if 'spot' in sig.parameters:
            kwargs['spot'] = True
        else:
            print("⚠️  'spot' argument not supported by 'request_training'. Launching on-demand instance.")

    try:
        tools.request_training(**kwargs)
        print(f"✅ Training {training_name} launched successfully!")
        return training_name
    except Exception as e:
        print(f"❌ request_training failed: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple LLaVA Training Launcher")
    parser.add_argument("--spot", action="store_true", help="Use spot instances for training")
    args = parser.parse_args()

    print("🚀 Simple LLaVA Training Launcher")
    print("🔧 With S3 Bucket Configuration")
    print("=" * 50)
    
    training_name = launch_training(spot=args.spot)
    
    if training_name:
        print(f"\n✅ Training launched: {training_name}")
        print(f"\n🔍 To monitor the training, run:")
        print(f"python3 /opt/wait_and_monitor.py {training_name}")
    else:
        print("\n❌ Training launch failed")
        print("\n💡 Manual launch options:")
        print("1. Use GitHub Actions workflow in terraform-scalable-training repo")
        print("2. Use AWS console to launch instances manually")
        print("3. Check veesion_data_core documentation for correct API") 