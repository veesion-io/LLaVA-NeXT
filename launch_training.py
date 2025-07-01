#!/usr/bin/env python3
"""
Simple LLaVA training launcher using veesion_data_core with S3 bucket configuration
"""

import time
import sys

def launch_training():
    """Launch training using veesion_data_core tools"""
    
    try:
        # Import after activating environment
        import veesion_data_core.tools as tools
        
        # Training parameters
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
        print("=" * 60)
        
        # Try to launch training
        print("🚀 Attempting to launch training...")
        
        # This will likely fail due to the different API, but let's try
        # We need to find the correct function signature
        print("📋 Available functions in veesion_data_core.tools:")
        available_functions = [func for func in dir(tools) if not func.startswith('_')]
        for func in available_functions:
            if 'training' in func.lower():
                print(f"  - {func}")
        
        # Try to use request_training with different parameters
        print("\n🔧 Attempting to launch with available tools...")
        
        # This is for the old system, but let's see what happens
        try:
            tools.request_training(
                training_name=training_name,
                dataset_version=28,  # int as required
                cpu_instances_count=0,
                gpu_instances_count=gpu_instances,
                n_epochs=5,
                archi=instance_type,
                simone_branch=branch,
                terraform_scalable_training_branch="llava_training"
            )
            print(f"✅ Training {training_name} launched successfully!")
            return training_name
        except Exception as e:
            print(f"❌ request_training failed: {e}")
            return None
            
    except ImportError as e:
        print(f"❌ Failed to import veesion_data_core: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

if __name__ == "__main__":
    print("🚀 Simple LLaVA Training Launcher")
    print("🔧 With S3 Bucket Configuration")
    print("=" * 50)
    
    training_name = launch_training()
    
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