#!/usr/bin/env python3

import sys
import os
from pathlib import Path
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from veesion_data_core.tools import request_training

def main():
    """Launch training using the package approach."""
    
    # Training configuration with unique name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_name = f"llava_video_training_{timestamp}"
    dataset_version = 1
    cpu_instances_count = 0
    gpu_instances_count = 1
    n_epochs = 3
    simone_branch = "new_development_branch"
    terraform_scalable_training_branch = "llava_training"
    
    print("Launching training with configuration:")
    print(f"  training_name: {training_name}")
    print(f"  dataset_version: {dataset_version}")
    print(f"  cpu_instances_count: {cpu_instances_count}")
    print(f"  gpu_instances_count: {gpu_instances_count}")
    print(f"  n_epochs: {n_epochs}")
    print(f"  simone_branch: {simone_branch}")
    print(f"  terraform_scalable_training_branch: {terraform_scalable_training_branch}")
    
    try:
        # Launch training using the package
        request_training(
            training_name=training_name,
            dataset_version=dataset_version,
            cpu_instances_count=cpu_instances_count,
            gpu_instances_count=gpu_instances_count,
            n_epochs=n_epochs,
            simone_branch=simone_branch,
            terraform_scalable_training_branch=terraform_scalable_training_branch
        )
        print(f"\n✅ Training launched successfully!")
        print(f"Training name: {training_name}")
        print(f"\nTo monitor the training, use:")
        print(f"python3 /opt/wait_and_monitor.py {training_name} --wait")
        
    except Exception as e:
        print(f"❌ Failed to launch training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 