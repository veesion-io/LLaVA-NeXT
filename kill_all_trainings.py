#!/usr/bin/env python3

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from veesion_data_core.tools import stop_training, destroy_training

def main():
    """Kill all running LLaVA trainings using the package."""
    
    # List of training names to stop (we'll need to get these from the running workflows)
    # For now, let's try common patterns
    training_names_to_stop = [
        "llava_video_training_new_branch",
        "llava_s3_bucket_1751304879",
        # Add more training names as needed
    ]
    
    print("Stopping all running LLaVA trainings...")
    
    for training_name in training_names_to_stop:
        try:
            print(f"Stopping training: {training_name}")
            stop_training(training_name)
            print(f"✅ Successfully stopped: {training_name}")
        except Exception as e:
            print(f"❌ Failed to stop {training_name}: {e}")
    
    print("\nAll trainings have been stopped.")

if __name__ == "__main__":
    main() 