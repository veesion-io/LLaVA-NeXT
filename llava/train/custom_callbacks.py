import os
import torch
from transformers import TrainerCallback
from llava.utils import rank0_print
import json
from typing import Dict, Any


class SelectiveLoggingCallback(TrainerCallback):
    """Callback to log loss and video descriptions every N batches instead of every batch."""
    
    def __init__(self, log_every_n_steps=10, log_video_descriptions=True):
        self.log_every_n_steps = log_every_n_steps
        self.log_video_descriptions = log_video_descriptions
        self.step_count = 0
        self.last_logged_step = 0
        
    def on_step_end(self, args, state, control, model=None, **kwargs):
        self.step_count += 1
        
        # Only log every N steps
        if self.step_count % self.log_every_n_steps == 0:
            self.last_logged_step = self.step_count
            
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Only log if this is one of our selected steps
        if self.step_count != self.last_logged_step:
            return
            
        if logs is not None:
            # Log loss and other metrics
            rank0_print(f"Step {self.step_count}: Loss = {logs.get('loss', 'N/A'):.4f}")
            
            # Log learning rate
            if 'learning_rate' in logs:
                rank0_print(f"Step {self.step_count}: LR = {logs['learning_rate']:.2e}")
                
            # Log GPU memory usage if available
            if torch.cuda.is_available():
                memory_usage = []
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                    reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
                    memory_usage.append(f"GPU{i}: {allocated:.1f}GB/{reserved:.1f}GB")
                rank0_print(f"Step {self.step_count}: Memory - {' | '.join(memory_usage)}")


class VideoDescriptionCallback(TrainerCallback):
    """Callback to log video descriptions every N batches."""
    
    def __init__(self, log_every_n_steps=10):
        self.log_every_n_steps = log_every_n_steps
        self.step_count = 0
        self.last_logged_step = 0
        
    def on_step_end(self, args, state, control, model=None, **kwargs):
        self.step_count += 1
        
        # Only log every N steps
        if self.step_count % self.log_every_n_steps == 0:
            self.last_logged_step = self.step_count
            
    def on_prediction_step(self, args, state, control, model=None, inputs=None, **kwargs):
        # Only log if this is one of our selected steps
        if self.step_count != self.last_logged_step:
            return
            
        if inputs is not None and 'prompts' in inputs:
            # Log video descriptions from prompts
            for i, prompt in enumerate(inputs['prompts'][:2]):  # Log first 2 videos
                if isinstance(prompt, str) and len(prompt) > 100:
                    # Truncate long prompts for readability
                    truncated_prompt = prompt[:200] + "..." if len(prompt) > 200 else prompt
                    rank0_print(f"Step {self.step_count} - Video {i+1} Description: {truncated_prompt}")
                elif isinstance(prompt, str):
                    rank0_print(f"Step {self.step_count} - Video {i+1} Description: {prompt}")


class PerformanceOptimizationCallback(TrainerCallback):
    """Callback to optimize performance and memory usage."""
    
    def __init__(self):
        self.optimization_applied = False
        
    def on_train_begin(self, args, state, control, **kwargs):
        if not self.optimization_applied:
            # Set environment variables for better performance
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
            os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
            os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
            
            # Optimize PyTorch settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            rank0_print("Applied performance optimizations for H200 GPUs")
            self.optimization_applied = True 