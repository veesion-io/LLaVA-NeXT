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
                try:
                    memory_usage = []
                    total_allocated = 0
                    total_reserved = 0
                    
                    for i in range(torch.cuda.device_count()):
                        try:
                            # Force sync and context switch to get accurate memory stats
                            torch.cuda.synchronize(i)
                            with torch.cuda.device(i):
                                allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                                reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
                                total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
                                memory_usage.append(f"GPU{i}: {allocated:.1f}GB/{reserved:.1f}GB/{total:.1f}GB")
                                total_allocated += allocated
                                total_reserved += reserved
                        except Exception as e:
                            memory_usage.append(f"GPU{i}: Error - {e}")
                            total_allocated += 0
                            total_reserved += 0
                    
                    rank0_print(f"Step {self.step_count}: Memory - {' | '.join(memory_usage)}")
                    
                    # Add memory metrics to logs for TensorBoard
                    if logs is not None:
                        logs.update({
                            'memory/allocated_gb': total_allocated,
                            'memory/reserved_gb': total_reserved,
                            'memory/utilization_percent': (total_allocated / (total_reserved + 1e-6)) * 100
                        })
                        
                except Exception as e:
                    rank0_print(f"Step {self.step_count}: Memory logging failed - {e}")


class VideoDescriptionCallback(TrainerCallback):
    """Callback to log video descriptions every N batches."""
    
    def __init__(self, log_every_n_steps=10):
        self.log_every_n_steps = log_every_n_steps
        self.step_count = 0
        self.last_logged_step = 0
        
    def on_step_end(self, args, state, control, model=None, inputs=None, **kwargs):
        self.step_count += 1
        
        # Only log every N steps
        if self.step_count % self.log_every_n_steps == 0:
            self.last_logged_step = self.step_count
            
            # Try to generate video descriptions from the current batch
            if inputs is not None and 'images' in inputs and model is not None:
                try:
                    # Generate descriptions for first 2 videos in batch
                    for i in range(min(2, len(inputs['images']))):
                        if inputs['images'][i] is not None:
                            # Create a simple prompt for video description
                            from llava.constants import DEFAULT_IMAGE_TOKEN
                            prompt = f"{DEFAULT_IMAGE_TOKEN}Describe this video in detail."
                            
                            # Tokenize the prompt
                            from llava.mm_utils import tokenizer_image_token
                            from llava.constants import IMAGE_TOKEN_INDEX
                            input_ids = tokenizer_image_token(prompt, model.config.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
                            
                            # Generate description
                            with torch.no_grad():
                                outputs = model.generate(
                                    input_ids,
                                    images=[inputs['images'][i]],
                                    image_sizes=inputs.get('image_sizes', [None]),
                                    modalities=inputs.get('modalities', ['video']),
                                    do_sample=False,
                                    temperature=0.0,
                                    max_new_tokens=100,
                                    pad_token_id=model.config.tokenizer.pad_token_id,
                                    eos_token_id=model.config.tokenizer.eos_token_id,
                                )
                            
                            # Decode the output
                            description = model.config.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                            # Remove the input prompt from the output
                            description = description.replace(prompt, "").strip()
                            
                            if description:
                                rank0_print(f"Step {self.step_count} - Video {i+1} Description: {description}")
                            else:
                                rank0_print(f"Step {self.step_count} - Video {i+1} Description: [No description generated]")
                        else:
                            rank0_print(f"Step {self.step_count} - Video {i+1} Description: [No video data]")
                except Exception as e:
                    rank0_print(f"Step {self.step_count} - Error generating video descriptions: {e}")
            else:
                rank0_print(f"Step {self.step_count} - No video data available in current batch")


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
            
            rank0_print("Applied performance optimizations for H100 GPUs")
            self.optimization_applied = True 