#!/usr/bin/env python3
"""
Simple PyTorch Benchmark with Flash Attention, FP16, and KV Caching
Based on eval.py model - clean and focused
"""

import time
import torch
import numpy as np
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
torch.backends.cudnn.enabled = True

def create_test_video():
    """Create 35-frame 384x384 test video"""
    print("🎬 Creating 35-frame 384x384 test video...")
    frames = []
    for i in range(35):
        # Create animated test pattern
        frame = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
        center_x = int(192 + 100 * np.sin(i * 0.2))
        center_y = int(192 + 100 * np.cos(i * 0.2))
        
        # Add moving circle
        y, x = np.ogrid[:384, :384]
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= 50 ** 2
        frame[mask] = [255, 100, 100]
        
        frames.append(Image.fromarray(frame))
    
    print(f"✅ Created {len(frames)} frames at 384x384 resolution")
    return frames

def load_optimized_model():
    """Load LLaVA model with Flash Attention and FP16"""
    print("🔧 Loading optimized LLaVA model from local checkpoint...")
    
    # Load from local checkpoint directory
    pretrained = "./checkpoint-11000"  # Checkpoint directory contains the model files
    model_name = "llava_qwen"
    device_map = "auto"
    
    # Enable Flash Attention and multimodal
    llava_model_args = {
        "multimodal": True
    }
    
    print("📡 Loading checkpoint-11000 with Flash Attention...")
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        pretrained, None, model_name, 
        device_map=device_map,
        attn_implementation="flash_attention_2",  # Enable Flash Attention
        **llava_model_args
    )
    
    # Convert to FP16
    print("🎯 Converting to FP16...")
    model = model.half()
    model.eval()
    model = torch.compile(model, mode="max-autotune")
    
    print("✅ Checkpoint-11000 loaded with Flash Attention + FP16")
    return tokenizer, model, image_processor

def prepare_inputs(video_frames, tokenizer, image_processor, model):
    """Prepare model inputs"""
    print("📋 Preparing inputs...")
    
    # Process video frames
    image_tensor = process_images(video_frames, image_processor, model.config)
    image_tensor = image_tensor.half().cuda()  # FP16
    
    # Prepare conversation
    conv = conv_templates["qwen_1_5"].copy()
    question = "Describe this video."
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    # Tokenize
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    input_ids = input_ids.unsqueeze(0).cuda()
    
    print(f"✅ Input prepared - Image: {image_tensor.shape}, Text: {input_ids.shape}")
    return image_tensor, input_ids

def benchmark_model(tokenizer, model, image_tensor, input_ids, warmup_runs=3, benchmark_runs=5):
    """Benchmark model with Flash Attention, FP16, and KV caching"""
    print("\n🚀 OPTIMIZED PYTORCH BENCHMARK")
    print("=" * 50)
    print("✅ Flash Attention: ENABLED")
    print("✅ FP16 Precision: ENABLED") 
    print("✅ KV Caching: ENABLED")
    print("=" * 50)
    
    # Warmup runs
    print(f"🔥 Warming up ({warmup_runs} runs)...")
    for i in range(warmup_runs):
        with torch.inference_mode():
            with torch.cuda.amp.autocast(dtype=torch.float16):  # Ensure FP16
                _ = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[(384, 384)] * 35,  # Provide image sizes for video frames
                    do_sample=False,
                    temperature=0.,
                    max_new_tokens=4096,
                    use_cache=True,  # Enable KV caching
                    pad_token_id=tokenizer.eos_token_id
                )
        print(f"  Warmup {i+1}/{warmup_runs} complete")

    # Benchmark runs
    print(f"\n🏃 Benchmarking ({benchmark_runs} runs)...")
    times = []
    
    for i in range(benchmark_runs):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        with torch.inference_mode():
            with torch.cuda.amp.autocast(dtype=torch.float16):  # Ensure FP16
                output = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[(384, 384)] * 35,  # Provide image sizes for video frames
                    do_sample=False,
                    temperature=0.,
                    max_new_tokens=4096,
                    use_cache=True,  # Enable KV caching
                    pad_token_id=tokenizer.eos_token_id
                )
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        times.append(duration)
        print(f"  Run {i+1}: {duration:.4f}s")
    
    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    std_time = np.std(times)
    
    print(f"\n📊 BENCHMARK RESULTS:")
    print("=" * 50)
    print(f"Average time: {avg_time:.4f}s")
    print(f"Min time:     {min_time:.4f}s") 
    print(f"Max time:     {max_time:.4f}s")
    print(f"Std dev:      {std_time:.4f}s")
    print(f"Throughput:   {1/avg_time:.2f} inferences/sec")
    
    # Decode sample output
    if len(output[0]) > len(input_ids[0]):
        response = tokenizer.decode(output[0][len(input_ids[0]):], skip_special_tokens=True)
        print(f"\n💬 Sample output: {response[:100]}...")
    
    return {
        'avg_time': avg_time,
        'min_time': min_time, 
        'max_time': max_time,
        'std_time': std_time,
        'throughput': 1/avg_time,
        'config': {
            'flash_attention': True,
            'fp16': True,
            'kv_cache': True,
            'max_new_tokens': 4096,
            'video_frames': 35,
            'resolution': '384x384'
        }
    }

def profile_model_components(tokenizer, model, image_tensor, input_ids, benchmark_runs=3):
    """Profile individual model components to identify bottlenecks"""
    print(f"\n🔍 COMPONENT PROFILING")
    print("=" * 50)
    print("Analyzing where time is spent in the model...")
    print("=" * 50)
    
    component_times = {
        'vision_tower': [],
        'mm_projector': [], 
        'text_generation': [],
        'total_time': []
    }
    
    for run in range(benchmark_runs):
        print(f"\n📊 Profiling Run {run+1}/{benchmark_runs}")
        
        torch.cuda.synchronize()
        total_start = time.perf_counter()
        
        with torch.inference_mode():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                
                # 1. Time Vision Tower
                torch.cuda.synchronize()
                vision_start = time.perf_counter()
                
                vision_tower = model.get_vision_tower()
                # Reshape for vision tower: [35, 2, 3, 384, 384] -> [70, 3, 384, 384]
                batch_size, num_crops, channels, height, width = image_tensor.shape
                flattened_images = image_tensor.view(batch_size * num_crops, channels, height, width)
                vision_features = vision_tower(flattened_images)
                
                torch.cuda.synchronize()
                vision_end = time.perf_counter()
                vision_time = vision_end - vision_start
                
                # 2. Time MM Projector  
                torch.cuda.synchronize()
                proj_start = time.perf_counter()
                
                projected_features = model.get_model().mm_projector(vision_features)
                
                torch.cuda.synchronize()
                proj_end = time.perf_counter()
                proj_time = proj_end - proj_start
                
                # 3. Time Text Generation (full generate call)
                torch.cuda.synchronize()
                gen_start = time.perf_counter()
                
                output = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[(384, 384)] * 35,
                    do_sample=False,
                    temperature=0.,
                    max_new_tokens=50,  # Shorter for profiling
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                torch.cuda.synchronize()
                gen_end = time.perf_counter()
                text_gen_time = gen_end - gen_start
        
        torch.cuda.synchronize()
        total_end = time.perf_counter()
        total_time = total_end - total_start
        
        # Store times
        component_times['vision_tower'].append(vision_time)
        component_times['mm_projector'].append(proj_time)
        component_times['text_generation'].append(text_gen_time)
        component_times['total_time'].append(total_time)
        
        # Print this run's results
        print(f"   🔍 Vision Tower: {vision_time:.3f}s")
        print(f"   🔗 MM Projector: {proj_time:.3f}s") 
        print(f"   📝 Text Generation: {text_gen_time:.3f}s")
        print(f"   📊 Total Time: {total_time:.3f}s")
    
    # Calculate averages and percentages
    print(f"\n📈 COMPONENT ANALYSIS (Average over {benchmark_runs} runs)")
    print("="*70)
    
    total_avg = np.mean(component_times['total_time'])
    results = {}
    
    for component in ['vision_tower', 'mm_projector', 'text_generation']:
        times = component_times[component]
        avg_time = np.mean(times)
        std_time = np.std(times)
        percentage = (avg_time / total_avg) * 100
        
        results[component] = {
            'avg_time': avg_time,
            'std_time': std_time,
            'percentage': percentage
        }
        
        print(f"🎯 {component.replace('_', ' ').title():<15}: {avg_time:.3f}s ± {std_time:.3f}s ({percentage:.1f}%)")
    
    # Identify bottleneck
    bottleneck = max(results.items(), key=lambda x: x[1]['avg_time'])
    print(f"\n🚨 BIGGEST BOTTLENECK: {bottleneck[0].replace('_', ' ').title()}")
    print(f"   Takes {bottleneck[1]['avg_time']:.3f}s ({bottleneck[1]['percentage']:.1f}% of total time)")
    
    # Recommendations
    print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
    if bottleneck[0] == 'vision_tower':
        print("   - Consider TensorRT optimization for vision tower")
        print("   - Try 8-bit quantization (if not already used)")
        print("   - Reduce image resolution or number of crops")
    elif bottleneck[0] == 'text_generation':
        print("   - Reduce max_new_tokens if possible")
        print("   - Consider speculative decoding")
        print("   - Optimize attention with better KV caching")
    
    return results

def main():
    """Main benchmark function"""
    print("🎊 SIMPLE PYTORCH BENCHMARK")
    print("🔧 Flash Attention + FP16 + KV Caching")
    print("=" * 60)
    
    try:
        # Load optimized model
        tokenizer, model, image_processor = load_optimized_model()
        
        # Create test video
        video_frames = create_test_video()
        
        # Prepare inputs
        image_tensor, input_ids = prepare_inputs(video_frames, tokenizer, image_processor, model)
        
        # Run component profiling
        profile_results = profile_model_components(tokenizer, model, image_tensor, input_ids)
        
        # Run full benchmark
        benchmark_results = benchmark_model(tokenizer, model, image_tensor, input_ids)
        
        # Combine results
        combined_results = {
            'profiling': profile_results,
            'benchmark': benchmark_results
        }
        
        # Save results
        import json
        with open('pytorch_benchmark_results.json', 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        print(f"\n💾 Results saved to 'pytorch_benchmark_results.json'")
        print(f"🎯 Optimized PyTorch: {benchmark_results['avg_time']:.4f}s average")
        print("✅ Benchmark complete!")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 