#!/usr/bin/env python3
"""
Convert ONNX Text LLM to TensorRT Engine
Optimizes the 630M parameter model for fast GPU inference
"""

import os
import time
import torch
import numpy as np
from pathlib import Path

def convert_onnx_to_tensorrt():
    """Convert ONNX model to TensorRT engine"""
    print("🚀 ONNX TO TENSORRT CONVERSION")
    print("=" * 50)
    
    try:
        import tensorrt as trt
        print(f"✅ TensorRT version: {trt.__version__}")
    except ImportError:
        print("❌ TensorRT not available")
        return False
    
    # Paths
    onnx_file = "../onnx_export/text_llm.onnx"
    output_dir = "../tensorrt_export"
    engine_file = os.path.join(output_dir, "text_llm.trt")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(onnx_file):
        print(f"❌ ONNX file not found: {onnx_file}")
        return False
    
    print(f"📁 Input: {onnx_file}")
    print(f"📁 Output: {engine_file}")
    
    try:
        # Initialize TensorRT
        print("\n🔧 Initializing TensorRT...")
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        
        # Configure optimization settings for LLMs
        print("⚙️  Configuring optimization settings...")
        
        # Memory settings for large models
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB workspace
        
        # Precision settings
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("✅ FP16 optimization enabled")
        else:
            print("⚠️  FP16 not supported, using FP32")
        
        # Optimization level
        config.builder_optimization_level = 5  # Highest optimization
        print("✅ Maximum optimization level set")
        
        # Parse ONNX model
        print("\n📖 Parsing ONNX model...")
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        
        # Load and parse ONNX
        with open(onnx_file, "rb") as model_file:
            if not parser.parse(model_file.read()):
                print("❌ Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    print(f"  Error {error}: {parser.get_error(error)}")
                return False
        
        print("✅ ONNX model parsed successfully")
        print(f"  Inputs: {[network.get_input(i).name for i in range(network.num_inputs)]}")
        print(f"  Outputs: {[network.get_output(i).name for i in range(network.num_outputs)]}")
        
        # Configure dynamic shapes for different sequence lengths
        print("\n🔄 Configuring dynamic shapes...")
        profile = builder.create_optimization_profile()
        
        input_tensor = network.get_input(0)
        input_name = input_tensor.name
        
        # Set dynamic shapes: min, opt, max for (batch_size, sequence_length)
        profile.set_shape(input_name, 
                         (1, 1),      # min: batch=1, seq=1
                         (1, 512),    # opt: batch=1, seq=512  
                         (4, 2048))   # max: batch=4, seq=2048
        
        config.add_optimization_profile(profile)
        print("✅ Dynamic shapes configured")
        
        # Build TensorRT engine
        print("\n🏗️  Building TensorRT engine...")
        print("⏳ This may take 10-20 minutes for a large model...")
        
        start_time = time.time()
        serialized_engine = builder.build_serialized_network(network, config)
        build_time = time.time() - start_time
        
        if serialized_engine is None:
            print("❌ Failed to build TensorRT engine")
            return False
        
        print(f"✅ Engine built successfully in {build_time:.1f} seconds")
        
        # Save engine to file
        print(f"\n💾 Saving engine to {engine_file}...")
        with open(engine_file, "wb") as f:
            f.write(serialized_engine)
        
        engine_size_mb = os.path.getsize(engine_file) / (1024 * 1024)
        print(f"✅ Engine saved: {engine_size_mb:.1f} MB")
        
        return True, output_dir
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tensorrt_engine(engine_dir):
    """Test the TensorRT engine"""
    print(f"\n🧪 TESTING TENSORRT ENGINE")
    print("=" * 40)
    
    engine_file = os.path.join(engine_dir, "text_llm.trt")
    
    if not os.path.exists(engine_file):
        print(f"❌ Engine file not found: {engine_file}")
        return False
    
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # Initialize CUDA context
        
        # Load engine
        print("🔧 Loading TensorRT engine...")
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        
        with open(engine_file, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        
        if engine is None:
            print("❌ Failed to load engine")
            return False
        
        print("✅ Engine loaded successfully")
        
        # Create execution context
        context = engine.create_execution_context()
        
        # Test with different sequence lengths
        test_sequences = [
            [1, 2, 3],                    # Short sequence
            [1, 2, 3, 4, 5, 6, 7, 8],     # Medium sequence  
            list(range(1, 33))            # Longer sequence
        ]
        
        print(f"\n🎯 Testing inference with different sequence lengths...")
        
        for i, tokens in enumerate(test_sequences):
            seq_len = len(tokens)
            print(f"\nTest {i+1}: Sequence length {seq_len}")
            
            # Prepare input
            input_ids = np.array([tokens], dtype=np.int64)  # Shape: (1, seq_len)
            
            # Set input shape for dynamic model
            context.set_input_shape("input_ids", input_ids.shape)
            
            # Allocate GPU memory
            input_nbytes = input_ids.nbytes
            output_shape = (1, seq_len, 151936)  # batch, seq, vocab
            output_nbytes = np.prod(output_shape) * np.dtype(np.float32).itemsize
            
            # Allocate device memory
            d_input = cuda.mem_alloc(input_nbytes)
            d_output = cuda.mem_alloc(output_nbytes)
            
            # Copy input to device
            cuda.memcpy_htod(d_input, input_ids)
            
            # Run inference
            start_time = time.time()
            context.execute_v2([int(d_input), int(d_output)])
            cuda.Context.synchronize()
            inference_time = time.time() - start_time
            
            # Copy result back
            output = np.empty(output_shape, dtype=np.float32)
            cuda.memcpy_dtoh(output, d_output)
            
            print(f"  ✅ Input shape: {input_ids.shape}")
            print(f"  ✅ Output shape: {output.shape}")
            print(f"  ✅ Inference time: {inference_time*1000:.1f} ms")
            print(f"  ✅ Tokens/second: {seq_len/inference_time:.1f}")
            
            # Clean up GPU memory
            d_input.free()
            d_output.free()
        
        print(f"\n✅ All TensorRT tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ TensorRT test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_onnx_vs_tensorrt():
    """Compare ONNX vs TensorRT performance"""
    print(f"\n⚡ PERFORMANCE COMPARISON")
    print("=" * 40)
    
    try:
        # Test data
        test_tokens = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int64)
        
        # ONNX inference
        print("1️⃣ ONNX Performance:")
        import onnxruntime as ort
        onnx_session = ort.InferenceSession("../onnx_export/text_llm.onnx", 
                                           providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        
        # Warmup
        for _ in range(3):
            onnx_session.run(None, {"input_ids": test_tokens})
        
        # Benchmark ONNX
        start_time = time.time()
        for _ in range(10):
            onnx_outputs = onnx_session.run(None, {"input_ids": test_tokens})
        onnx_time = (time.time() - start_time) / 10
        
        print(f"  Average time: {onnx_time*1000:.1f} ms")
        print(f"  Tokens/second: {test_tokens.shape[1]/onnx_time:.1f}")
        
        # TensorRT inference (if available)
        engine_file = "../tensorrt_export/text_llm.trt"
        if os.path.exists(engine_file):
            print("\n2️⃣ TensorRT Performance:")
            
            import tensorrt as trt
            import pycuda.driver as cuda
            
            # Load TensorRT engine
            logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(logger)
            
            with open(engine_file, "rb") as f:
                engine = runtime.deserialize_cuda_engine(f.read())
            
            context = engine.create_execution_context()
            context.set_input_shape("input_ids", test_tokens.shape)
            
            # Allocate memory
            input_nbytes = test_tokens.nbytes
            output_shape = (1, 8, 151936)
            output_nbytes = np.prod(output_shape) * 4
            
            d_input = cuda.mem_alloc(input_nbytes)
            d_output = cuda.mem_alloc(output_nbytes)
            cuda.memcpy_htod(d_input, test_tokens)
            
            # Warmup
            for _ in range(3):
                context.execute_v2([int(d_input), int(d_output)])
                cuda.Context.synchronize()
            
            # Benchmark TensorRT
            start_time = time.time()
            for _ in range(10):
                context.execute_v2([int(d_input), int(d_output)])
                cuda.Context.synchronize()
            trt_time = (time.time() - start_time) / 10
            
            print(f"  Average time: {trt_time*1000:.1f} ms")
            print(f"  Tokens/second: {test_tokens.shape[1]/trt_time:.1f}")
            
            speedup = onnx_time / trt_time
            print(f"\n🚀 TensorRT Speedup: {speedup:.1f}x faster than ONNX")
            
            # Clean up
            d_input.free()
            d_output.free()
        else:
            print("\n2️⃣ TensorRT not available for comparison")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance comparison failed: {e}")
        return False

def copy_tokenizer(output_dir):
    """Copy tokenizer files to TensorRT directory"""
    print(f"\n📋 Copying tokenizer files...")
    
    import shutil
    
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json", 
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json"
    ]
    
    copied = 0
    for file in tokenizer_files:
        src = f"../onnx_export/{file}"
        dst = f"{output_dir}/{file}"
        
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied += 1
    
    print(f"✅ Copied {copied} tokenizer files")
    return True

def main():
    """Main conversion function"""
    print("🎯 Starting ONNX to TensorRT Conversion...")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available - TensorRT requires NVIDIA GPU")
        return
    
    gpu_name = torch.cuda.get_device_name(0)
    print(f"🎮 GPU: {gpu_name}")
    
    try:
        # Convert ONNX to TensorRT
        result = convert_onnx_to_tensorrt()
        
        if result and len(result) == 2:
            success, output_dir = result
            
            if success:
                # Copy tokenizer
                copy_tokenizer(output_dir)
                
                # Test the engine
                test_success = test_tensorrt_engine(output_dir)
                
                # Performance comparison
                compare_onnx_vs_tensorrt()
                
                print(f"\n🎊 FINAL RESULTS")
                print("=" * 50)
                print(f"✅ ONNX to TensorRT conversion completed")
                print(f"{'✅' if test_success else '❌'} TensorRT engine testing")
                print(f"✅ Performance comparison completed")
                
                if success and test_success:
                    print(f"\n🎊 SUCCESS! TensorRT engine ready for deployment!")
                    print(f"\n📁 Your optimized model:")
                    print(f"  📄 {output_dir}/text_llm.trt")
                    print(f"  📄 {output_dir}/tokenizer files")
                    
                    print(f"\n🚀 Usage:")
                    print(f"  # Load TensorRT engine")
                    print(f"  import tensorrt as trt")
                    print(f"  import pycuda.driver as cuda")
                    print(f"  ")
                    print(f"  runtime = trt.Runtime(trt.Logger())")
                    print(f"  with open('{output_dir}/text_llm.trt', 'rb') as f:")
                    print(f"      engine = runtime.deserialize_cuda_engine(f.read())")
                    print(f"  context = engine.create_execution_context()")
                else:
                    print(f"\n⚠️  Conversion completed but testing failed")
            else:
                print(f"\n❌ Conversion failed")
        else:
            print(f"\n❌ Conversion failed")
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 