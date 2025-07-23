#!/usr/bin/env python3
"""
Direct Weight Extraction from LLaVA Model
Extract text LLM weights directly from safetensors and create a clean model
"""

import torch
import torch.nn as nn
import json
import os
from safetensors import safe_open
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

def load_safetensors_weights(checkpoint_path):
    """Load weights from safetensors file"""
    print("🔧 Loading weights from safetensors...")
    
    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    if not os.path.exists(safetensors_path):
        print(f"❌ Safetensors file not found: {safetensors_path}")
        return None
    
    weights = {}
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)
    
    print(f"✅ Loaded {len(weights)} weight tensors")
    return weights

def filter_text_llm_weights(weights):
    """Filter out vision and multimodal projector weights"""
    print("🎯 Filtering text LLM weights...")
    
    text_weights = {}
    vision_keys = []
    projector_keys = []
    other_keys = []
    
    for key, tensor in weights.items():
        if 'vision_tower' in key:
            vision_keys.append(key)
        elif 'mm_projector' in key:
            projector_keys.append(key)
        else:
            text_weights[key] = tensor
            other_keys.append(key)
    
    print(f"📊 Weight distribution:")
    print(f"  Text LLM weights: {len(text_weights)}")
    print(f"  Vision tower weights: {len(vision_keys)}")
    print(f"  MM projector weights: {len(projector_keys)}")
    
    print(f"\n📝 Sample text LLM weight keys:")
    for i, key in enumerate(list(text_weights.keys())[:10]):
        print(f"  {key}: {text_weights[key].shape}")
    
    return text_weights

def create_qwen2_config(checkpoint_path):
    """Create a clean Qwen2 config from the checkpoint config"""
    print("🔧 Creating Qwen2 config...")
    
    config_path = os.path.join(checkpoint_path, "config.json")
    with open(config_path, 'r') as f:
        llava_config = json.load(f)
    
    # Extract the relevant parameters for Qwen2
    qwen2_config = {
        "vocab_size": llava_config.get("vocab_size", 151936),  # Qwen2 vocab size
        "hidden_size": llava_config["hidden_size"],
        "intermediate_size": llava_config["intermediate_size"],
        "num_hidden_layers": llava_config["num_hidden_layers"],
        "num_attention_heads": llava_config["num_attention_heads"],
        "num_key_value_heads": llava_config["num_key_value_heads"],
        "hidden_act": llava_config["hidden_act"],
        "max_position_embeddings": llava_config["max_position_embeddings"],
        "initializer_range": llava_config["initializer_range"],
        "rms_norm_eps": llava_config["rms_norm_eps"],
        "rope_theta": llava_config["rope_theta"],
        "attention_dropout": llava_config.get("attention_dropout", 0.0),
        "sliding_window": llava_config.get("sliding_window", None),
        "model_type": "qwen2",
        "torch_dtype": "float32",
        "use_cache": True,
    }
    
    print(f"✅ Created Qwen2 config:")
    print(f"  Hidden size: {qwen2_config['hidden_size']}")
    print(f"  Layers: {qwen2_config['num_hidden_layers']}")
    print(f"  Vocab size: {qwen2_config['vocab_size']}")
    
    return Qwen2Config(**qwen2_config)

def create_clean_qwen2_model(text_weights, config):
    """Create a clean Qwen2 model with the text weights"""
    print("🔨 Creating clean Qwen2 model...")
    
    # Create the model
    model = Qwen2ForCausalLM(config)
    
    # Load the text weights
    print("📋 Loading text weights into Qwen2 model...")
    
    # Create a mapping for weight keys
    model_state_dict = {}
    for key, tensor in text_weights.items():
        # Remove 'model.' prefix if it exists, since Qwen2ForCausalLM expects it
        clean_key = key
        if key.startswith('model.'):
            clean_key = key[6:]  # Remove 'model.' prefix
        
        model_state_dict[clean_key] = tensor
    
    # Load the weights
    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
    
    print(f"📊 Weight loading results:")
    print(f"  Missing keys: {len(missing_keys)}")
    print(f"  Unexpected keys: {len(unexpected_keys)}")
    
    if missing_keys:
        print("⚠️  Missing keys (first 5):")
        for key in missing_keys[:5]:
            print(f"    {key}")
    
    if unexpected_keys:
        print("⚠️  Unexpected keys (first 5):")
        for key in unexpected_keys[:5]:
            print(f"    {key}")
    
    model.eval()
    print("✅ Clean Qwen2 model created successfully!")
    
    return model

def export_to_onnx(model, tokenizer, output_path="qwen2_text_llm.onnx"):
    """Export the clean Qwen2 model to ONNX"""
    print(f"\n🚀 Exporting to ONNX: {output_path}")
    
    # Prepare dummy inputs
    batch_size = 1
    seq_length = 8
    
    dummy_input_ids = torch.randint(1, 1000, (batch_size, seq_length), dtype=torch.long)
    dummy_attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
    
    print(f"Dummy input shape: {dummy_input_ids.shape}")
    
    # Test the model first
    print("🧪 Testing model...")
    try:
        with torch.no_grad():
            outputs = model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask)
            print(f"Test output shape: {outputs.logits.shape}")
            print("✅ Model test passed!")
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False
    
    # Export to ONNX
    try:
        print("📦 Exporting to ONNX...")
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask),
            output_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"}
            },
            opset_version=11,
            do_constant_folding=True,
            export_params=True,
            verbose=False
        )
        
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024*1024)
            print(f"✅ Successfully exported to ONNX: {output_path} ({file_size:.1f} MB)")
            return True
        else:
            print(f"❌ ONNX file was not created")
            return False
            
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        return False

def main():
    """Main function"""
    print("🎊 DIRECT LLaVA TEXT LLM EXTRACTION")
    print("=" * 50)
    
    checkpoint_path = "../checkpoint-11000"
    
    try:
        # Load tokenizer
        print("🔧 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        print(f"✅ Tokenizer loaded: vocab_size = {tokenizer.vocab_size}")
        
        # Load weights from safetensors
        weights = load_safetensors_weights(checkpoint_path)
        if weights is None:
            return
        
        # Filter text LLM weights
        text_weights = filter_text_llm_weights(weights)
        
        # Create clean Qwen2 config
        config = create_qwen2_config(checkpoint_path)
        
        # Create clean model
        model = create_clean_qwen2_model(text_weights, config)
        
        # Save clean model
        print("\n💾 Saving clean model...")
        model.save_pretrained("../qwen2_text_llm", safe_serialization=True)
        tokenizer.save_pretrained("../qwen2_text_llm")
        print("✅ Clean model saved to ../qwen2_text_llm/")
        
        # Export to ONNX
        onnx_success = export_to_onnx(model, tokenizer)
        
        print(f"\n🎯 Summary:")
        print(f"✅ Text LLM weights extracted ({len(text_weights)} tensors)")
        print(f"✅ Clean Qwen2 model created")
        print(f"✅ Model saved to ../qwen2_text_llm/")
        print(f"{'✅' if onnx_success else '❌'} ONNX export")
        
        if onnx_success:
            print("\n📁 Output files:")
            print("  - qwen2_text_llm.onnx (ONNX model)")
            print("  - ../qwen2_text_llm/ (Clean PyTorch model)")
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 