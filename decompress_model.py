import torch
import pickle
import numpy as np
import os
from model import RhymingTransformer

def load_compressed_model(compressed_path, output_path):
    # Load compressed data
    with open(compressed_path, 'rb') as f:
        compressed_data = pickle.load(f)
    
    # Create state dict for the model
    state_dict = {}
    
    for key, value in compressed_data.items():
        # Handle different compression methods
        if isinstance(value, dict) and 'quantized' in value:
            # Extreme quantization reconstruction
            quantized = value['quantized']
            scale = value['scale']
            min_val = value['min_val']
            shape = value['shape']
            
            # Dequantize
            dequantized = (quantized.astype(np.float32) * scale) + min_val
            state_dict[key] = torch.tensor(dequantized).reshape(shape)
            
        elif isinstance(value, dict) and 'indices' in value:
            # Weight clustering reconstruction
            indices = value['indices']
            centroids = value['centroids']
            shape = value['shape']
            
            # Reconstruct tensor
            reconstructed = np.zeros(indices.size, dtype=np.float32)
            for i, centroid in enumerate(centroids):
                reconstructed[indices.flatten() == i] = centroid
            
            state_dict[key] = torch.tensor(reconstructed.reshape(shape))
            
        elif isinstance(value, dict) and 'u' in value:
            # Matrix decomposition reconstruction
            u = value['u']
            s = value['s']
            vh = value['vh']
            shape = value['shape']
            
            # Reconstruct matrix
            reconstructed = np.dot(u * s, vh)
            state_dict[key] = torch.tensor(reconstructed.reshape(shape))
            
        elif isinstance(value, dict) and 'mask' in value:
            # Sparse coding reconstruction
            mask = value['mask'].astype(bool)
            values = value['values']
            shape = value['shape']
            
            # Reconstruct tensor
            reconstructed = np.zeros(shape, dtype=np.float32)
            reconstructed[mask] = values
            
            state_dict[key] = torch.tensor(reconstructed)
            
        else:
            # Normal tensor
            state_dict[key] = value
    
    # Save the reconstructed state dict
    torch.save(state_dict, output_path)
    print(f"Model decompressed and saved to {output_path}")
    
    return state_dict

if __name__ == "__main__":
    # Auto-detect paths
    compressed_path = "peppenwriter_extreme.pkl"  # Will be replaced
    output_path = "peppenwriter.pth"
    
    if not os.path.exists(compressed_path):
        print(f"Error: Compressed model not found at {compressed_path}")
        exit(1)
    
    load_compressed_model(compressed_path, output_path)
    print("Model successfully decompressed and ready to use")
