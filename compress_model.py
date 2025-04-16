import torch
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle

def quantize_model(input_path, output_path):
    """
    Compress the model through quantization and save the result
    """
    print(f"Loading model from {input_path}")
    print(f"Original file size: {os.path.getsize(input_path) / (1024 * 1024):.2f} MB")
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the model state dictionary
    state_dict = torch.load(input_path, map_location=device)
    
    # Apply direct weight quantization 
    # Convert all floating point tensors to half precision (float16)
    for key in state_dict:
        if isinstance(state_dict[key], torch.Tensor) and state_dict[key].dtype == torch.float32:
            state_dict[key] = state_dict[key].half()  # Convert to float16
    
    # Save the quantized model
    torch.save(state_dict, output_path)
    
    # Print file size reduction
    if os.path.exists(output_path):
        print(f"Compressed file size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
        print(f"Compression ratio: {os.path.getsize(input_path) / os.path.getsize(output_path):.2f}x")
        
        return os.path.getsize(output_path) / (1024 * 1024) < 25  # Return True if under 25MB
    return False

def extreme_quantization(input_path, output_path):
    """
    Extremely aggressive quantization using int8 and custom weight formats
    """
    print(f"Applying extreme quantization to {input_path}")
    print(f"Original file size: {os.path.getsize(input_path) / (1024 * 1024):.2f} MB")
    
    # Load the model state dictionary
    state_dict = torch.load(input_path, map_location='cpu')
    
    # New state dict with compressed weights
    compressed_state_dict = {}
    
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor) and len(tensor.shape) > 1:
            # Compress large matrices/tensors
            tensor_np = tensor.detach().cpu().numpy()
            
            # For large weight matrices, use a more aggressive approach
            if tensor.numel() > 10000:
                # Calculate scale factors for quantization to int8
                min_val = tensor_np.min()
                max_val = tensor_np.max()
                scale = (max_val - min_val) / 255
                
                # Quantize to int8
                quantized = np.round((tensor_np - min_val) / scale).astype(np.uint8)
                
                # Store as tuple of (quantized values, scale, min_val) to reconstruct later
                compressed_state_dict[key] = {
                    'quantized': quantized,
                    'scale': scale,
                    'min_val': min_val,
                    'shape': tensor.shape
                }
            else:
                # For smaller tensors, use normal half precision
                compressed_state_dict[key] = tensor.half()
        else:
            # Keep small tensors as is
            compressed_state_dict[key] = tensor
    
    # Save with pickle format (more efficient than torch save for custom objects)
    with open(output_path, 'wb') as f:
        pickle.dump(compressed_state_dict, f, protocol=4)
    
    # Print file size reduction
    if os.path.exists(output_path):
        print(f"Extremely quantized file size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
        print(f"Compression ratio: {os.path.getsize(input_path) / os.path.getsize(output_path):.2f}x")
        
        return os.path.getsize(output_path) / (1024 * 1024) < 25  # Return True if under 25MB
    return False

def weight_clustering(input_path, output_path, n_clusters=16):
    """
    Compress weights by clustering similar values
    """
    print(f"Applying weight clustering to {input_path}")
    print(f"Original file size: {os.path.getsize(input_path) / (1024 * 1024):.2f} MB")
    
    # Load the model state dictionary
    state_dict = torch.load(input_path, map_location='cpu')
    clustered_state_dict = {}
    
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor) and len(tensor.shape) > 1 and tensor.numel() > 1000:
            # Only cluster larger tensors
            tensor_np = tensor.detach().cpu().numpy()
            original_shape = tensor_np.shape
            flattened = tensor_np.reshape(-1)
            
            # Sort values to find natural clusters
            sorted_weights = np.sort(np.abs(flattened))
            n = len(sorted_weights)
            
            # Use percentile-based clustering
            centroids = []
            for i in range(n_clusters):
                idx = int(i * n / n_clusters)
                centroids.append(sorted_weights[idx])
            
            # Add zero as a centroid
            centroids.append(0)
            centroids = np.array(centroids)
            
            # Assign each weight to nearest centroid
            clustered = np.zeros_like(flattened)
            indices = np.zeros(flattened.shape, dtype=np.uint8)
            
            for i in range(len(flattened)):
                # Find closest centroid
                idx = np.argmin(np.abs(np.abs(flattened[i]) - centroids))
                indices[i] = idx
                # Preserve sign
                clustered[i] = centroids[idx] * np.sign(flattened[i])
            
            # Store just indices and centroids to save space
            clustered_state_dict[key] = {
                'indices': indices.reshape(original_shape),
                'centroids': centroids,
                'shape': original_shape
            }
        else:
            # Keep small tensors as is
            clustered_state_dict[key] = tensor
    
    # Save with pickle format
    with open(output_path, 'wb') as f:
        pickle.dump(clustered_state_dict, f, protocol=4)
    
    # Print file size reduction
    if os.path.exists(output_path):
        print(f"Clustered weights file size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
        print(f"Compression ratio: {os.path.getsize(input_path) / os.path.getsize(output_path):.2f}x")
        
        return os.path.getsize(output_path) / (1024 * 1024) < 25  # Return True if under 25MB
    return False

def matrix_decomposition(input_path, output_path):
    """
    Compress large matrices using SVD decomposition
    """
    print(f"Applying matrix decomposition to {input_path}")
    print(f"Original file size: {os.path.getsize(input_path) / (1024 * 1024):.2f} MB")
    
    # Load the model state dictionary
    state_dict = torch.load(input_path, map_location='cpu')
    decomposed_state_dict = {}
    
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor) and len(tensor.shape) == 2 and min(tensor.shape) > 50:
            # Only decompose larger matrices
            tensor_np = tensor.detach().cpu().numpy()
            
            # Compute SVD decomposition
            try:
                u, s, vh = np.linalg.svd(tensor_np, full_matrices=False)
                
                # Keep only enough components to retain 90% of variance
                total_variance = np.sum(s**2)
                variance_threshold = 0.9 * total_variance
                
                cumulative_variance = np.cumsum(s**2)
                k = np.argmax(cumulative_variance >= variance_threshold) + 1
                k = max(k, 10)  # Ensure we keep at least 10 components
                
                # Truncate to k components
                u_k = u[:, :k]
                s_k = s[:k]
                vh_k = vh[:k, :]
                
                # Store decomposed components
                decomposed_state_dict[key] = {
                    'u': u_k,
                    's': s_k,
                    'vh': vh_k,
                    'shape': tensor.shape
                }
            except Exception as e:
                # If SVD fails, fallback to half precision
                decomposed_state_dict[key] = tensor.half()
        else:
            # Keep small tensors as is
            decomposed_state_dict[key] = tensor
    
    # Save with pickle format
    with open(output_path, 'wb') as f:
        pickle.dump(decomposed_state_dict, f, protocol=4)
    
    # Print file size reduction
    if os.path.exists(output_path):
        print(f"Decomposed file size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
        print(f"Compression ratio: {os.path.getsize(input_path) / os.path.getsize(output_path):.2f}x")
        
        return os.path.getsize(output_path) / (1024 * 1024) < 25  # Return True if under 25MB
    return False

def sparse_coding(input_path, output_path, sparsity=0.8):
    """
    Create a sparse representation of the model
    """
    print(f"Applying sparse coding to {input_path}")
    print(f"Original file size: {os.path.getsize(input_path) / (1024 * 1024):.2f} MB")
    
    # Load the model state dictionary
    state_dict = torch.load(input_path, map_location='cpu')
    sparse_state_dict = {}
    
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor) and len(tensor.shape) > 1:
            # Convert to numpy for processing
            tensor_np = tensor.detach().cpu().numpy()
            
            # Create a mask of the weights to keep (largest magnitude)
            flat = np.abs(tensor_np.reshape(-1))
            threshold = np.quantile(flat, sparsity)
            
            # Create sparse representation
            mask = np.abs(tensor_np) > threshold
            values = tensor_np[mask]
            
            # Store only non-zero values and their indices
            sparse_state_dict[key] = {
                'mask': mask.astype(np.uint8),  # 8x smaller than bool
                'values': values.astype(np.float16),  # half precision
                'shape': tensor.shape
            }
        else:
            # For small tensors, just use half precision
            if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32:
                sparse_state_dict[key] = tensor.half()
            else:
                sparse_state_dict[key] = tensor
    
    # Save with pickle format
    with open(output_path, 'wb') as f:
        pickle.dump(sparse_state_dict, f, protocol=4)
    
    # Print file size reduction
    if os.path.exists(output_path):
        print(f"Sparse model file size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
        print(f"Compression ratio: {os.path.getsize(input_path) / os.path.getsize(output_path):.2f}x")
        
        return os.path.getsize(output_path) / (1024 * 1024) < 25  # Return True if under 25MB
    return False

def create_loader_script(output_path, compressed_path, method):
    """Create a script that can load the compressed model"""
    loader_code = """import torch
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
    compressed_path = "COMPRESSED_MODEL_PATH"  # Will be replaced
    output_path = "peppenwriter.pth"
    
    if not os.path.exists(compressed_path):
        print(f"Error: Compressed model not found at {compressed_path}")
        exit(1)
    
    load_compressed_model(compressed_path, output_path)
    print("Model successfully decompressed and ready to use")
"""
    
    # Replace the placeholder with the actual compressed model filename
    loader_code = loader_code.replace("COMPRESSED_MODEL_PATH", os.path.basename(compressed_path))
    
    # Write the loader script
    with open(output_path, 'w') as f:
        f.write(loader_code)
    
    print(f"Created loader script at {output_path}")

def try_all_methods():
    """Try multiple compression methods and pick the best one"""
    input_path = "peppenwriter.pth"
    
    # Ensure the input file exists
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found")
        return False
    
    # Create backup of original file
    backup_path = "peppenwriter_original.pth"
    print(f"Backing up original model to {backup_path}")
    if not os.path.exists(backup_path):
        # Copy the file (don't use os.rename to keep the original intact during testing)
        with open(input_path, 'rb') as src, open(backup_path, 'wb') as dst:
            dst.write(src.read())
    
    # Try extreme quantization (int8)
    extreme_path = "peppenwriter_extreme.pkl"
    extreme_success = extreme_quantization(input_path, extreme_path)
    
    # Try weight clustering
    clustered_path = "peppenwriter_clustered.pkl"
    clustered_success = weight_clustering(input_path, clustered_path)
    
    # Try matrix decomposition
    decomposed_path = "peppenwriter_decomposed.pkl"
    decomposed_success = matrix_decomposition(input_path, decomposed_path)
    
    # Try sparse coding
    sparse_path = "peppenwriter_sparse.pkl"
    sparse_success = sparse_coding(input_path, sparse_path)
    
    # Find the smallest file under 25MB
    candidates = []
    for path, method in [
        (extreme_path, "extreme_quantization"),
        (clustered_path, "weight_clustering"),
        (decomposed_path, "matrix_decomposition"),
        (sparse_path, "sparse_coding")
    ]:
        if os.path.exists(path) and os.path.getsize(path) / (1024 * 1024) < 25:
            candidates.append((path, os.path.getsize(path), method))
    
    if candidates:
        # Sort by file size (smallest first)
        candidates.sort(key=lambda x: x[1])
        best_path, size, method = candidates[0]
        
        print(f"\nBest compression method: {method}")
        print(f"File size: {size / (1024 * 1024):.2f} MB")
        
        # Create a loader script that can reconstruct the model
        loader_script_path = "decompress_model.py"
        create_loader_script(loader_script_path, best_path, method)
        
        print(f"\nCompressed model saved as {best_path}")
        print(f"To use the compressed model, run 'python decompress_model.py'")
        
        # Update the model loading code in model.py to handle the compressed format
        # (Future enhancement)
        
        return True
    else:
        # If all else fails, try the most aggressive sparse coding
        print("\nTrying ultra-aggressive sparse coding (95% sparsity)")
        ultra_sparse_path = "peppenwriter_ultra_sparse.pkl"
        ultra_sparse_success = sparse_coding(input_path, ultra_sparse_path, sparsity=0.95)
        
        if ultra_sparse_success:
            print(f"\nUltra-aggressive sparse coding successful")
            print(f"File size: {os.path.getsize(ultra_sparse_path) / (1024 * 1024):.2f} MB")
            
            # Create a loader script
            loader_script_path = "decompress_model.py"
            create_loader_script(loader_script_path, ultra_sparse_path, "sparse_coding")
            
            print(f"\nCompressed model saved as {ultra_sparse_path}")
            print(f"To use the compressed model, run 'python decompress_model.py'")
            
            return True
        
        print("\nFailed to compress model to under 25 MB with any method")
        return False

if __name__ == "__main__":
    try_all_methods()
