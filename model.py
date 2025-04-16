import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import sys
from typing import Optional
import pickle
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = math.sqrt(self.d_head)
    
    def forward(self, q, k, v, mask: Optional[torch.Tensor] = None):
        batch_size = q.shape[0]
        
        # Project and reshape
        q = self.q_proj(q).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(k).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(v).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class RhymeEnhancedTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Special rhyme attention head
        self.rhyme_attn = MultiHeadAttention(d_model, 1, dropout)
        self.rhyme_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask: Optional[torch.Tensor] = None):
        # Regular self-attention
        attn_out, _ = self.attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Rhyme-focused attention (attending to last tokens more strongly)
        rhyme_out, _ = self.rhyme_attn(x[:, -1:, :], x, x, mask)
        x = self.rhyme_norm(x + self.dropout(rhyme_out))
        
        # Feed forward
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x

class RhymingTransformer(nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 d_model: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 d_ff: int = 2048,
                 max_seq_len: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            RhymeEnhancedTransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.out = nn.Linear(d_model, vocab_size)
        
        # Special rhyme embedding
        self.rhyme_embedding = nn.Embedding(vocab_size, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, rhyme_tokens=None):
        # Get embeddings
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Create attention mask (prevent attending to future tokens)
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.to(x.device)
        
        # Add rhyme information if provided
        if rhyme_tokens is not None:
            # Get rhyme embeddings and reshape to match input
            rhyme_emb = self.rhyme_embedding(rhyme_tokens)  # [batch, seq, dim]
            # Expand rhyme embedding to match input sequence length
            rhyme_emb = rhyme_emb[:, 0:1, :].expand(-1, x.size(1), -1)
            x = x + rhyme_emb
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Project to vocabulary
        output = self.out(x)
        
        return output

class RhymingTextGenerator:
    def __init__(self, model_path='peppenwriter_extreme.pkl'):
        """Initialize the RhymingTextGenerator."""
        try:
            # Check CUDA availability
            if not torch.cuda.is_available():
                print("Warning: CUDA not available, using CPU")
                self.device = torch.device("cpu")
            else:
                self.device = torch.device("cuda")
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
                print(f"CUDA Device Count: {torch.cuda.device_count()}")
                print(f"Current CUDA Device: {torch.cuda.current_device()}")
        except Exception as e:
            print(f"Error initializing CUDA: {str(e)}. Using CPU.")
            self.device = torch.device("cpu")
        
        # Initialize vocabulary (we'll build this from data)
        self.vocab_size = 16567  # Match the trained model size
        self.max_seq_len = 128   
        
        try:
            # Create model
            self.model = RhymingTransformer(
                vocab_size=self.vocab_size,
                d_model=256,  # Reduced from 512
                num_heads=4,  # Reduced from 8
                num_layers=4,  # Reduced from 6
                d_ff=1024,    # Reduced from 2048
                max_seq_len=self.max_seq_len,
                dropout=0.2   # Increased dropout for better stability
            ).to(self.device)
            
            # Initialize vocabulary
            self.vocab = {
                '<pad>': 0,
                '<sos>': 1,
                '<eos>': 2,
                '<unk>': 3,
            }
            print("[DEBUG] Initial vocabulary:", self.vocab)
            
            # Handle model path for compiled version
            try:
                state_dict = None
                # First try the path as provided
                if os.path.exists(model_path):
                    print(f"Loading model from {model_path}")
                    if model_path.endswith('.pkl'):
                        # Load compressed/quantized model
                        with open(model_path, 'rb') as f:
                            compressed_data = pickle.load(f)
                        state_dict = self._decompress_state_dict(compressed_data)
                    else:
                        # Load standard PyTorch checkpoint
                        state_dict = torch.load(model_path, map_location=self.device)
                else:
                    # Try to find model in the executable's directory (for packaged version)
                    exe_dir = os.path.dirname(sys.executable)
                    model_path_alt = os.path.join(exe_dir, 'peppenwriter_extreme.pkl')
                    if os.path.exists(model_path_alt):
                        print(f"Loading model from executable directory: {model_path_alt}")
                        if model_path_alt.endswith('.pkl'):
                            with open(model_path_alt, 'rb') as f:
                                compressed_data = pickle.load(f)
                            state_dict = self._decompress_state_dict(compressed_data)
                        else:
                            state_dict = torch.load(model_path_alt, map_location=self.device)
                    else:
                        # Try current directory
                        current_dir = os.path.dirname(os.path.abspath(__file__))
                        model_path_alt = os.path.join(current_dir, 'peppenwriter_extreme.pkl')
                        if os.path.exists(model_path_alt):
                            print(f"Loading model from current directory: {model_path_alt}")
                            if model_path_alt.endswith('.pkl'):
                                with open(model_path_alt, 'rb') as f:
                                    compressed_data = pickle.load(f)
                                state_dict = self._decompress_state_dict(compressed_data)
                            else:
                                state_dict = torch.load(model_path_alt, map_location=self.device)
                        else:
                            print(f"ERROR: Could not find model file at any of these locations:")
                            print(f"  - {model_path}")
                            print(f"  - {os.path.join(exe_dir, 'peppenwriter_extreme.pkl')}")
                            print(f"  - {os.path.join(current_dir, 'peppenwriter_extreme.pkl')}")
                            self.model_loaded = False
                            return
                # If we get here, model loaded successfully
                self.model.load_state_dict(state_dict)
                print("[DEBUG] Model state_dict loaded.")
                # Check for NaNs in parameters
                nan_found = False
                for name, param in self.model.named_parameters():
                    if torch.isnan(param).any():
                        print(f"[DEBUG] Parameter {name} contains NaNs!")
                        nan_found = True
                if not nan_found:
                    print("[DEBUG] No NaNs found in model parameters.")
                print("[DEBUG] Model parameters loaded successfully.")
                self.model_loaded = True
                print("Model loaded successfully!")
                
                # DEBUG: Check vocabulary size
                if len(self.vocab) <= 4:
                    print("[WARNING] Vocabulary has only special tokens! Need to build vocabulary.")
                    # Build vocabulary from a word list or corpus
                    vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lines.txt")
                    if os.path.exists(vocab_path):
                        print(f"Loading vocabulary from {vocab_path}")
                        with open(vocab_path, "r", encoding="utf-8") as f:
                            # Read all lines and split into words
                            content = f.read()
                            words = set()
                            for line in content.splitlines():
                                for word in line.split():
                                    if word.strip() and word not in self.vocab:
                                        words.add(word)
                            # Add words to vocabulary
                            for i, word in enumerate(sorted(words)):
                                self.vocab[word] = i + 4
                        print(f"Loaded vocabulary with {len(self.vocab)} tokens")
                    else:
                        print(f"[WARNING] No lines.txt file found at {vocab_path}")
                
                # Build reverse vocabulary for detokenization
                self.reverse_vocab = {v: k for k, v in self.vocab.items()}
                print(f"Vocabulary size: {len(self.vocab)} tokens")
            except Exception as e:
                print(f"ERROR loading model: {str(e)}")
                import traceback
                traceback.print_exc()
                self.model_loaded = False
                return
            # Initialize vocabulary
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}
            # Load training data to build vocabulary
            if os.path.exists('lines.txt'):
                with open('lines.txt', 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                self.build_vocab(lines)
            self.model.train()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"ERROR initializing model: {str(e)}")
            import traceback
            traceback.print_exc()
            self.model_loaded = False

    def _decompress_state_dict(self, compressed_data):
        state_dict = {}
        for key, value in compressed_data.items():
            # Handle different compression methods
            if isinstance(value, dict) and 'quantized' in value:
                quantized = np.array(value['quantized'], dtype=np.float32)
                scale = value['scale']
                min_val = value['min_val']
                shape = value['shape']
                dequantized = (quantized * scale) + min_val
                state_dict[key] = torch.tensor(dequantized.reshape(shape), dtype=torch.float32)
            elif isinstance(value, dict) and 'indices' in value:
                indices = np.array(value['indices'])
                centroids = np.array(value['centroids'], dtype=np.float32)
                shape = value['shape']
                reconstructed = np.zeros(indices.size, dtype=np.float32)
                for i, centroid in enumerate(centroids):
                    reconstructed[(indices.flatten() == i)] = centroid
                state_dict[key] = torch.tensor(reconstructed.reshape(shape), dtype=torch.float32)
            elif isinstance(value, dict) and 'u' in value:
                u = np.array(value['u'], dtype=np.float32)
                s = np.array(value['s'], dtype=np.float32)
                vh = np.array(value['vh'], dtype=np.float32)
                shape = value['shape']
                us = u * s[..., None] if s.ndim == 1 else u * s
                reconstructed = np.dot(us, vh)
                state_dict[key] = torch.tensor(reconstructed.reshape(shape), dtype=torch.float32)
            elif isinstance(value, dict) and 'mask' in value:
                mask = np.array(value['mask']).astype(bool)
                values = np.array(value['values'], dtype=np.float32)
                shape = value['shape']
                reconstructed = np.zeros(shape, dtype=np.float32)
                reconstructed[mask] = values
                state_dict[key] = torch.tensor(reconstructed, dtype=torch.float32)
            else:
                state_dict[key] = value
        return state_dict

    def pad_sequence(self, tokens, max_len=None):
        """Pad or truncate token sequence to specified length"""
        if max_len is None:
            max_len = self.max_seq_len
        
        if len(tokens) > max_len:
            return tokens[:max_len]
        else:
            padding = torch.full((max_len - len(tokens),), self.vocab['<pad>'], dtype=tokens.dtype)
            return torch.cat([tokens, padding])
    
    def tokenize(self, text):
        """Convert text to token IDs and pad sequence"""
        tokens = []
        words = text.strip().split()
        for word in words:
            token = self.vocab.get(word, self.vocab['<unk>'])
            tokens.append(token)
        return tokens

    def detokenize(self, tokens):
        """Convert token IDs back to text, removing padding"""
        words = []
        for token in tokens:
            # Skip special tokens
            if token < 4:  # Skip <pad>, <sos>, <eos>, <unk>
                continue
                
            # Get word from reverse vocab
            word = self.reverse_vocab.get(token, '<unk>')
            if word not in ['<pad>', '<sos>', '<eos>', '<unk>']:
                words.append(word)
                
        return ' '.join(words)
    
    def build_vocab(self, texts):
        """Build vocabulary from training data"""
        word_freq = {}
        for text in texts:
            words = text.split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and take top vocab_size words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        sorted_words = sorted_words[:self.vocab_size-4]  
        
        # Reset vocabulary
        self.vocab = {
            '<pad>': 0,
            '<sos>': 1,
            '<eos>': 2,
            '<unk>': 3,
        }
        
        # Add words to vocab
        for i, (word, _) in enumerate(sorted_words):
            self.vocab[word] = i + 4
        
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Update model embeddings for new vocabulary size
        actual_vocab_size = len(self.vocab)
        self.model.token_embedding = torch.nn.Embedding(actual_vocab_size, self.model.d_model).to(self.device)
        self.model.rhyme_embedding = torch.nn.Embedding(actual_vocab_size, self.model.d_model).to(self.device)
        self.model.out = torch.nn.Linear(self.model.d_model, actual_vocab_size).to(self.device)
        
        print(f"Built vocabulary with {len(self.vocab)} tokens")

    def get_last_word(self, text):
        """Get the last word of a text"""
        words = text.split()
        return words[-1] if words else ''
    
    def generate_response(self, input_text, target_length=None, temperature=1):
        """Generate a response to the input text with specified target length."""
        # Check if model failed to load
        if not hasattr(self, 'model_loaded') or not self.model_loaded:
            fallback_responses = [
                "Es scheint, als könnte ich dir nicht antworten. Die KI funktioniert nicht richtig.",
                "Die Welt der Wörter liegt im Dunkeln. Ich kann nicht antworten.",
                "Die Inspiration ist versiegt. Ich kann keine Antwort generieren.",
                "Meine Kreativität ist eingefroren. Keine Antwort möglich.",
                "Die KI-Komponente funktioniert nicht. Bitte überprüfe die Installation.",
                "[Die AI konnte nicht geladen werden. Bitte stelle sicher, dass PyTorch korrekt installiert ist.]"
            ]
            import random
            return random.choice(fallback_responses)
        
        try:
            self.model.eval()
            with torch.no_grad():
                # Process input
                input_tokens = self.tokenize(input_text)
                print("Tokenized input:", input_tokens)
                if not input_tokens:  # Handle empty input
                    print("[DEBUG] Input tokenization produced an empty list.")
                    return ""
                
                # Convert to tensor
                input_tensor = torch.tensor(input_tokens, dtype=torch.long, device=self.device).unsqueeze(0)
                
                # Initialize output with start token
                output_tokens = [self.vocab['<sos>']]
                current_length = 0
                
                # Generate until we reach target length or max sequence length
                max_len = min(self.max_seq_len, target_length * 2 if target_length else 50)
                
                while len(output_tokens) < max_len:
                    curr_tensor = torch.tensor(output_tokens, dtype=torch.long, device=self.device).unsqueeze(0)
                    
                    # Get model predictions
                    with torch.cuda.amp.autocast():
                        outputs = self.model(curr_tensor)
                        next_token_logits = outputs[0, -1, :]
                        print("Next token logits:", next_token_logits.cpu().numpy())
                        
                        # Apply temperature and softmax
                        next_token_logits = next_token_logits / max(temperature, 1e-6)
                        next_token_logits = torch.nan_to_num(next_token_logits, -float('inf'))  # Replace NaNs
                        
                        # Apply softmax with proper numerical stability
                        next_token_logits = next_token_logits - torch.max(next_token_logits)
                        probs = F.softmax(next_token_logits, dim=-1)
                        print("Probabilities:", probs.cpu().numpy())
                        
                        # Ensure valid probabilities
                        probs = torch.nan_to_num(probs, 0.0)  # Replace any remaining NaNs with 0
                        if torch.sum(probs) == 0:  # If all probs are 0, use uniform distribution
                            probs = torch.ones_like(probs) / probs.size(0)
                    
                    # Sample from the distribution
                    try:
                        next_token = torch.multinomial(probs, 1).item()
                    except RuntimeError:
                        print(f"Error sampling: min={probs.min()}, max={probs.max()}, sum={probs.sum()}")
                        next_token = self.vocab['<eos>']
                    
                    # Stop if we hit the end token or reached target length
                    if next_token == self.vocab['<eos>'] or \
                       (target_length and current_length >= target_length):
                        break
                    
                    output_tokens.append(next_token)
                    if next_token >= 4:  # Skip special tokens in length count
                        current_length += 1
                
                # Convert tokens back to text
                print("Output tokens:", output_tokens)
                response = self.detokenize(output_tokens)
                print("Detokenized response:", response)
                
            self.model.train()
            return response.strip()

        except Exception as e:
            print(f"Error generating response: {str(e)}")
            import traceback
            traceback.print_exc()
            return "[Die Antwort konnte nicht generiert werden. Bitte überprüfe die Installation.]"

    def save_model(self, path):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

# Beispielverwendung
if __name__ == "__main__":
    generator = RhymingTextGenerator()
    print("Testing text generation...")
    response = generator.generate_response("This is a test line to see if it works")
    print(f"Input: This is a test line to see if it works")
    print(f"Response: {response}")
