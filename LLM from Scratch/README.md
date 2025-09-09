# GPT-2 From Scratch in PyTorch

A clean, minimal re-implementation of **GPT-2** built entirely from scratch in PyTorch.  
This project demonstrates the core building blocks of transformer architectures, including multi-head causal attention, layer normalization, and autoregressive text generation.

---

## Features
- **Custom Transformer Implementation**  
  - Causal multi-head self-attention  
  - Pre-LayerNorm residual connections  
  - Position-wise feedforward (MLP with GELU)  
- **GPT-2 Architecture**  
  - Token & positional embeddings  
  - Configurable number of layers, heads, and context length  
  - Final projection to vocabulary logits  
- **Autoregressive Text Generation**  
  - Greedy or temperature-scaled sampling  
  - Top-k filtering for diversity  
  - Early stopping with EOS token  
- **Weight Loading from TensorFlow Checkpoints**  
  - Convert and load pretrained GPT-2 weights into this PyTorch model
- **Fine-Tuning the custom GPT for AGNews Classification**
  - The pre-trained model is adapted with a classification head for prediction


