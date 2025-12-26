#!/usr/bin/env python3
"""
Script to generate visualizations for Lesson 7: Transformers, Attention & Pre-trained Models

This script generates the following visualizations and saves them to slides/assets/:
1. Self-attention heatmap
2. Causal (masked) attention heatmap for GPT
3. Sinusoidal positional encoding visualization
4. Multi-head attention patterns

Run from repo root:
    python scripts/generate_transformer_visualizations.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "slides", "assets")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def scaled_dot_product_attention(query, key, value, mask=None):
    """Compute scaled dot-product attention."""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    return output, attention_weights


def create_causal_mask(seq_len):
    """Create a lower-triangular mask for causal attention."""
    return torch.tril(torch.ones(seq_len, seq_len))


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from 'Attention Is All You Need'."""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module."""
    
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(attn_output)
        return output, attn_weights


def generate_self_attention_heatmap():
    """Generate self-attention weights heatmap."""
    print("Generating self-attention heatmap...")
    
    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    seq_len = len(tokens)
    d_model = 8
    
    x = torch.randn(1, seq_len, d_model)
    Q, K, V = x, x, x
    _, attn_weights = scaled_dot_product_attention(Q, K, V)
    
    weights = attn_weights.squeeze(0).detach().numpy()
    
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(weights, cmap='Blues')
    
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, fontsize=11)
    ax.set_yticklabels(tokens, fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    
    ax.set_xlabel("Keys (attending to)", fontsize=12)
    ax.set_ylabel("Queries (from)", fontsize=12)
    ax.set_title("Self-Attention Weights", fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label="Attention Weight")
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "self_attention_heatmap.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {output_path}")


def generate_causal_attention_heatmap():
    """Generate causal (GPT-style) attention heatmap."""
    print("Generating causal attention heatmap...")
    
    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    seq_len = len(tokens)
    d_model = 8
    
    x = torch.randn(1, seq_len, d_model)
    Q, K, V = x, x, x
    causal_mask = create_causal_mask(seq_len)
    _, attn_weights = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
    
    weights = attn_weights.squeeze(0).detach().numpy()
    
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(weights, cmap='Oranges')
    
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, fontsize=11)
    ax.set_yticklabels(tokens, fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    
    ax.set_xlabel("Keys (attending to)", fontsize=12)
    ax.set_ylabel("Queries (from)", fontsize=12)
    ax.set_title("Causal (Masked) Attention — GPT Style", fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label="Attention Weight")
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "causal_attention_heatmap.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {output_path}")


def generate_positional_encoding_visualization():
    """Generate sinusoidal positional encoding visualization."""
    print("Generating positional encoding visualization...")
    
    d_model = 64
    max_len = 100
    
    pos_encoder = SinusoidalPositionalEncoding(d_model, max_len)
    pe = pos_encoder.pe.squeeze(0).numpy()
    
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(pe[:50, :], aspect='auto', cmap='RdBu', vmin=-1, vmax=1)
    
    ax.set_xlabel('Embedding Dimension', fontsize=12)
    ax.set_ylabel('Position in Sequence', fontsize=12)
    ax.set_title('Sinusoidal Positional Encoding', fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label="Encoding Value")
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "positional_encoding.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {output_path}")


def generate_multihead_attention_visualization():
    """Generate multi-head attention patterns visualization."""
    print("Generating multi-head attention visualization...")
    
    tokens = ["The", "quick", "brown", "fox", "jumps", "over"]
    d_model = 64
    n_heads = 4
    seq_len = len(tokens)
    batch_size = 1
    
    mha = MultiHeadAttention(d_model, n_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    _, attn_weights = mha(x)
    
    fig, axes = plt.subplots(1, n_heads, figsize=(16, 4))
    
    for head_idx in range(n_heads):
        weights = attn_weights[0, head_idx].detach().numpy()
        ax = axes[head_idx]
        im = ax.imshow(weights, cmap='Purples')
        ax.set_title(f"Head {head_idx + 1}", fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(tokens, fontsize=9)
    
    fig.suptitle("Multi-Head Attention Patterns", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "multihead_attention.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {output_path}")


def generate_attention_formula_diagram():
    """Generate a diagram explaining the attention formula."""
    print("Generating attention formula diagram...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    # Create a visual representation of Q, K, V
    ax.text(0.5, 0.95, "Scaled Dot-Product Attention", fontsize=18, fontweight='bold', 
            ha='center', va='top', transform=ax.transAxes)
    
    ax.text(0.5, 0.85, r"$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$", 
            fontsize=16, ha='center', va='top', transform=ax.transAxes)
    
    # Explanation boxes
    explanations = [
        ("Q (Query)", "What am I looking for?", 0.15, 0.55, '#e3f2fd'),
        ("K (Keys)", "What does each position offer?", 0.5, 0.55, '#e8f5e9'),
        ("V (Values)", "The content to retrieve", 0.85, 0.55, '#fff3e0'),
    ]
    
    for label, desc, x, y, color in explanations:
        box = plt.Rectangle((x-0.12, y-0.15), 0.24, 0.28, 
                            facecolor=color, edgecolor='gray', linewidth=2,
                            transform=ax.transAxes, zorder=1)
        ax.add_patch(box)
        ax.text(x, y+0.05, label, fontsize=12, fontweight='bold', 
                ha='center', va='center', transform=ax.transAxes, zorder=2)
        ax.text(x, y-0.05, desc, fontsize=10, ha='center', va='center', 
                transform=ax.transAxes, zorder=2, style='italic')
    
    ax.text(0.5, 0.25, r"$\sqrt{d_k}$ scaling prevents dot products from becoming too large", 
            fontsize=12, ha='center', va='center', transform=ax.transAxes)
    
    ax.text(0.5, 0.1, "Softmax gives attention weights that sum to 1", 
            fontsize=12, ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "attention_formula_diagram.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved to {output_path}")


def generate_bert_vs_gpt_comparison():
    """Generate BERT vs GPT comparison diagram."""
    print("Generating BERT vs GPT comparison...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # BERT (Bidirectional)
    ax1 = axes[0]
    ax1.set_title("BERT — Bidirectional", fontsize=14, fontweight='bold')
    tokens_bert = ["[CLS]", "The", "[MASK]", "sat", "on", "mat", "[SEP]"]
    n = len(tokens_bert)
    
    # Draw bidirectional connections
    for i in range(n):
        for j in range(n):
            if i != j:
                ax1.annotate("", xy=(j, 0.3), xytext=(i, 0.7),
                           arrowprops=dict(arrowstyle="->", color='blue', alpha=0.3, lw=1))
    
    # Draw tokens
    for i, tok in enumerate(tokens_bert):
        color = '#ffeb3b' if tok == "[MASK]" else '#e3f2fd'
        ax1.add_patch(plt.Rectangle((i-0.3, 0.4), 0.6, 0.2, facecolor=color, edgecolor='black'))
        ax1.text(i, 0.5, tok, ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax1.set_xlim(-0.5, n-0.5)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.text(n/2-0.5, 0.15, "Each token sees ALL other tokens", fontsize=11, ha='center', style='italic')
    ax1.text(n/2-0.5, 0.0, "Best for: Understanding tasks (QA, NER, Classification)", fontsize=10, ha='center', color='green')
    
    # GPT (Autoregressive)
    ax2 = axes[1]
    ax2.set_title("GPT — Autoregressive (Causal)", fontsize=14, fontweight='bold')
    tokens_gpt = ["The", "cat", "sat", "on", "the", "???"]
    n = len(tokens_gpt)
    
    # Draw left-to-right connections only
    for i in range(n):
        for j in range(i):
            ax2.annotate("", xy=(j, 0.3), xytext=(i, 0.7),
                       arrowprops=dict(arrowstyle="->", color='orange', alpha=0.5, lw=1.5))
    
    # Draw tokens
    for i, tok in enumerate(tokens_gpt):
        color = '#ffeb3b' if tok == "???" else '#fff3e0'
        ax2.add_patch(plt.Rectangle((i-0.3, 0.4), 0.6, 0.2, facecolor=color, edgecolor='black'))
        ax2.text(i, 0.5, tok, ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax2.set_xlim(-0.5, n-0.5)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.text(n/2-0.5, 0.15, "Each token sees only PREVIOUS tokens", fontsize=11, ha='center', style='italic')
    ax2.text(n/2-0.5, 0.0, "Best for: Generation tasks (Text completion, Chat)", fontsize=10, ha='center', color='green')
    
    plt.suptitle("BERT vs GPT: Context Understanding", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "bert_vs_gpt_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved to {output_path}")


def generate_transformer_block_diagram():
    """Generate a diagram showing the Transformer encoder/decoder blocks."""
    print("Generating transformer block diagram...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    
    # Encoder block
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 12)
    ax1.axis('off')
    ax1.set_title("Encoder Block", fontsize=14, fontweight='bold', pad=20)
    
    # Components (bottom to top)
    encoder_components = [
        (1, "Input Embedding\n+ Positional Encoding", '#e3f2fd'),
        (3, "Multi-Head\nSelf-Attention", '#bbdefb'),
        (5, "Add & Norm", '#90caf9'),
        (7, "Feed-Forward\nNetwork", '#64b5f6'),
        (9, "Add & Norm", '#42a5f5'),
        (11, "To Next Layer\nor Output", '#e3f2fd'),
    ]
    
    for y, label, color in encoder_components:
        box = plt.Rectangle((2, y-0.8), 6, 1.4, facecolor=color, 
                            edgecolor='black', linewidth=2)
        ax1.add_patch(box)
        ax1.text(5, y, label, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows
    for i in range(len(encoder_components)-1):
        ax1.annotate('', xy=(5, encoder_components[i+1][0]-0.8), 
                    xytext=(5, encoder_components[i][0]+0.6),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Residual connections
    ax1.annotate('', xy=(1.8, 5), xytext=(1.8, 3.6),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5, 
                               connectionstyle='arc3,rad=-0.3'))
    ax1.text(0.8, 4.3, 'Residual', fontsize=8, color='green', rotation=90, va='center')
    
    ax1.annotate('', xy=(1.8, 9), xytext=(1.8, 7.6),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5,
                               connectionstyle='arc3,rad=-0.3'))
    
    # Decoder block
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 14)
    ax2.axis('off')
    ax2.set_title("Decoder Block", fontsize=14, fontweight='bold', pad=20)
    
    decoder_components = [
        (1, "Target Embedding\n+ Positional Encoding", '#fff3e0'),
        (3, "Masked Multi-Head\nSelf-Attention", '#ffe0b2'),
        (5, "Add & Norm", '#ffcc80'),
        (7, "Multi-Head\nCross-Attention", '#ffb74d'),
        (9, "Add & Norm", '#ffa726'),
        (11, "Feed-Forward\nNetwork", '#ff9800'),
        (13, "Add & Norm → Output", '#fff3e0'),
    ]
    
    for y, label, color in decoder_components:
        box = plt.Rectangle((2, y-0.8), 6, 1.4, facecolor=color, 
                            edgecolor='black', linewidth=2)
        ax2.add_patch(box)
        ax2.text(5, y, label, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrows
    for i in range(len(decoder_components)-1):
        ax2.annotate('', xy=(5, decoder_components[i+1][0]-0.8), 
                    xytext=(5, decoder_components[i][0]+0.6),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Encoder output arrow
    ax2.annotate('From\nEncoder', xy=(8.2, 7), xytext=(9.5, 7),
                fontsize=9, ha='center', va='center',
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    
    plt.suptitle("Transformer Block Architecture", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "transformer_blocks.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved to {output_path}")


def generate_attention_flow_diagram():
    """Generate a diagram showing how attention scores are computed."""
    print("Generating attention flow diagram...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Input
    ax.add_patch(plt.Rectangle((0.5, 2), 1.5, 2, facecolor='#e8f5e9', edgecolor='black', lw=2))
    ax.text(1.25, 3, 'Input\nX', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Q, K, V projections
    for i, (label, color) in enumerate([('Q', '#e3f2fd'), ('K', '#fff3e0'), ('V', '#fce4ec')]):
        y = 4.5 - i * 1.5
        ax.add_patch(plt.Rectangle((3, y-0.4), 1.2, 0.8, facecolor=color, edgecolor='black', lw=2))
        ax.text(3.6, y, f'W{label}', ha='center', va='center', fontsize=10, fontweight='bold')
        ax.annotate('', xy=(3, y), xytext=(2, 3), arrowprops=dict(arrowstyle='->', lw=1.5))
        
        ax.add_patch(plt.Rectangle((5, y-0.4), 1, 0.8, facecolor=color, edgecolor='black', lw=2))
        ax.text(5.5, y, label, ha='center', va='center', fontsize=11, fontweight='bold')
        ax.annotate('', xy=(5, y), xytext=(4.2, y), arrowprops=dict(arrowstyle='->', lw=1.5))
    
    # Dot product Q·K^T
    ax.add_patch(plt.Rectangle((7, 3.5), 1.5, 1.2, facecolor='#e1bee7', edgecolor='black', lw=2))
    ax.text(7.75, 4.1, r'$QK^T$', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.annotate('', xy=(7, 4.1), xytext=(6, 4.5), arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.annotate('', xy=(7, 4.1), xytext=(6, 3), arrowprops=dict(arrowstyle='->', lw=1.5))
    
    # Scale
    ax.add_patch(plt.Rectangle((7, 1.8), 1.5, 1, facecolor='#d1c4e9', edgecolor='black', lw=2))
    ax.text(7.75, 2.3, r'$\div\sqrt{d_k}$', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.annotate('', xy=(7.75, 2.8), xytext=(7.75, 3.5), arrowprops=dict(arrowstyle='->', lw=1.5))
    
    # Softmax
    ax.add_patch(plt.Rectangle((9, 2.6), 1.5, 1, facecolor='#b39ddb', edgecolor='black', lw=2))
    ax.text(9.75, 3.1, 'Softmax', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.annotate('', xy=(9, 3.1), xytext=(8.5, 2.3), arrowprops=dict(arrowstyle='->', lw=1.5))
    
    # Attention weights × V
    ax.add_patch(plt.Rectangle((11, 2.1), 1.5, 1.8, facecolor='#9575cd', edgecolor='black', lw=2))
    ax.text(11.75, 3, r'$\times V$', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax.annotate('', xy=(11, 3), xytext=(10.5, 3.1), arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.annotate('', xy=(11, 2.5), xytext=(6, 1.5), arrowprops=dict(arrowstyle='->', lw=1.5, 
                connectionstyle='arc3,rad=0.3'))
    
    # Output
    ax.add_patch(plt.Rectangle((13, 2.5), 0.8, 1, facecolor='#7e57c2', edgecolor='black', lw=2))
    ax.text(13.4, 3, 'Out', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    ax.annotate('', xy=(13, 3), xytext=(12.5, 3), arrowprops=dict(arrowstyle='->', lw=1.5))
    
    ax.set_title("Scaled Dot-Product Attention: Data Flow", fontsize=14, fontweight='bold', y=1.05)
    
    output_path = os.path.join(OUTPUT_DIR, "attention_flow.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved to {output_path}")


def main():
    print("=" * 60)
    print("Generating Transformer Visualizations for Lesson 7")
    print("=" * 60)
    print()
    
    generate_self_attention_heatmap()
    generate_causal_attention_heatmap()
    generate_positional_encoding_visualization()
    generate_multihead_attention_visualization()
    generate_attention_formula_diagram()
    generate_bert_vs_gpt_comparison()
    generate_transformer_block_diagram()
    generate_attention_flow_diagram()
    
    print()
    print("=" * 60)
    print("All visualizations generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()


