# Transformer Implementation

A simple implementation of the Transformer model introduced in the "Attention Is All You Need" paper.

## Training

- Framework: PyTorch
- Hardware: NVIDIA A100 24GB GPU on Lightning AI
- Architecture: Standard Transformer encoder-decoder

## Testing

To test the trained model using inference:

1. Open the Jupyter notebook provided in the repository
2. Load the trained model checkpoint
3. Run inference examples through the notebook cells

## References

- Paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Tutorial by Umar Jamil:
  - [Attention is all you need (Transformer) - Model explanation (including math), Inference and Training](https://youtu.be/bCz4OMemCcA)
  - [Coding a Transformer from scratch on PyTorch, with full explanation, training and inference](https://youtu.be/ISNdQcPhsts)

---

<details>
<summary>LOGBOOK</summary>

### 1. pytorch matmul

<details>
<summary>Batched MatMul in Attention</summary>

</br>

(B, h, S, d_k) × (B, h, d_k, S) → (B, h, S, S)

---

### 1. Shapes before multiplication
- **Query (Q):** `(B, h, S, d_k)`  
- **Keyᵀ (Kᵀ):** `(B, h, d_k, S)`  

---

### 2. PyTorch matmul rule
- `torch.matmul` multiplies the **last two dimensions** as matrices.  
- All earlier dimensions (`B, h`) are treated as **batch dimensions** and are broadcast automatically.  

---

### 3. Result
- Each slice `(S, d_k) × (d_k, S)` → `(S, S)`  
- This happens independently for every `(B, h)` pair.  
- Final output shape: `(B, h, S, S)`  

---

### 4. Dim intuition
- `B`: batch size, keeps computations independent across examples.  
- `h`: number of heads, keeps computations independent across heads.  
- `S`: sequence length, tokens in the sequence.  
- `d_k`: feature dimension per head, inner dimension that cancels in matmul.  

---

</details>

</details>
