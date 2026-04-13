def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    """
    预计算 RoPE (Rotary Position Embedding) 的频率矩阵
    
    RoPE 通过旋转矩阵将位置信息编码到 Query 和 Key 中，使模型能够理解 token 的相对位置。
    本函数预计算所有位置的 cos 和 sin 值，避免在每次前向传播时重复计算。
    
    支持 YaRN (Yet another RoPE extensioN) 外推方法，可以处理超过训练时最大长度的序列。
    
    Args:
        dim: 每个注意力头的维度（head_dim）
        end: 最大序列长度（默认 32768）
        rope_base: RoPE 的基频率参数（默认 1e6）
        rope_scaling: RoPE 外推配置字典（YaRN 方法），如果为 None 则不使用外推
        
    Returns:
        freqs_cos: 预计算的 cos 值 [end, dim]
        freqs_sin: 预计算的 sin 值 [end, dim]
    """
    # ========== 步骤 1：计算基础频率
    # 预计算 RoPE 的 cos/sin，减少每步推理开销
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            # 当推理长度超过训练长度时，按 YaRN 对高频部分做平滑拉伸
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    # RoPE 的“半维旋转”：前后半维交换并取负，实现复数旋转等价变换
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed
