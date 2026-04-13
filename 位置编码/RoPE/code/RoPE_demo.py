import torch
import math

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    把最后一维按两两分组，执行:
    [x1, x2] -> [-x2, x1]
    它的数学本质是乘以复数 i, 从而进行了逆时针 90° 的旋转

    例如:
    [a, b, c, d] -> [-b, a, -d, c]
    """
    x1 = x[..., ::2]   # 偶数位
    x2 = x[..., 1::2]  # 奇数位
    
    x_rot = torch.stack((-x2, x1), dim=-1)
    return x_rot.flatten(start_dim=-2)

def build_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    device=None,
    dtype=None,
):
    """
    构造 RoPE 所需的 cos / sin 缓存

    返回:
        cos: [seq_len, head_dim]
        sin: [seq_len, head_dim]
    """
    assert head_dim % 2 == 0, "head_dim 必须是偶数"

    # 对应公式中的 theta_j = base^(-2j/head_dim)
    half_dim = head_dim // 2
    freq_seq = torch.arange(half_dim, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (freq_seq / half_dim))

    # 位置 [0, 1, 2, ..., seq_len-1]
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)

    # [seq_len, half_dim]
    freqs = torch.outer(positions, inv_freq)

    # 因为每对维度共享同一个角度，所以复制成 [seq_len, head_dim]
    emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)

    cos = emb.cos()
    sin = emb.sin()

    if dtype is not None:
        cos = cos.to(dtype)
        sin = sin.to(dtype)

    return cos, sin

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    对输入 x 应用 RoPE

    参数:
        x:   [batch, num_heads, seq_len, head_dim]
        cos: [seq_len, head_dim]
        sin: [seq_len, head_dim]

    返回:
        旋转后张量，shape 不变
    """
    # 扩展到可广播形状
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]

    return x * cos + rotate_half(x) * sin


# 假设:
# q, k shape = [batch, num_heads, seq_len, head_dim]

batch = 2
num_heads = 8
seq_len = 16
head_dim = 64

q = torch.randn(batch, num_heads, seq_len, head_dim)
k = torch.randn(batch, num_heads, seq_len, head_dim)

cos, sin = build_rope_cache(
    seq_len=seq_len,
    head_dim=head_dim,
    device=q.device,
    dtype=q.dtype,
)

q_rope = apply_rope(q, cos, sin)
k_rope = apply_rope(k, cos, sin)

print(q_rope.shape)  # [2, 8, 16, 64]
print(k_rope.shape)  # [2, 8, 16, 64]