"""
demo代码展示正余弦位置编码的实现
"""
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # pe: [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # position: [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # div_term: [d_model/2]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / d_model)
        )

        # 偶数维
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数维
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加 batch 维度，变成 [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # register_buffer: 不是可训练参数，但会随着 model.to(device) 一起移动
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x
    
batch_size = 2
seq_len = 10
d_model = 128

x = torch.randn(batch_size, seq_len, d_model)

pos_encoder = PositionalEncoding(d_model=d_model, max_len=50)
out = pos_encoder(x)

print(out.shape)  # torch.Size([2, 10, 128])
print(out)