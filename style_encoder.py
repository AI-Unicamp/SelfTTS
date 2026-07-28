import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
import math
import numpy as np

class ReferenceEncoder(nn.Module):
    """
    inputs:  [B, T, n_mels]
    outputs: [B, embedding_dim // 2]
    """

    def __init__(self, num_mel, embedding_dim):
        super().__init__()

        self.num_mel = num_mel
        filters = [1, 32, 32, 64, 64, 128, 128]

        self.convs = nn.ModuleList([
            nn.Conv2d(
                filters[i],
                filters[i + 1],
                kernel_size=3,
                stride=2,
                padding=1,
            )
            for i in range(len(filters) - 1)
        ])

        self.bns = nn.ModuleList([
            nn.BatchNorm2d(f) for f in filters[1:]
        ])

        post_conv_height = self.calculate_post_conv_height(
            num_mel, kernel_size=3, stride=2, pad=1, n_convs=len(self.convs)
        )

        self.gru = nn.GRU(
            input_size=filters[-1] * post_conv_height,
            hidden_size=embedding_dim // 2,
            batch_first=True,
        )

    def forward(self, inputs):
        '''
            x: (B, F=80, T) and needs to become (B, 1, T, F=80)
        '''

        B = inputs.size(0)

        x = inputs.permute(0, 2, 1).unsqueeze(1)

        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x)))

        x = x.transpose(1, 2)
        x = x.contiguous().view(B, x.size(1), -1)

        self.gru.flatten_parameters()
        _, h = self.gru(x)

        return h.squeeze(0)

    @staticmethod
    def calculate_post_conv_height(h, kernel_size, stride, pad, n_convs):
        for _ in range(n_convs):
            h = (h - kernel_size + 2 * pad) // stride + 1
        return h



def main():
    torch.manual_seed(0)

    batch_size = 4
    random_length = np.random.randint(100, 200)

    input_tensor = torch.randn(batch_size, 80, random_length)

    # ============================================================
    # Test ReferenceEncoder
    # ============================================================
    print("\nTesting ReferenceEncoder...")
    re_embedding_dim = 256
    ref_encoder = ReferenceEncoder(num_mel=80, embedding_dim=re_embedding_dim)

    ref_out = ref_encoder(input_tensor)
    print(f"ReferenceEncoder output shape: {ref_out.shape}")
    # Expected: [B, re_embedding_dim // 2]


if __name__ == "__main__":
    main()
