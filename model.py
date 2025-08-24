import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, dim_embedding=768, norm_layer=None):
        super().__init__()

        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size # [img_size, img_size] (= [224, 224])
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size # [patch_size, patch_size] (= [16, 16])
        self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1]) # 按patch_size切patches, 切完之后的形状: [img_size / patch_size, img_size / patch_size] (= [14, 14])
        self.num_patches = self.grid_size[0] * self.grid_size[1] # 切出patches的总个数 (14 * 14 = 196)

        # 定义卷积层
        self.conv = nn.Conv2d(in_channels, dim_embedding, kernel_size=self.patch_size, stride=self.patch_size) # [Batches, 3, 224, 224] -> [Batches, 768, 14, 14]

        # 定义归一化层
        self.norm = norm_layer(dim_embedding) if norm_layer is not None else nn.Identity()


    # X: [Batches, in_channels, H, W] = [Batches, 3, 224, 224]
    def forward(self, X):
        B, C, H, W = X.shape

        # 检查图像高宽和预先设定是否一致，不一致则报错 (ViT与CNN不同, 输入图片高宽必须固定, 因为后面有接FC层)
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size {[H, W]} doesn't match the model {[self.img_size[0], self.img_size[1]]}."

        X = self.conv(X).flatten(2) # [B, 768, 14, 14]  --flatten->  [B, 768, 196]
        X = self.norm(X.transpose(1, 2)) # [B, 196, 768], 归一化处理

        return X


class Attention(nn.Module):
    def __init__(self,
                 dim_token, # 输入token的dim
                 num_heads=8,
                 qkv_bias=False, # 计算QKV时是否启用偏置 (因为此处QKV是通过FC层产生的, 而非公式中写的矩阵乘法, 当然本质是一样的)
                 qk_scale=None, # 公式 Attention = softmax(QK^T / \sqrt{d_k}) 中的\sqrt{d_k}, 若无输入则默认为\sqrt{d_k}
                 attention_drop_ratio=0., # 计算出Attention = softmax(QK^T / qk_scale)后, 先dropout, 再乘V, 得到初步output
                 linear_drop_ratio=0. # 得到初步output后, 先输入一个FC层, 最后dropout得到最终output
                 ):
        super().__init__()

        self.dim_token = dim_token

        self.num_heads = num_heads

        self.dim_head = dim_token // num_heads # Multi-head时, 此处采取的策略是最终生成的num_heads个output都是低维的, 拼起来成为最终的output, 而每个head里QKV, output维度都保持一致, 即dim_head

        self.qk_scale = qk_scale or self.dim_head ** -0.5 # 遇到None时用or后面接默认值, 即若qk_scale is None, 则self.qk_scale = \sqrt{dim_scale}, 也就与公式一致

        # 当输入为张量时, Linear层只变换最后一个维度
        # 即input: [Batches, num_patches + 1, dim_token] -> [Batches, num_patches + 1, 3 * dim_token]
        self.qkv_layer = nn.Linear(dim_token, 3 * dim_token, bias=qkv_bias)

        self.attention_drop = nn.Dropout(attention_drop_ratio)

        self.linear = nn.Linear(dim_token, dim_token)
        self.linear_dropout = nn.Dropout(linear_drop_ratio)


    # X: [batch_size, num_patches + 1, dim_embedding (dim_token)], 其中num_patches + 1是因为加了class_token
    def forward(self, X):
        B, N, C = X.shape # [batch_size, num_patches + 1, dim_token]

        qkv = self.qkv_layer(X).reshape([B, N, 3, self.num_heads, self.dim_head]).permute(2, 0, 3, 1, 4) # [3, B, num_heads, N, dim_head]
        Q, K, V = qkv[0], qkv[1], qkv[2] # [B, num_heads, N, dim_heads]

        attention = (Q @ K.transpose(2, 3)) * self.scale # [B, num_heads, N, N], 处理高维张量时, 矩阵乘法只在最后两个维度进行

        # 回顾 Attention = softmax(Q K^T / scale), 其中QK^T的每一行代表每个token的attention向量, 因此应该为矩阵的每一行进行softmax操作 (将QKV视作2维矩阵)
        # 现在attention: [B, num_heads, N, N], 不妨设B=1, num_heads=2, N=3, 即可视觉化成2个3x3的矩阵 (每个注意力头一个)
        # Head 0:               Head 1:
        # [[a, b, c],           [[p, q, r],
        #  [d, e, f],            [s, t, u],
        #  [g, h, i]]            [v, w, x]]
        # softmax(dim=-1)就是对最后一个维度softmax, 也就是对于每个样本 (B)的每个注意力头 (N)的每一行 (N), 将其归一化为一个概率分布, 其和为1
        attention = attention.softmax(dim=-1)
        attention = self.attention_drop(attention) # [B, num_heads, N, N], dropout操作作用于每一个元素, 与输入张量维度、形状无关

        # @: multiply -> [B, num_heads, N, dim_heads]
        # transpose -> [B, N, num_heads, dim_heads]
        # reshape -> [B, N, dim_token (dim_embedding)], 这一步reshape其实就相当于把每个head的output给concat在一起了, 从而形成最终的output
        X = (attention @ V).transpose(1, 2).reshape([B, N, self.dim_token]) # [B, N, dim_token]

        X = self.linear(X) # [B, N, dim_token]
        X = self.linear_dropout(X) # [B, N, dim_token]

        return X # [B, N, dim_token] = [Batches, num_patches + 1, dim_embedding]