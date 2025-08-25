import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, dim_embedding=768, norm_layer=None):
        super().__init__()

        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size # (int, int)型的list
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size # (int, int)型的list

        # 定义卷积层
        # input: [B, in_c, H, W] -> [B, dim_embedding, H_new, W_new] = [B, dim_embedding, H // patch_size, W // patch_size]
        self.conv = nn.Conv2d(in_channels, dim_embedding, kernel_size=self.patch_size, stride=self.patch_size)

        # 定义归一层
        self.norm = norm_layer or nn.Identity() # 遇到None时用or后接默认值


    # X: [B, N, H, W]
    def forward(self, X):
        B, N, H, W = X.shape

        # 检查输入图像的高宽是否与预先设定一致 (ViT与CNN不同, 不可随意更改输入图像高宽, 因为后面还接了FC层)
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size {[H, W]} doesn't match the model {[self.img_size[0], self.img_size[1]]}."

        # self.conv -> [B, dim_embedding, H_new, W_new]
        # flatten -> [B, dim_embedding, num_patches] = [B, dim_embedding, H_new * W_new]
        # transpose -> [B, num_patches, dim_embedding]
        X = self.conv(X).flatten(2).transpose(1, 2)
        X = self.norm(X) # 归一化

        return X # [B, num_patches, dim_embedding]



# Attention -> Dropout -> Output -> Dropout
class Attention(nn.Module):
    def __init__(self,
                 dim_token, # 输入token的dim
                 num_heads=8,
                 qkv_bias=False, # 计算QKV时是否启用偏置 (因为此处QKV是通过FC层产生的, 而非公式中写的矩阵乘法, 当然本质是一样的)
                 qk_scale=None, # 公式 Attention = softmax(QK^T / \sqrt{d_k}) 中的\sqrt{d_k}, 若无输入则默认为1 / \sqrt{d_k}
                 attention_drop_ratio=0., # 计算出Attention = softmax(QK^T * qk_scale)后, 先dropout, 再乘V, 得到初步output
                 linear_drop_ratio=0. # 得到初步output后, 先输入一个FC层, 最后dropout得到最终output
                 ):
        super().__init__()

        self.dim_token = dim_token

        self.num_heads = num_heads

        dim_head = dim_token // num_heads # Multi-head时, 此处采取的策略是最终生成的num_heads个output都是低维的, 拼起来成为最终的output, 而每个head里QKV, output维度都保持一致, 即dim_head

        self.qk_scale = qk_scale or dim_head ** -0.5 # 遇到None时用or后面接默认值, 即若qk_scale is None, 则self.qk_scale = \sqrt{dim_scale}, 也就与公式一致

        # 当输入为张量时, Linear层只变换最后一个维度
        # 即input: [Batches, num_patches + 1, dim_token] -> [Batches, num_patches + 1, 3 * dim_token]
        self.qkv_layer = nn.Linear(dim_token, 3 * dim_token, bias=qkv_bias)

        self.attention_drop = nn.Dropout(attention_drop_ratio)

        self.linear = nn.Linear(dim_token, dim_token)
        self.linear_dropout = nn.Dropout(linear_drop_ratio)


    # X: [batch_size, num_patches + 1, dim_embedding (dim_token)], 其中num_patches + 1是因为加了class_token
    def forward(self, X):
        B, N, C = X.shape # [batch_size, num_patches + 1, dim_token]

        qkv = self.qkv_layer(X).reshape([B, N, 3, self.num_heads, C // self.num_heads]).permute(2, 0, 3, 1, 4) # [3, B, num_heads, N, dim_head]
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
        X = (attention @ V).transpose(1, 2).reshape([B, N, C]) # [B, N, dim_token]

        X = self.linear(X) # [B, N, dim_token]
        X = self.linear_dropout(X) # [B, N, dim_token]

        return X # [B, N, dim_token] = [Batches, num_patches + 1, dim_embedding]



class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, active_layer=nn.GELU, dropout=0.):
        super().__init__()

        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            active_layer(),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(dropout)
        )


    def forward(self, X):
        return self.net(X)



def drop_path(X, drop_prob: float = 0., training: bool = False):
    # drop概率为0或者为推理 (eval)模式时, 不执行DropPath操作
    if drop_prob == 0. or not training:
        return X
    keep_prob = 1 - drop_prob

    # 准备一个与输入x具有相同批次大小（batch_size）但其他维度均为1的张量形状
    # 例如, x的形状是 [batch_size, N, C], 则shape为 [batch_size, 1, 1]
    # 这样做是为了确保随机掩码 (random mask)是逐样本的 (per-sample), 但对每个样本的所有通道、高、宽都保持一致
    # 以X: [B, N, C]为例, (X.shape[0],) = (B,)为一个元组; (1,) * 2 = (1, 1); (B,) + (1, 1) = (B, 1, 1)
    shape = (X.shape[0],) + (1,) * (X.ndim - 1) # (B, 1, 1)

    # 生成一个随机张量, 其值在区间 [keep_prob, 1 + keep_prob) 内均匀分布
    # 例如, keep_prob=0.9，则随机值在 [0.9, 1.9) 之间
    random_tensor = keep_prob + torch.rand(shape, dtype=X.dtype, device=X.device) # [B, 1, 1]

    random_tensor.floor_() # [B, 1, 1], 向下取整, 随机张量中的每个元素有 keep_prob 的概率变为1, 有 drop_prob 的概率变为0

    # 1. 首先对输入 x 进行缩放：X.div(keep_prob), 即 X / keep_prob
    #    - 这是因为在测试时, 我们会使用整个网络 (所有路径都存活)
    #    - 为了保持训练和测试时的输出期望 (均值)一致, 需要在训练时对保留的路径进行缩放
    #    - 例如, 如果keep_prob=0.9, 我们只使用了90%的路径, 那么为了补偿, 需要将输出放大到原来的 1/0.9 倍
    #    - 详细可见https://blog.csdn.net/weixin_44012667/article/details/144516551
    # 2. random_tensor充当掩码,
    #    - 逐样本 (per-sample)操作, 对于输入的一个batch中的每个样本，都会独立地以概率 drop_prob 决定是否跳过该层 (即让残差分支的输出为0)
    output = X.div(keep_prob) * random_tensor # [B, N, C]

    return X # [B, N, C]



# 逐样本 (per-sample)操作, 对于输入的一个batch中的每个样本，都会独立地以概率 drop_prob 决定是否跳过该层 (即让残差分支的输出为0)
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    # X: [B, N, C]
    def forward(self, X):
        return drop_path(X, self.drop_prob, self.training) # [B, N, C]



