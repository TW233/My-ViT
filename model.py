import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, dim_embedding=768, norm_layer=None):
        super().__init__()

        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size # (int, int)型的list
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size # (int, int)型的list

        # patches总数, 在VisionTransformer类中定义位置编码token时会用到
        self.num_patches = (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])

        # 定义卷积层
        # input: [B, in_c, H, W] -> [B, dim_embedding, H_new, W_new] = [B, dim_embedding, H // patch_size, W // patch_size]
        self.proj = nn.Conv2d(in_channels, dim_embedding, kernel_size=self.patch_size, stride=self.patch_size)

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
        X = self.proj(X).flatten(2).transpose(1, 2)
        X = self.norm(X) # 归一化

        return X # [B, num_patches, dim_embedding]



# Attention -> Dropout -> Output -> Dropout
class Attention(nn.Module):
    def __init__(self,
                 dim_token, # 输入token的dim, ViT中通常为dim_embedding
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

        self.scale = qk_scale or dim_head ** -0.5 # 遇到None时用or后面接默认值, 即若qk_scale is None, 则self.qk_scale = \sqrt{dim_scale}, 也就与公式一致

        # 当输入为张量时, Linear层只变换最后一个维度
        # 即input: [Batches, num_patches + 1, dim_token] -> [Batches, num_patches + 1, 3 * dim_token]
        self.qkv = nn.Linear(dim_token, 3 * dim_token, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attention_drop_ratio)

        self.proj = nn.Linear(dim_token, dim_token)
        self.proj_drop = nn.Dropout(linear_drop_ratio)


    # X: [batch_size, num_patches + 1, dim_embedding (dim_token)], 其中num_patches + 1是因为加了class_token
    def forward(self, X):
        B, N, C = X.shape # [batch_size, num_patches + 1, dim_token]

        QKV = self.qkv(X).reshape([B, N, 3, self.num_heads, C // self.num_heads]).permute(2, 0, 3, 1, 4) # [3, B, num_heads, N, dim_head]
        Q, K, V = QKV[0], QKV[1], QKV[2] # [B, num_heads, N, dim_heads]

        attention = (Q @ K.transpose(2, 3)) * self.scale # [B, num_heads, N, N], 处理高维张量时, 矩阵乘法只在最后两个维度进行

        # 回顾 Attention = softmax(Q K^T / scale), 其中QK^T的每一行代表每个token的attention向量, 因此应该为矩阵的每一行进行softmax操作 (将QKV视作2维矩阵)
        # 现在attention: [B, num_heads, N, N], 不妨设B=1, num_heads=2, N=3, 即可视觉化成2个3x3的矩阵 (每个注意力头一个)
        # Head 0:               Head 1:
        # [[a, b, c],           [[p, q, r],
        #  [d, e, f],            [s, t, u],
        #  [g, h, i]]            [v, w, x]]
        # softmax(dim=-1)就是对最后一个维度softmax, 也就是对于每个样本 (B)的每个注意力头 (N)的每一行 (N), 将其归一化为一个概率分布, 其和为1
        attention = attention.softmax(dim=-1)
        attention = self.attn_drop(attention) # [B, num_heads, N, N], dropout操作作用于每一个元素, 与输入张量维度、形状无关

        # @: multiply -> [B, num_heads, N, dim_heads]
        # transpose -> [B, N, num_heads, dim_heads]
        # reshape -> [B, N, dim_token (dim_embedding)], 这一步reshape其实就相当于把每个head的output给concat在一起了, 从而形成最终的output
        X = (attention @ V).transpose(1, 2).reshape([B, N, C]) # [B, N, dim_token]

        X = self.proj(X) # [B, N, dim_token]
        X = self.proj_drop(X) # [B, N, dim_token]

        return X # [B, N, dim_token] = [Batches, num_patches + 1, dim_embedding]



class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, active_layer=nn.GELU, dropout=0.):
        super().__init__()

        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = active_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)


    def forward(self, X):
        X = self.fc1(X)
        X = self.act(X)
        X = self.drop(X)
        X = self.fc2(X)
        X = self.drop(X)

        return X



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



# LayerNorm -> Multihead-Attention -> Dropout / DropPath -> Add (Residual)
# -> LayerNorm -> MLP Block -> Dropout / DropPath -> Add (Residual)
class Encoder_Block(nn.Module):
    def __init__(self,
                 dim_token,  # 输入token的dim, ViT中通常为dim_embedding
                 num_heads, # Attention类, Multi-head个数
                 qkv_bias=False,  # Attention类, 计算QKV时是否启用偏置
                 qk_scale=None,  # Attention类
                 attention_drop_ratio=0.,  # Attention类, 计算完attention后dropout概率
                 drop_ratio=0.,   # Attention类 & MLP类, Attention类最后输出前dropout概率 & MLP类最后dropout概率
                 mlp_ratio=4., # MLP类, 第一个FC层中hidden_features与in_features的比值
                 active_layer=nn.GELU, # MLP类, 激活函数
                 norm_layer=nn.LayerNorm, # 归一化层, 默认为LayerNorm
                 drop_path_prob=0. # DropPath类, droppath概率
                 ):
        super().__init__()

        # 与BatchNorm不同, LayerNorm适用于那些在不同样本之间难以直接比较的情况, 如Transformer中的自注意力机制
        # 在这些模型中, 每个位置上的数据代表了不同的特征, 因此直接归一化可能会失去意义
        # LayerNorm的解决方案是对每个样本的所有特征进行单独归一化, 而不是基于整个批次
        self.norm1 = norm_layer(dim_token)

        self.attn = Attention(dim_token, num_heads, qkv_bias, qk_scale, attention_drop_ratio, drop_ratio)

        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity()

        self.norm2 = norm_layer(dim_token)

        hidden_features = int(dim_token * mlp_ratio)
        self.mlp = MLP(in_features=dim_token, hidden_features=hidden_features,active_layer=active_layer, dropout=drop_ratio)


    # X: [B, N, C] = [Batch, num_patches + 1, dim_token (dim_embedding)]
    def forward(self, X):
        X = X + self.drop_path(self.attn(self.norm1(X))) # 第一个Add (Residual)层
        X = X + self.drop_path(self.mlp(self.norm2(X))) # 第二个Add (Residual)层
        return X



class VisionTransformer(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 dim_embedding=768,
                 embedding_norm_layer=None,
                 num_heads=12,  # Attention类, Multi-head个数
                 qkv_bias=False,  # Attention类, 计算QKV时是否启用偏置
                 qk_scale=None,  # Attention类
                 attention_drop_ratio=0.,  # Attention类, 计算完attention后dropout概率
                 drop_ratio=0.,  # 位置编码 & Attention类 & MLP类, 位置编码后dropout概率 & Attention类最后输出前dropout概率 & MLP类最后dropout概率
                 mlp_ratio=4.,  # MLP类, 第一个FC层中hidden_features与in_features的比值
                 active_layer=None,  # MLP类, 激活函数, 默认为nn.GELU
                 norm_layer=None,  # Encoder_Block类 & MLP Head前, 归一化层, 默认为LayerNorm
                 drop_path_prob=0.,  # DropPath类, DropPath概率
                 representation_size=None, # MLP Head, Pre-Logits里Linear层的out_features个数, 默认None即没有Pre-Logits层
                 embedding_layer=PatchEmbedding, # Patch-Embedding层选用什么结构
                 num_classes=1000, # 最后分类的种类数
                 depth=12 # Encoder_Block重复次数
                 ):
        super().__init__()

        self.num_classes = num_classes # 最后分类的种类数
        self.num_features = self.dim_embedding = dim_embedding # num_features是用于最后MLP Head的最后那个Linear层的输入特征数
        self.num_extra_tokens = 1 # 额外的token数, 这里1表示token for classification

        # 用partial方法提前指定部分参数, 使得调用时输入参数减少 (详见https://blog.csdn.net/qq_39450134/article/details/121871432)
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        active_layer = active_layer or nn.GELU

        # 开始构建网络架构, 先是Patch-Embedding层
        self.patch_embed = embedding_layer(img_size, patch_size, in_channels, dim_embedding, embedding_norm_layer)
        num_patches = self.patch_embed.num_patches # patches总数

        # 定义class token & 位置编码 (position embedding)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim_embedding)) # [1, 1, dim_embedding]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_extra_tokens, dim_embedding)) # [1, num_patches + 1, dim_embedding]

        # 加上位置编码 (position embedding)后的Dropout层
        self.pos_drop = nn.Dropout(p=drop_ratio)

        # Encoder_Block (重复depth次)

        # 先来设置每个Encoder_Block里Dropout/DropPath的概率, 每个Block的概率是不同的, 越往后Dropout/DropPath概率越大
        # 用等差递增数列来定义每个Block的DropPath概率, 最大值为输入的drop_path_ratio (默认为0, 即不采用Dropout/DropPath)
        # 储存为list而非直接"torch.linspace()"存为tensor的原因：
        #     - 1. 避免设备（Device）不匹配问题：使用列表推导式list是创建在内存上, 与设备无关；而tensor是默认创建在CPU上
        #     - 2. 避免不必要的计算图保留：torch.Tensor 会保留计算历史 (用于反向传播), 即使你并不需要, 而且用list还能避免意外梯度计算
        drop_path_prob = [x.item() for x in torch.linspace(0, drop_path_prob, depth)]

        # *[]是解包操作, 将此处的Block组成的list拆解为一个个独立的Block (共depth个)
        self.blocks = nn.Sequential(*[
            Encoder_Block(dim_token=dim_embedding, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                          attention_drop_ratio=attention_drop_ratio, drop_ratio=drop_ratio, mlp_ratio=mlp_ratio,
                          active_layer=active_layer, norm_layer=norm_layer, drop_path_prob=drop_path_prob[i])
            for i in range(depth)
        ])

        # Encoder后的归一化层 (默认为LayerNorm)
        self.norm = norm_layer(dim_embedding)

        # MLP Head块
        # 先是Pre-Logits (如果需要的话), Linear -> tanh
        if representation_size is not None:
            self.has_logits = True # 保存模型结构信息, 在后续训练时会用到
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(dim_embedding, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # MLP Head最后的Linear层
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # 初始化参数
        nn.init.trunc_normal_(self.cls_token, std=0.02) # 截断的正态分布, 标准差std=0.02, 默认范围：[a, b] = [-2., 2.]
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(_init_vit_weights)

    # X: [B, C, H, W]
    def forward(self, X):
        # patch_embedding: [B, C, H, W] -> [B, num_patches, dim_embedding]
        X = self.patch_embed(X) # [B, num_patches, dim_embedding]

        # expand: [1, 1, dim_embedding] -> [B, 1, dim_embedding]
        # 不要直接修改 self.cls_token, 应该创建一个新的局部变量来存储 repeat 后的结果
        # 因为nn.Parameter是可训练的参数, self.cls_token.repeat(...) 会创建一个新的普通张量 (torch.cuda.FloatTensor),
        # 然后你试图用这个普通张量去覆盖原来的 nn.Parameter, 这是不允许的
        # .expand() 比 .repeat() 更高效, 因为它不复制数据, 只是在元数据层面扩展
        cls_token = self.cls_token.repeat([X.shape[0], 1, 1]) # [B, 1, dim_embedding]

        # concat一个token for classification
        X = torch.cat((cls_token, X), dim=1) # [B, num_patches + 1, dim_embedding]

        # 加上位置编码 (position embedding) & Dropout
        X = self.pos_drop(X + self.pos_embed) # [B, num_patches + 1, dim_embedding]

        # Encoder_Blocks
        X = self.blocks(X) # [B, num_patches + 1, dim_embedding]

        # 归一化 (默认LayerNorm)
        X = self.norm(X) # [B, num_patches + 1, dim_embedding]

        # MLP Head块
        # 单独取出class token, 先通过Pre-Logits
        # X[:, 0]: [B, 1, dim_embedding]
        X = self.pre_logits(X[:, 0]) # [B, 1, num_features]

        # MLP Head -- Linear
        X = self.head(X) # [B, 1, num_classes]

        return X



# 初始化VisionTransformer类的参数
def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.Conv2d):
        # mode="fan_in"：当你更关心前向传播过程中激活值的方差稳定性时使用
        # mode="fan_out"：当你更关心反向传播过程中梯度的方差稳定性时使用
        # 对于普通的前馈网络, 两者差异不大, 通常默认使用 fan_in
        # 对于需要特别稳定梯度流动的网络 (如很深的网络或Transformer), fan_out 可能更合适
        # 对于一些特殊结构 (如残差连接), 可能需要调整模式
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)



# ViT-Base Model
def vit_base_patch16_224(num_classes: int = 1000):
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              in_channels=3,
                              dim_embedding=768,
                              depth=12,
                              num_heads=12,
                              qkv_bias=True,
                              representation_size=None,
                              num_classes=num_classes)
    return model