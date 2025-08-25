import torch
# 确保你的 model.py 文件在当前目录下，或者在Python的搜索路径中
from model import vit_base_patch16_224
import os


def load_pretrained_weights(model, weights_path, num_classes):
    """
    加载预训练权重，并处理分类头不匹配的问题
    参数:
        model: 你实例化的模型
        weights_path: 预训练权重文件的路径
        num_classes: 你自己任务的类别数
    """
    # 1. 加载预训练权重字典
    print(f"Loading pretrained weights from {weights_path}")
    weights_dict = torch.load(weights_path, map_location="cpu")

    # 2. 移除与分类头相关的权重
    # 预训练模型的分类头（head）的权重维度是针对原任务的（比如1000类或21843类）
    # 我们的新任务是5类，维度不匹配，所以必须移除
    # 通常分类头的权重key是 'head.weight' 和 'head.bias'
    # 有些模型还有一个 'pre_logits' 层，如果你的模型没有，可以忽略
    keys_to_remove = ['head.weight', 'head.bias']

    # 检查 'pre_logits' 相关的键是否存在于权重文件中，如果存在也一并删除
    if 'pre_logits.fc.weight' in weights_dict:
        keys_to_remove.extend(['pre_logits.fc.weight', 'pre_logits.fc.bias'])

    for k in keys_to_remove:
        if k in weights_dict:
            print(f"Removing key '{k}' from pretrained weights.")
            del weights_dict[k]

    # 3. 调整位置编码（Position Embedding）的尺寸
    # 预训练模型的位置编码可能和你的模型不完全匹配，特别是cls_token部分
    # 这里的代码检查并调整位置编码的维度以匹配你的模型
    pos_embed_pretrained = weights_dict['pos_embed']
    pos_embed_model = model.state_dict()['pos_embed']
    if pos_embed_pretrained.shape != pos_embed_model.shape:
        print(f"Resizing position embedding from {pos_embed_pretrained.shape} to {pos_embed_model.shape}")
        # 从预训练权重中提取除了cls_token之外的patch embedding部分
        pos_embed_patches = pos_embed_pretrained[:, 1:, :]
        # 创建一个新的、尺寸正确的位置编码张量
        new_pos_embed = torch.cat((pos_embed_pretrained[:, 0:1, :], pos_embed_patches), dim=1)
        weights_dict['pos_embed'] = new_pos_embed

    # 4. 加载权重到模型
    # strict=False 参数非常重要，它允许加载的权重字典中缺少某些键（我们新初始化的分类头）
    # 或者模型中缺少某些键（如果权重文件里有多余的键）
    missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")

    return model


if __name__ == '__main__':
    # --- 使用示例 ---
    # a. 定义你的模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = 5  # 花卉分类是5类
    model = vit_base_patch16_224(num_classes=num_classes).to(device)

    # b. 指定预训练权重路径（请确保你已经下载了）
    # vit_base_patch16_224 在ImageNet-21k上预训练的权重下载地址:
    # https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    weights_path = "./pretrained_weights/vit_base_patch16_224_in21k.pth"

    # c. 调用函数加载权重
    if os.path.exists(weights_path):
        model = load_pretrained_weights(model, weights_path, num_classes)
    else:
        print(f"Weight file not found at {weights_path}, training from scratch.")