import os
import math
import argparse

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models  # 导入torchvision中的models
from tqdm import tqdm

# 复用我们之前创建好的数据集加载代码
from my_dataset import read_split_data, MyDataSet


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    训练一个epoch的函数。
    这个函数与ViT训练脚本中的完全一样，因为训练逻辑是通用的。
    """
    model.train()  # 设置模型为训练模式
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, desc=f"[训练 Epoch {epoch}]")
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))

        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.set_postfix(loss=accu_loss.item() / (step + 1),
                                acc=accu_num.item() / sample_num)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    """
    评估一个epoch的函数。
    这个函数也与ViT训练脚本中的完全一样。
    """
    model.eval()  # 设置模型为评估模式
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)

    sample_num = 0
    data_loader = tqdm(data_loader, desc=f"[验证 Epoch {epoch}]")
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.set_postfix(loss=accu_loss.item() / (step + 1),
                                acc=accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def main(args):
    # 1. 设备选择
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    # 2. 创建权重保存目录
    # 为了和ViT区分，我们保存在一个新的文件夹 'resnet_weights'
    if not os.path.exists("./resnet_weights"):
        os.makedirs("./resnet_weights")

    # 3. Tensorboard设置
    tb_writer = SummaryWriter(log_dir="runs/resnet_experiment")

    # 4. 数据集准备 (与ViT完全一致)
    # 动态获取类别数，这是非常好的实践
    train_images_path, train_images_label, val_images_path, val_images_label, num_classes = read_split_data(
        args.data_path)
    args.num_classes = num_classes

    # 5. 定义数据预处理/增强策略 (与ViT完全一致)
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        # 使用ImageNet的均值和标准差
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 6. 实例化数据集和DataLoader (与ViT完全一致)
    train_dataset = MyDataSet(images_path=train_images_path, images_class=train_images_label,
                              transform=data_transform["train"])
    val_dataset = MyDataSet(images_path=val_images_path, images_class=val_images_label, transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f'每个进程使用 {nw} 个 dataloader workers')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                               num_workers=nw, collate_fn=train_dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                             num_workers=nw, collate_fn=val_dataset.collate_fn)

    # --- 核心区别：模型创建 ---
    # 7. 创建ResNet34模型并加载预训练权重
    # PyTorch 1.9+ 推荐使用新的 weights API，它能自动处理权重加载
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    print("成功加载ResNet34在ImageNet上的预训练权重。")

    # 8. 冻结所有层 (可选，但推荐用于微调)
    if args.freeze_layers:
        for param in model.parameters():
            param.requires_grad = False
        print("已冻结所有预训练层。")

    # 9. 替换分类头
    # ResNet的分类层叫做 'fc' (fully connected)
    # 首先获取 'fc' 层的输入特征数
    in_features = model.fc.in_features
    # 然后用一个新的、输出维度为我们数据集类别数的全连接层来替换它
    # 这个新的层默认是 `requires_grad=True` 的，所以可以被训练
    model.fc = nn.Linear(in_features, args.num_classes)
    model.to(device)
    print(f"已将分类头替换为输出 {args.num_classes} 个类别的新层。")

    # 10. 定义优化器
    # 只将需要训练的参数（即我们新加的分类头'fc'）传入优化器
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)

    # 11. 定义学习率调度器 (与ViT完全一致)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # 12. 开始训练 (与ViT完全一致)
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model=model, optimizer=optimizer, data_loader=train_loader,
                                                device=device, epoch=epoch)
        scheduler.step()
        val_loss, val_acc = evaluate(model=model, data_loader=val_loader, device=device, epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        # 保存到新的文件夹
        torch.save(model.state_dict(), f"./resnet_weights/model-{epoch}.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 保留所有与ViT脚本相同的参数，以确保公平比较
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    parser.add_argument('--data-path', type=str, default="../data/flower_photos")

    # 冻结权重是一个重要的超参数，保持默认开启
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    main(opt)
