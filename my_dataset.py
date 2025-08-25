import os
import json
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def read_split_data(root: str, val_rate: float = 0.2):
    """
    功能: 划分数据集为训练集和验证集，并返回图片路径、标签以及发现的类别总数。
    参数:
        root: 数据集根目录，目录下每个子文件夹代表一个类别。
        val_rate: 验证集所占的比例。
    返回:
        训练集图片路径列表, 训练集标签列表, 验证集图片路径列表, 验证集标签列表, 类别总数
    """
    random.seed(0)
    assert os.path.exists(root), f"dataset root: {root} does not exist."

    # 自动寻找数据根目录下的所有类别文件夹
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    flower_class.sort()

    # --- 🌟 修复: 动态地从文件夹数量获取类别总数 ---
    num_classes = len(flower_class)

    # 创建类别到索引的映射
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_class_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]

    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        images.sort()

        image_class = class_indices[cla]
        every_class_num.append(len(images))

        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print(f"在数据集中共发现 {sum(every_class_num)} 张图片。")
    print(f"共发现 {num_classes} 个类别。")  # 打印出发现的类别数
    print(f"{len(train_images_path)} 张图片用于训练。")
    print(f"{len(val_images_path)} 张图片用于验证。")

    # --- 🌟 修复: 返回发现的类别总数 ---
    return train_images_path, train_images_label, val_images_path, val_images_label, num_classes


class MyDataSet(Dataset):
    """自定义数据集类"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode != 'RGB':
            raise ValueError(f"image: {self.images_path[item]} isn't RGB mode.")
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
