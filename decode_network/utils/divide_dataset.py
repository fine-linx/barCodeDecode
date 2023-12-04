# by CSDN 迪菲赫尔曼
import os
import random
import shutil

# 设置随机数种子
random.seed(233)

# 定义文件夹路径
root_dir = "E:/work/barCode/20231127_img/folder_2/cropped/"
flag = "_1127_"
image_dir = root_dir
output_dir = "E:/work/barCode/net_dataset_v1/"
# 定义训练集、验证集和测试集比例
train_ratio = 0.8
valid_ratio = 0.1
test_ratio = 0.1

# 获取所有图像文件和标签文件的文件名（不包括文件扩展名）
image_filenames = [os.path.splitext(f)[0] for f in os.listdir(image_dir)]

# 随机打乱文件名列表
random.shuffle(image_filenames)

# 计算训练集、验证集和测试集的数量
total_count = len(image_filenames)
train_count = int(total_count * train_ratio)
valid_count = int(total_count * valid_ratio)
test_count = total_count - train_count - valid_count

# 定义输出文件夹路径
train_image_dir = os.path.join(output_dir, 'train')
valid_image_dir = os.path.join(output_dir, 'valid')
test_image_dir = os.path.join(output_dir, 'test')

# 创建输出文件夹
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(valid_image_dir, exist_ok=True)
os.makedirs(test_image_dir, exist_ok=True)

# 将图像和标签文件划分到不同的数据集中
for i, filename in enumerate(image_filenames):
    if i < train_count:
        output_image_dir = train_image_dir
    elif i < train_count + valid_count:
        output_image_dir = valid_image_dir
    else:
        output_image_dir = test_image_dir

    # 复制图像文件
    src_image_path = os.path.join(image_dir, filename + '.png')
    dst_image_path = os.path.join(output_image_dir, filename + flag + '.png')
    shutil.copy(src_image_path, dst_image_path)
