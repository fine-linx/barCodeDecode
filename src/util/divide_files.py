import os
import random
import shutil

if __name__ == '__main__':
    # 指定原始文件夹路径和目标文件夹路径
    source_folder = "E:/work/barCode/20231130_img/results/"
    # target_folder =

    # # 创建目标文件夹
    # if not os.path.exists(target_folder):
    #     os.makedirs(target_folder)

    # 初始化计数器和子文件夹编号
    count = 0
    folder_num = 1

    folder_1 = "E:/work/barCode/20231130_img/folder_1/"
    folder_2 = "E:/work/barCode/20231130_img/folder_2/"
    os.makedirs(folder_1, exist_ok=True)
    os.makedirs(folder_2, exist_ok=True)

    # 循环遍历原始文件夹中的所有文件
    for filename in os.listdir(source_folder):
        if filename.endswith(".jpg") or filename.endswith("JPG"):
            if random.random() > 0.5:
                shutil.move(os.path.join(source_folder, filename), folder_1 + filename)
            else:
                shutil.move(os.path.join(source_folder, filename), folder_2 + filename)
            # 构造原始文件路径和目标文件路径
            # source_path = os.path.join(source_folder, filename)
            # target_path = os.path.join(source_folder, f'folder_{folder_num}', filename)
            #
            # # 如果目标子文件夹不存在，则创建
            # if not os.path.exists(os.path.dirname(target_path)):
            #     os.makedirs(os.path.dirname(target_path))
            #
            # # 移动文件到目标子文件夹中
            # shutil.move(source_path, target_path)
            #
            # # 计数器加1
            # count += 1
            #
            # # 如果当前子文件夹中已经有5000个文件，则重新创建一个子文件夹
            # if count == 10000:
            #     count = 0
            #     folder_num += 1
