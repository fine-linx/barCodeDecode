import os
import shutil

if __name__ == '__main__':
    # 指定原始文件夹路径和目标文件夹路径
    source_folder = "../../db/20231023"
    # target_folder =

    # # 创建目标文件夹
    # if not os.path.exists(target_folder):
    #     os.makedirs(target_folder)

    # 初始化计数器和子文件夹编号
    count = 0
    folder_num = 1

    # 循环遍历原始文件夹中的所有文件
    for filename in os.listdir(source_folder):
        if filename.endswith("JPG"):
            # 构造原始文件路径和目标文件路径
            source_path = os.path.join(source_folder, filename)
            target_path = os.path.join(source_folder, f'folder_{folder_num}', filename)

            # 如果目标子文件夹不存在，则创建
            if not os.path.exists(os.path.dirname(target_path)):
                os.makedirs(os.path.dirname(target_path))

            # 移动文件到目标子文件夹中
            shutil.move(source_path, target_path)

            # 计数器加1
            count += 1

            # 如果当前子文件夹中已经有5000个文件，则重新创建一个子文件夹
            if count == 5000:
                count = 0
                folder_num += 1
