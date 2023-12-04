import os
import random
import shutil


def main():
    folder = "E:/work/barCode/net_dataset3/resolved/valid/sharp/"
    source_folder = "E:/work/barCode/net_dataset3/resolved/blur/"
    target_folder = "E:/work/barCode/net_dataset3/resolved/valid/blur/"
    os.makedirs(target_folder, exist_ok=True)
    files = os.listdir(folder)
    for file in files:
        shutil.copy(os.path.join(source_folder, file), os.path.join(target_folder, file))


def random_select():
    folder = "E:/work/barCode/20231119_img/"
    target_folder = "E:/work/barCode/20231119_img/selected/"
    os.makedirs(target_folder, exist_ok=True)
    files = os.listdir(folder)
    random.shuffle(files)
    for file in files:
        if random.random() < 0.33:
            shutil.copy(os.path.join(folder, file), os.path.join(target_folder, file))


def change_ext():
    folder = "E:/work/barCode/yolo_dataset/images/"
    files = os.listdir(folder)
    for file in files:
        if file.endswith(".JPG"):
            shutil.move(folder + file, folder + file.replace(".JPG", ".jpg"))


def add_noise_labels():
    folder = "E:/work/barCode/yolo_dataset/labels/"
    files = os.listdir(folder)
    for file in files:
        for suffix in ["localvar", "gaussian", "dilate", "erode"]:
            new_file = file.replace(".txt", "_" + suffix + ".txt")
            shutil.copy(folder + file, folder + new_file)


if __name__ == '__main__':
    # main()
    # random_select()
    # change_ext()
    add_noise_labels()
