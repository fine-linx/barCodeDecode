import os
import shutil


def main():
    folder = "E:/work/barCode/net_dataset3/resolved/valid/sharp/"
    source_folder = "E:/work/barCode/net_dataset3/resolved/blur/"
    target_folder = "E:/work/barCode/net_dataset3/resolved/valid/blur/"
    os.makedirs(target_folder, exist_ok=True)
    files = os.listdir(folder)
    for file in files:
        shutil.copy(os.path.join(source_folder, file), os.path.join(target_folder, file))


if __name__ == '__main__':
    main()
