import os
import random
import shutil

if __name__ == '__main__':
    folder = "E:/work/barCode/net_dataset4/valid/"
    target_folder = folder + "sub/"
    os.makedirs(target_folder, exist_ok=True)
    ratio = 0.5
    for file in os.listdir(folder):
        if file.endswith(".png"):
            if random.random() <= ratio:
                shutil.copy(folder + file, target_folder + file)
