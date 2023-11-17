import os
import shutil

if __name__ == '__main__':
    # folder = "../datasets/barCode/all/"
    # file_names = list()
    # for file in os.listdir(folder + "labels/"):
    #     with open(folder + "labels/" + file, "r") as f:
    #         lines = f.readlines()
    #         if len(lines) > 1:
    #             print(file)
    #             file_names.append(file)
    # for file in file_names:
    #     shutil.move(folder + "labels/" + file, folder + "labels_multi/" + file)
    #     img_name = file.split(".")[0] + ".jpg"
    #     shutil.move(folder + "images/" + img_name, folder + "images_multi/" + img_name)
    folder = "E:/work/barCode/20231102_img/results/"
    target_folder = "../datasets/barCode/all/"
    for file in os.listdir(target_folder + "labels9/"):
        img_name = file.split(".")[0] + ".JPG"
        shutil.copy(folder + img_name, target_folder + "images9/" + file.split(".")[0] + ".jpg")
