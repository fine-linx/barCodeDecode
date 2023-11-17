import os

if __name__ == '__main__':
    folder = "D:/work/barCodeDecode/db/20231019/folder_3/rotated/"
    target_folder = folder + "unresolved/network/"
    for file in os.listdir(target_folder):
        if file.endswith(".png"):
            try:
                os.remove(folder + file)
            except FileNotFoundError:
                print(file, "not found")
