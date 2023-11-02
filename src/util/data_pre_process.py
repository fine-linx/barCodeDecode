import json
import os
import shutil
from collections import Counter


def main():
    folder = "E:/work/barCode/1030_json/"
    image_folder = "E:/work/barCode/20231030/"
    result_folder = os.path.join(image_folder, "results")

    if not os.path.exists(result_folder):
        os.mkdir(result_folder)

    result_dict = Counter()

    for file in os.listdir(folder):
        if file.endswith(".json"):
            json_file_path = os.path.join(folder, file)
            with open(json_file_path) as f:
                parsed_json = json.load(f)
                if parsed_json["result"] == "OK":
                    img_name = parsed_json["image file name"][0]
                    read_data = parsed_json["read result"][0]["read data"]
                    decode_result = read_data[1::2]
                    result_count = result_dict[decode_result]
                    result_dict[decode_result] += 1

                    new_img_name = f"{decode_result}_{result_count}.JPG"
                    shutil.move(os.path.join(image_folder, img_name), os.path.join(result_folder, new_img_name))


if __name__ == "__main__":
    main()
