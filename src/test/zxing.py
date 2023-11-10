import pyzxing

if __name__ == '__main__':
    reader = pyzxing.BarCodeReader()
    file = "C:/Users/PC/Desktop/cropped/8033873185026_2_rotated_0.png"
    barcode = reader.decode(file)
    data = barcode[0].get("parsed")
    print(data.decode("utf-8"))
    # folder = "../../db/final_unresolved/rotated/cropped/"
    # files = os.listdir(folder)
    # all_barcode = 0
    # right = 0
    # for file in files:
    #     all_barcode += 1
    #     barcode = reader.decode(folder + file)
    #     data = barcode[0].get("parsed")
    #     if data:
    #         right += 1
    #     print(all_barcode, end="\t")
    #     print(barcode[0].get("parsed"))
    # print("all: ", all_barcode)
    # print("right: ", right)
    # print("acc: ", right / all_barcode if all_barcode > 0 else 0)
