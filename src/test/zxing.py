import os

import pyzxing

if __name__ == '__main__':
    reader = pyzxing.BarCodeReader()
    folder = "E:/work/barCode/net_dataset3/cropped/test/"
    files = os.listdir(folder)
    all_barcode = 0
    right = 0
    for file in files:
        all_barcode += 1
        print(all_barcode, end="\t")
        barcode = reader.decode(folder + file)
        if barcode:
            data = barcode[0].get("parsed")
            if data:
                right += 1
            print(data)
    print("all: ", all_barcode)
    print("right: ", right)
    print("acc: ", right / all_barcode if all_barcode > 0 else 0)
