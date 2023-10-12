import cv2
import os
import shutil

def combine_file_data():
    path_img_train = "./dataset/images/train/"
    path_txt_train = "./dataset/labels/train/"
    path_img_val = "./dataset/images/val/"
    path_txt_val = "./dataset/labels/val/"

    for dir in os.listdir(path_img_train):
        src_img_path = os.path.join(path_img_train, dir)
        dest_img_path = os.path.join("./dataset/images/", dir)
        shutil.move(src_img_path, dest_img_path)

        src_txt_path = os.path.join(path_txt_train, f"{dir.split('.')[0]}.txt")
        dest_txt_path = os.path.join("./dataset/labels/", f"{dir.split('.')[0]}.txt")
        shutil.move(src_txt_path, dest_txt_path)

    for dir in os.listdir(path_img_val):
        src_img_path = os.path.join(path_img_val, dir)
        dest_img_path = os.path.join("./dataset/images/", dir)
        shutil.move(src_img_path, dest_img_path)

        src_txt_path = os.path.join(path_txt_val, f"{dir.split('.')[0]}.txt")
        dest_txt_path = os.path.join("./dataset/labels/", f"{dir.split('.')[0]}.txt")
        shutil.move(src_txt_path, dest_txt_path)

def get_img_handwrite():
    path_img = "./dataset/images"
    path_txt = "./dataset/labels"
    i = 0
    for dir in os.listdir(path_img):
        name = dir.split('.')[0]
        file_img = os.path.join(path_img, f"{name}.jpg")
        file_txt = os.path.join(path_txt, f"{name}.txt")
        img = cv2.imread(file_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        (h, w, c) = img.shape
        with open(file_txt, "r") as f:
            for line in f.readlines():
                # format line: <class> <x_center> <y_center> <w> <h>
                line = line.rstrip()
                args = line.split(' ')
                if int(args[0]) == 0:
                    x_center = int(float(args[1]) * w)
                    y_center = int(float(args[2]) * h)
                    w_box = int(float(args[3]) * w)
                    h_box = int(float(args[4]) * h)
                    xmin = x_center - (w_box // 2)
                    ymin = y_center - (h_box // 2)
                    xmax = x_center + (w_box // 2)
                    ymax = y_center + (h_box // 2)
                    cv2.imwrite(f"./crop_img/img{i}.jpg", img[ymin:ymax, xmin:xmax])
                    i += 1

def get_text(path):
    txt_path = f"{path.split('/')[-1]}.txt"
    with open(txt_path, "w") as f:
        i = 0
        for dir in os.listdir(path):
            old_path = os.path.join(path, dir)
            new_path = os.path.join(path, f"img{i}.jpg")

            text = dir.split('_')[0]
            line = f"img{i}.jpg@{text}\n"
            f.write(line)
            os.rename(old_path, new_path)
            i += 1

# get_text(path = "/home/minhtuan/Desktop/ASAP/JapaneseOCR/dataset")

print(len(os.listdir("dataset")))

with open("all.txt", "r") as f:
    print(len(f.readlines()))

with open("val.txt", "r") as f:
    print(len(f.readlines()))