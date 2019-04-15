import os, os.path
from random import random
import numpy as np
import cv2 as cv

def get_test_data():
    bg_name = ["city", "nature", "sky", "tree", "waterfall"]
    fg_name_normal = ["butterfly", "walk1", "walk2", "walk3", "walk4"]

    ret_data = []

    for bg in bg_name:
        for fg in fg_name_normal:
            orig_dest = str(bg + "_" + fg)
            raw_folder = "D:\\joci\\projects\\Szakdoga_PE\\Szakdoga\\Dataset\\Generated\\database\\" + orig_dest + "\\orig\\"
            mask_folder = "D:\\joci\\projects\\Szakdoga_PE\\Szakdoga\\Dataset\\Generated\\database\\" + orig_dest + "\\gt\\"
            ret_data.append([raw_folder, mask_folder])

    return ret_data


def get_image_sequence(src_array, number_of_image):
    sequence = int(random() * src_array.shape[0])
    max_image = len(os.listdir(src_array[sequence, 0]))
    image_start = int(random() * max((max_image - number_of_image - 1), 0))
    ret = []
    for i in range(number_of_image):
        path = str(src_array[sequence, 0]) + "%04d" % (i + image_start) + ".png"
        ret.append(path)

    return ret


def main():

    t_data = np.asarray(get_test_data())

    a = np.zeros(t_data.shape[0])

    for i in range(t_data.shape[0]):
        a[i] = len(os.listdir(t_data[i, 0]))

    while True:
        for im in get_image_sequence(t_data, 30):
            image = cv.imread(im)
            cropped = image[200:300, 200:300]
            cv.imshow("image", cropped)
            cv.waitKey(25)


if __name__ == "__main__":
    main()
