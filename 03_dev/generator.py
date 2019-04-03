import cv2 as cv
import numpy as np
import os, errno

Threshold = 30


def captureImage(folderName, imageStringWithoutNumber, fileFormat, i):
    if True and isinstance(folderName, str) \
            and isinstance(imageStringWithoutNumber, str) \
            and isinstance(fileFormat, str):
        path = str(folderName).replace('\\', '/') + '/'
        genPath = str(path + imageStringWithoutNumber + "%04d" % i + fileFormat)
        image = cv.imread(genPath)
        return image


def procImage(fg_p, bg_p):
    fg_i = np.asarray(fg_p, int)
    bg_i = np.asarray(bg_p, int)
    ret = abs(fg_i - bg_i)
    ret[ret < Threshold] = 0
    ret[ret >= Threshold] = 1
    ret = np.sum(ret, axis=2)
    return ret * 255


def save_image(bg_path,fg_path,orig_dest):
    cap = cv.VideoCapture("D:\\joci\\projects\\Szakdoga_PE\\Szakdoga\\Dataset\\Generated\\greenScreen\\" + fg_path + ".mp4")
    cap_bg = cv.VideoCapture("D:\\joci\\projects\\Szakdoga_PE\\Szakdoga\\Dataset\\Generated\\background\\" + bg_path + ".mp4")
    dest_folder = "D:\\joci\\projects\\Szakdoga_PE\\Szakdoga\\Dataset\\Generated\\database\\" + orig_dest + "\\orig\\"
    mask_folder = "D:\\joci\\projects\\Szakdoga_PE\\Szakdoga\\Dataset\\Generated\\database\\" + orig_dest + "\\gt\\"
    try:
        os.makedirs(dest_folder)
        os.makedirs(mask_folder)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    rows = 800
    cols = 600
    ret = cap.set(3, rows)
    ret = cap.set(4, cols)

    thresh = 230
    blur = 3
    final = ""
    display = 1
    kernel = np.ones((5, 5), np.uint8)

    lower = np.array([0, 100, 0])
    upper = np.array([80, 255, 80])

    im_num = 0
    while (True):


        KEY_PRESSED = 0

        ret_b, background = cap_bg.read()
        ret, frame = cap.read()

        if ret == 0 or ret_b == 0:
            break

        frame = cv.resize(frame, (600, 400))
        background = cv.resize(background, (600, 400))
        img_fg = frame

        mask = cv.inRange(frame, lower, upper)
        frame[mask != 0] = [0, 0, 0]
        background[mask == 0] = [0, 0, 0]
        outputImage = frame + background
        # outputImage = np.where(frame == (0, 255, 0), background, frame)
        mask = 255 - mask

        cv.imwrite(dest_folder + "%04d" % im_num + ".png", outputImage)
        cv.imwrite(mask_folder + "%04d" % im_num + ".png", mask)

        # img_res = procImage(img_fg, img_bg)

        # cv.imshow("mask", np.uint8(img_res))
        im_num = im_num + 1

        print(dest_folder)
        if ret == 0 or ret_b == 0:
            break


def main():
    bg_name = ["city", "nature", "sky", "tree", "waterfall"]
    fg_name = ["butterfly", "tigger1", "tigger2", "tigger3", "walk1",
               "walk2", "walk3", "walk4", "raptor", "girl", "dance"]

    for bg in bg_name:
        for fg in fg_name:
            save_image(bg, fg, str(bg + "_" + fg))

if __name__ == "__main__":
    main()
