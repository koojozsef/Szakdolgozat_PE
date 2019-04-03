import cv2 as cv
import numpy as np

Threshold = 30

def captureImage(folderName, imageStringWithoutNumber, fileFormat, i):
    if True and isinstance(folderName, str) \
            and isinstance(imageStringWithoutNumber, str) \
            and isinstance(fileFormat, str):

        path = str(folderName).replace('\\','/') + '/'
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
    return ret*255

def main():
    KEY_PRESSED = 1
    folder_name_bg = "D:\joci\projects\Szakdoga_PE\Szakdoga\Dataset\SBM\SBMnet_dataset\SBMnet_dataset\\basic\MPEG4_40\input"
    path_bg = str(folder_name_bg).replace('\\', '/') + '/'
    img_bg = cv.imread(path_bg + "in000000.jpg")

    folder_name_fg = "D:\joci\projects\Szakdoga_PE\Szakdoga\Dataset\SBM\SBMnet_dataset\SBMnet_dataset\\basic\MPEG4_40\input"
    im_num = 0
    while(True):

        if KEY_PRESSED == 0:
            k = cv.waitKey() & 0xff
            KEY_PRESSED = 1
            print(f"key pressed: {k}")
        else:
            KEY_PRESSED = 0
            img_fg = captureImage(folder_name_fg, "in00", ".jpg", im_num)
            im_num = im_num + 1

            cv.imshow("original", img_fg)

            img_res = procImage(img_fg, img_bg)

            cv.imshow("mask", np.uint8(img_res))

            k = cv.waitKey(30) & 0xff
        if k == 27:
            print(f"exit on 'esc' key: k = {k}")
            break


if __name__ == "__main__":
    main()
