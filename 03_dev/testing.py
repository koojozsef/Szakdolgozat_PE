import cv2 as cv
import numpy as np
import sklearn.metrics
import time as t


def captureImage(folderName, imageStringWithoutNumber, fileFormat, i):
    if True and isinstance(folderName, str) \
            and isinstance(imageStringWithoutNumber, str) \
            and isinstance(fileFormat, str):
        path = str(folderName).replace('\\', '/') + '/'
        genPath = str(path + imageStringWithoutNumber + "%04d" % i + fileFormat)
        image = cv.imread(genPath)
        return image


def main(bg_name, fg_name):

    folder_name = str(bg_name + "_" + fg_name)

    orig_folder = "D:\\joci\\projects\\Szakdoga_PE\\Szakdoga\\Dataset\\Generated\\database\\" + folder_name + "\\orig\\"
    mask_folder = "D:\\joci\\projects\\Szakdoga_PE\\Szakdoga\\Dataset\\Generated\\database\\" + folder_name + "\\gt\\"

    f = open("D:\\joci\\projects\\Szakdoga_PE\\Szakdoga\\Dataset\\Generated\\test_results\\" + folder_name +
             "_testlog.csv", "a")

    fn_mog2 = cv.createBackgroundSubtractorMOG2(detectShadows=0)
    fn_knn = cv.createBackgroundSubtractorKNN(detectShadows=0)
    fn_mog = cv.bgsegm.createBackgroundSubtractorMOG()

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

    for i in range(1000):
        # Get images
        orig_im = cv.imread(orig_folder + "%04d" % i + ".png")
        mask_im = cv.imread(mask_folder + "%04d" % i + ".png")

        if orig_im is None or mask_im is None:
            print("no more image")
            break

        # Apply algorithm
        mog2_t_start = t.process_time()
        mog2_res = fn_mog2.apply(orig_im)
        mog2_t_stop = t.process_time()
        mog2_t = mog2_t_stop - mog2_t_start

        knn_t_start = t.process_time()
        knn_res = fn_knn.apply(orig_im)
        knn_t_stop = t.process_time()
        knn_t = knn_t_stop - knn_t_start

        mog_t_start = t.process_time()
        mog_res = fn_mog.apply(orig_im)
        mog_t_stop = t.process_time()
        mog_t = mog_t_stop - mog_t_start

        # Apply morphological operations
        # mog2_res = cv.morphologyEx(mog2_res, cv.MORPH_OPEN, kernel)
        # knn_res = cv.morphologyEx(knn_res, cv.MORPH_OPEN, kernel)
        # mog_res = cv.morphologyEx(mog_res, cv.MORPH_OPEN, kernel)

        # Show result and original images
        """
        cv.imshow("original", orig_im)
        cv.imshow("mog2", mog2_res)
        cv.imshow("knn", knn_res)
        cv.imshow("mog", mog_res)
        cv.imshow("ground truth", mask_im)
        """

        # Convert to bool numpy arrays
        mask_im = np.array((mask_im[:, :, 0] / 255).astype(bool))
        mog2_res = np.array((mog2_res / 255).astype(bool))
        mog_res = np.array((mog_res / 255).astype(bool))
        mask_im_not = np.logical_not(mask_im)

        # Test evaluation:
        # 1. False positive
        x1 = np.sum(mask_im_not)
        fp_mog = np.sum(mask_im_not & mog_res) / x1 if (x1 != 0) else 0
        fp_mog2 = np.sum(mask_im_not & mog2_res) / x1 if (x1 != 0) else 0
        fp_knn = np.sum(mask_im_not & knn_res) / x1 if (x1 != 0) else 0

        # 2. False negative
        x2 = np.sum(mask_im)
        fneg_mog = (np.sum(mask_im & np.logical_not(mog_res)) / x2) if (x2 != 0) else 0
        fneg_mog2 = (np.sum(mask_im & np.logical_not(mog2_res)) / x2) if (x2 != 0) else 0
        fneg_knn = (np.sum(mask_im & np.logical_not(knn_res)) / x2) if (x2 != 0) else 0

        # 3. Cohen kappa coefficient
        y1 = mask_im.flatten()
        x3 = np.sum(y1)
        kappa_mog = sklearn.metrics.cohen_kappa_score(y1, mog_res.flatten()) if (x3 != 0) else 0
        kappa_mog2 = sklearn.metrics.cohen_kappa_score(y1, mog2_res.flatten()) if (x3 != 0) else 0
        kappa_knn = sklearn.metrics.cohen_kappa_score(y1, knn_res.flatten()) if (x3 != 0) else 0


        # Write results to output
        """
        print("False positive: ")
        print(f"mog \t {fp_mog}")
        print(f"mog2 \t {fp_mog2}")
        print(f"knn \t {fp_knn}")
        print(f"----------")
        print("False negative: ")
        print(f"mog \t {fneg_mog}")
        print(f"mog2 \t {fneg_mog2}")
        print(f"knn \t {fneg_knn}")
        print(f"----------")
        print("Kappa: ")
        print(f"mog \t {kappa_mog}")
        print(f"mog2 \t {kappa_mog2}")
        print(f"knn \t {kappa_knn}")
        print(f"=========================")
        """

        # To file:
        #         |                MOG                   |                MOG2                  |                KNN                   |
        # Frame n | False pos | False neg | Kappa | time | False pos | False neg | Kappa | time | False pos | False neg | Kappa | time |
        if i == 0:
            f.write("i,"
                    "MOG,MOG,MOG,MOG,"
                    "MOG2,MOG2,MOG2,MOG2,"
                    "KNN,KNN,KNN,KNN\n")
            f.write("Frame n,"
                    "False pos,False neg,Kappa,time,"
                    "False pos,False neg,Kappa,time,"
                    "False pos,False neg,Kappa,time\n")
        f.write(f"{i},{fp_mog},{fneg_mog},{kappa_mog},{mog_t}," +
                f"{fp_mog2},{fneg_mog2},{kappa_mog2},{mog2_t}," +
                f"{fp_knn},{fneg_knn},{kappa_knn},{knn_t}\n")


        # Use key to navigate
        # k = cv.waitKey() & 0xff

        #if k == 27:
        #    break
    f.close()
    print(folder_name + "  DONE")


if __name__ == "__main__":
    bg_name = ["city", "nature", "sky", "tree", "waterfall"]
    fg_name = ["butterfly", "tigger1", "tigger2", "tigger3", "walk1",
               "walk2", "walk3", "walk4", "raptor", "girl", "dance"]

    for bg in bg_name:
        for fg in fg_name:
            main(bg, fg)
