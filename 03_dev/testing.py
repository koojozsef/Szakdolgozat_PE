import cv2 as cv
import numpy as np
import sklearn.metrics


def main():
    bg_name = ["city", "nature", "sky", "tree", "waterfall"]
    fg_name = ["butterfly", "tigger1", "tigger2", "tigger3", "walk1",
               "walk2", "walk3", "walk4", "raptor", "girl", "dance"]

    folder_name = str(bg_name[3] + "_" + fg_name[5])

    orig_folder = "D:\\joci\\projects\\Szakdoga_PE\\Szakdoga\\Dataset\\Generated\\database\\" + folder_name + "\\orig\\"
    mask_folder = "D:\\joci\\projects\\Szakdoga_PE\\Szakdoga\\Dataset\\Generated\\database\\" + folder_name + "\\gt\\"

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
        mog2_res = fn_mog2.apply(orig_im)
        knn_res = fn_knn.apply(orig_im)
        mog_res = fn_mog.apply(orig_im)

        # Apply morphological operations
        # mog2_res = cv.morphologyEx(mog2_res, cv.MORPH_OPEN, kernel)
        # knn_res = cv.morphologyEx(knn_res, cv.MORPH_OPEN, kernel)
        # mog_res = cv.morphologyEx(mog_res, cv.MORPH_OPEN, kernel)

        # Show result and original images
        cv.imshow("original", orig_im)
        cv.imshow("mog2", mog2_res)
        cv.imshow("knn", knn_res)
        cv.imshow("mog", mog_res)
        cv.imshow("ground truth", mask_im)

        # Convert to bool numpy arrays
        mask_im = np.array((mask_im[:, :, 0] / 255).astype(bool))
        mog2_res = np.array((mog2_res / 255).astype(bool))

        # Test evaluation:
        # 1. False positive
        fp_mog = np.sum(np.logical_not(mask_im) & mog_res) / np.sum(np.logical_not(mask_im))
        fp_mog2 = np.sum(np.logical_not(mask_im) & mog2_res) / np.sum(np.logical_not(mask_im))
        fp_knn = np.sum(np.logical_not(mask_im) & knn_res) / np.sum(np.logical_not(mask_im))

        # 2. False negative
        fneg_mog = np.sum(mask_im & np.logical_not(mog_res)) / np.sum(mask_im)
        fneg_mog2 = np.sum(mask_im & np.logical_not(mog2_res)) / np.sum(mask_im)
        fneg_knn = np.sum(mask_im & np.logical_not(knn_res)) / np.sum(mask_im)

        # 3. Cohen kappa coefficient
        y1 = mask_im.flatten()
        kappa_mog = sklearn.metrics.cohen_kappa_score(y1, mog_res.flatten())
        kappa_mog2 = sklearn.metrics.cohen_kappa_score(y1, mog2_res.flatten())
        kappa_knn = sklearn.metrics.cohen_kappa_score(y1, knn_res.flatten())


        # Write results to output
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

        # Use key to navigate
        k = cv.waitKey() & 0xff

        if k == 27:
            break


if __name__ == "__main__":
    main()
