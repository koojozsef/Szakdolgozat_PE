import cv2 as cv
import numpy as np
import sklearn.metrics
import time as t

import cv2 as cv
import numpy as np
import time
from matplotlib import pyplot as plt


class MixtureOfGaussian:

    def __init__(self, alpha=0.5, ro=0.75, ro_fg=0.5):
        self.alpha_g = alpha
        self.ro_g = ro
        self.ro_g_fg = ro_fg
        self._invoked_ = False
        self.__PIXELCOUNT__ = 0

    __MUE__ = 0
    __SIGMA__ = 1
    __OMEGA__ = 2
    __DIST_N__ = 20
    __DIST_BG__ = 4

    @staticmethod
    def captureImage(folderName, imageStringWithoutNumber, fileFormat, i):
        if True and isinstance(folderName, str) \
                and isinstance(imageStringWithoutNumber, str) \
                and isinstance(fileFormat, str):
            path = str(folderName).replace('\\', '/') + '/'
            genPath = str(path + imageStringWithoutNumber + "%04d" % i + fileFormat)
            image = cv.imread(genPath)
            return image
    __captureImage = captureImage

    @staticmethod
    def sigma_updater(ro_p, pixel_p, mue_p, sigma_p, M_p):
        """
        Sigma updater
            @ro_p
            @pixel_p
            @mue_p
            @sigma_p
            @M_p
        """
        distance = mue_p.T - pixel_p.T
        distance_sq = np.einsum('ijk,ijk->ik', distance, distance)
        sigma_p_sq = np.einsum('ijk,ijk->ik', sigma_p, sigma_p)

        a = (1 - ro_p) * sigma_p_sq
        b = ro_p * distance_sq.T
        c = np.einsum('ik,ki->ik', a + b, M_p)
        d = np.einsum('ik,ki->ik', sigma_p_sq, (1 - M_p))
        sigma_sq = c + d
        r = np.sqrt(sigma_sq)
        sigma_ret = np.stack((r, r, r), axis=1)
        return sigma_ret
    __sigma_updater = sigma_updater


    @staticmethod
    def mue_update(ro_p, pixel_p, mue_p, M_p):
        """
        Mue updater
            @ro_p: shall be ro_g
            @pixel_p: image matrix
            @mue_p: shall be mue_g
            @M_p:
        """
        a = (1 - ro_p) * mue_p
        aa = (ro_p * pixel_p)
        b = (a.T + aa.T).T  # addition instead of multiplication: b = np.einsum('ijk,ij->ijk', a, aa)
        mue = np.einsum('ijk,ki->ijk', b, M_p)
        # d= (1-M_p)*mue_p
        c = np.einsum('ij,jki->jki', (1 - M_p), mue_p)  # M_p*(a.T + b.T) + d
        result = c + mue
        return result
    __mue_update = mue_update


    @staticmethod
    def omega_update(omega_p, alpha_p, M_p):
        """
        Omega updater method
            @omega_p: omega_g shall be this
            @alpha_p: alpha_g shall be this
            @M_p: Marks if distribution matches to current pixel (1 or 0)
                matrix with a size of omega_g
        """
        return (1 - alpha_p) * omega_p + alpha_p * M_p.T
    __omega_update = omega_update

    @staticmethod
    def M(pixel_p, sigma_p, mue_p):
        """
        M algorithm
            @pixel_p: 3 element array
            @sigma_p: avarage of sigmas
            @mue_p
        """
        # sigma_avg = np.mean(sigma_p, axis=1).T

        a_min_b = mue_p.T - pixel_p.T
        b = (np.einsum('ijk,ijk->ik', a_min_b, a_min_b))
        a = b - (2.5 * np.einsum('ik,ik->ik', sigma_p.T, sigma_p.T))

        a[a > 0] = 0
        a[a < 0] = 1
        return a.astype(bool)
    __M = M

    def apply(self, frame):

        if frame is None:
            return False

        if not self._invoked_:
            self.__PIXELCOUNT__ = frame.shape[0] * frame.shape[1]
            self.__HEIGHT__ = frame.shape[0]
            self.__WIDTH__ = frame.shape[1]
            self.previous_background = np.ones((1, self.__PIXELCOUNT__))
            self.omega_g = (1. * np.ones((self.__PIXELCOUNT__, 3, self.__DIST_N__))).astype('f')

            self.mue_g = (1 * np.ones((self.__PIXELCOUNT__, 3, self.__DIST_N__))).astype(int)
            for n in range(self.__DIST_N__):
                x = n * int(255 / (self.__DIST_N__ - 1))
                self.mue_g[:, :, n] = self.mue_g[:, :, n] * x

                self.sigma_g = (10 * np.ones((self.__PIXELCOUNT__, 3, self.__DIST_N__))).astype(int)

            """distribution_g
                axis 0 :    0 - __PIXELCOUNT__  : pixel identifier
                axis 1 :            0 - 3       : R,G,B channels
                axis 2 :            0 - __DIST_N__       : distribution identifier
                axis 3 :            0 - 3       : mue,sigma,omega parameters
            """
            self.distribution_g = np.stack((self.mue_g, self.sigma_g, self.omega_g), axis=3)

            """foreground_dist_g
                axis 0 :    0 - __PIXELCOUNT__  : pixel identifier
                axis 1 :            0 - 3       : R,G,B channels
                axis 2 :            0 - 1       : distribution identifier
                axis 3 :            0 - 3       : mue,sigma,omega parameters
            """
            self.foreground_dist_g = np.zeros((self.__PIXELCOUNT__, 3, 1, 3))
            self._invoked_ = True

        # region---- apply algorithms ----
        result = []
        long_frame = np.reshape(frame, (self.__PIXELCOUNT__, 3))
        sigma_avg = self.distribution_g[:, :, :, self.__SIGMA__].sum(axis=1) / 3
        result = self.__M(long_frame, sigma_avg, self.distribution_g[:, :, :, self.__MUE__])
        self.distribution_g[:, 0, :, self.__OMEGA__] = self.__omega_update(self.distribution_g[:, 0, :, self.__OMEGA__],
                                                                           self.alpha_g,
                                                                           result)
        self.distribution_g[:, :, :, self.__MUE__] = self.__mue_update(self.ro_g, long_frame,
                                                                       self.distribution_g[:, :, :, self.__MUE__],
                                                                       result)
        self.distribution_g[:, :, :, self.__SIGMA__] = self.sigma_updater(self.ro_g,
                                                                          long_frame,
                                                                          self.distribution_g[:, :, :, self.__MUE__],
                                                                          self.distribution_g[:, :, :, self.__SIGMA__],
                                                                          result)

        omega_rec = 1 / self.distribution_g[:, 0, :, self.__OMEGA__]
        B_all = np.einsum('ij,ij->ij', omega_rec,
                          sigma_avg)  # stores the ordering value for every pixel and every distribution
        B_indx = np.argpartition(B_all, self.__DIST_BG__)[:, -self.__DIST_BG__:]  # Returns the top __DIST_BG__ distributions index
        B_indx_min = np.argpartition(B_all, 1)[:, 0]  # Returns the lowest 1 distribution index
        # B_ext = [sigma_avg, B_all]
        # B_extArr = np.asarray(B_ext)
        # B_extArr = np.moveaxis(B_extArr, 0, -1)
        # B_Res = []
        # for distr in B_extArr:
        #     B_Res.append(sorted(distr, key=lambda x: x[1]))
        # B_sorted = np.asarray(B_Res)
        # B = B_sorted[:, :__DIST_BG__, :]

        sigma_top = np.zeros((self.__PIXELCOUNT__, self.__DIST_BG__))
        row_n = 0
        for row in sigma_avg:
            a = row[B_indx[row_n]]
            sigma_top[row_n] = a
            row_n = row_n + 1

        mue_top = np.zeros((self.__PIXELCOUNT__, 3, self.__DIST_BG__))
        row_n = 0
        for row in self.distribution_g:
            b = row[:, B_indx[row_n], self.__MUE__]
            mue_top[row_n] = b
            row_n = row_n + 1
        # check if pixel is part of B:
        background = self.__M(long_frame, sigma_top, mue_top)

        # region foreground model
        background = np.sum(background, axis=0)
        background[background > 0] = 1
        background = background[np.newaxis, ...]
        new_foreground = (self.previous_background == 1) * (background == 0)
        self.previous_background = background
        # mue
        self.foreground_dist_g[..., 0, self.__MUE__] = (long_frame.T * new_foreground).T
        # sigma
        self.foreground_dist_g[..., 0, self.__SIGMA__] = (20 * np.ones((self.__PIXELCOUNT__, 3)).T * new_foreground).T

        # update parameters
        self.foreground_dist_g[..., self.__MUE__] = self.mue_update(self.ro_g_fg,
                                                                    long_frame,
                                                                    self.foreground_dist_g[..., self.__MUE__],
                                                                    np.logical_not(background))
        self.foreground_dist_g[..., self.__SIGMA__] = self.sigma_updater(self.ro_g_fg,
                                                                         long_frame,
                                                                         self.foreground_dist_g[..., self.__MUE__],
                                                                         self.foreground_dist_g[..., self.__SIGMA__],
                                                                         np.logical_not(background))

        # turn to background only when sigma is low
        low_sigmas = (self.foreground_dist_g[...].T * (self.foreground_dist_g[..., self.__SIGMA__] < 5).T).T

        self.foreground_dist_g = (self.foreground_dist_g.T * (self.foreground_dist_g[..., self.__SIGMA__] >= 5).T).T

        for row in range(self.__PIXELCOUNT__):
            if low_sigmas[row, :, 0, self.__MUE__].any() > 0:
                self.distribution_g[row, :, B_indx_min[row], self.__MUE__] = low_sigmas[row, :, 0, self.__MUE__]
                self.distribution_g[row, :, B_indx_min[row], self.__SIGMA__] = low_sigmas[row, :, 0, self.__SIGMA__]
                self.distribution_g[row, 0, B_indx_min[row], self.__OMEGA__] = 1.

        background = np.logical_not(np.reshape(background, (self.__HEIGHT__, self.__WIDTH__)))
        rr = background * 1.0
        rrr = np.uint8(rr * 255)
        return rrr


def captureImage(folderName, imageStringWithoutNumber, fileFormat, i):
    if True and isinstance(folderName, str) \
            and isinstance(imageStringWithoutNumber, str) \
            and isinstance(fileFormat, str):
        path = str(folderName).replace('\\', '/') + '/'
        genPath = str(path + imageStringWithoutNumber + "%04d" % i + fileFormat)
        image = cv.imread(genPath)
        image = cv.resize(image, (300, 200))
        return image


def main(bg_name, fg_name):

    folder_name = str(bg_name + "_" + fg_name)

    orig_folder = "D:\\joci\\projects\\Szakdoga_PE\\Szakdoga\\Dataset\\Generated\\database\\" + folder_name + "\\orig\\"
    mask_folder = "D:\\joci\\projects\\Szakdoga_PE\\Szakdoga\\Dataset\\Generated\\database\\" + folder_name + "\\gt\\"

    f = open("D:\\joci\\projects\\Szakdoga_PE\\Szakdoga\\Dataset\\Generated\\test_results\\00_" + folder_name +
             "_testlog.csv", "a")

    fn_mog2 = cv.createBackgroundSubtractorMOG2(detectShadows=0)
    fn_knn = cv.createBackgroundSubtractorKNN(detectShadows=0)
    fn_mog = cv.bgsegm.createBackgroundSubtractorMOG()
    fn_mymog = MixtureOfGaussian(alpha=0.5, ro=0.75, ro_fg=0.5)


    for i in range(40):
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

        mymog_t_start = t.process_time()
        mymog_res = fn_mymog.apply(orig_im)
        mymog_t_stop = t.process_time()
        mymog_t = mymog_t_stop - mymog_t_start

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
        knn_res = np.array((knn_res / 255).astype(bool))
        mymog_res = np.array((mymog_res / 255).astype(bool))
        mask_im_not = np.logical_not(mask_im)

        # Test evaluation:
        # 1. False positive
        x1 = np.sum(mask_im_not)
        fp_mog = np.sum(mask_im_not & mog_res) / x1 if (x1 != 0) else 0
        fp_mog2 = np.sum(mask_im_not & mog2_res) / x1 if (x1 != 0) else 0
        fp_knn = np.sum(mask_im_not & knn_res) / x1 if (x1 != 0) else 0
        fp_mymog = np.sum(mask_im_not & mymog_res) / x1 if (x1 != 0) else 0

        # 2. False negative
        x2 = np.sum(mask_im)
        fneg_mog = (np.sum(mask_im & np.logical_not(mog_res)) / x2) if (x2 != 0) else 0
        fneg_mog2 = (np.sum(mask_im & np.logical_not(mog2_res)) / x2) if (x2 != 0) else 0
        fneg_knn = (np.sum(mask_im & np.logical_not(knn_res)) / x2) if (x2 != 0) else 0
        fneg_mymog = (np.sum(mask_im & np.logical_not(mymog_res)) / x2) if (x2 != 0) else 0

        # 3. Cohen kappa coefficient
        y1 = mask_im.flatten()
        x3 = np.sum(y1)
        kappa_mog = sklearn.metrics.cohen_kappa_score(y1, mog_res.flatten()) if (x3 != 0) else 0
        kappa_mog2 = sklearn.metrics.cohen_kappa_score(y1, mog2_res.flatten()) if (x3 != 0) else 0
        kappa_knn = sklearn.metrics.cohen_kappa_score(y1, knn_res.flatten()) if (x3 != 0) else 0
        kappa_mymog = sklearn.metrics.cohen_kappa_score(y1, mymog_res.flatten()) if (x3 != 0) else 0


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
        #         |                MOG                   |                MOG2                  |                KNN                   |              MYMOG                   |
        # Frame n | False pos | False neg | Kappa | time | False pos | False neg | Kappa | time | False pos | False neg | Kappa | time | False pos | False neg | Kappa | time |
        if i == 0:
            f.write("i,"
                    "MOG,MOG,MOG,MOG,"
                    "MOG2,MOG2,MOG2,MOG2,"
                    "KNN,KNN,KNN,KNN,"
                    "MYMOG,MYMOG,MYMOG,MYMOG\n")
            f.write("Frame n,"
                    "False pos,False neg,Kappa,time,"
                    "False pos,False neg,Kappa,time,"
                    "False pos,False neg,Kappa,time,"
                    "False pos,False neg,Kappa,time\n")
        f.write(f"{i},{fp_mog},{fneg_mog},{kappa_mog},{mog_t}," +
                f"{fp_mog2},{fneg_mog2},{kappa_mog2},{mog2_t}," +
                f"{fp_knn},{fneg_knn},{kappa_knn},{knn_t}," +
                f"{fp_mymog},{fneg_mymog},{kappa_mymog},{mymog_t}\n")


        # Use key to navigate
        # k = cv.waitKey() & 0xff

        #if k == 27:
        #    break
    f.close()
    print(folder_name + "  DONE")


if __name__ == "__main__":
    bg_name = ["nature", "tree", "waterfall"]
    fg_name = ["butterfly", "walk1", "walk3", "walk4"]

    for bg in bg_name:
        for fg in fg_name:
            main(bg, fg)
