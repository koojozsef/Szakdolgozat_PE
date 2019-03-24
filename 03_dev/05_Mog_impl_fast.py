# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 10:50:37 2018

@author: koojo
"""

"""
TASKS:
    - [DONE] Implement background decider M()
    - [DONE] Evaluate omega value
    - [DONE] Evaluate mue and sigma
    - Create B list
"""

import cv2 as cv
import numpy as np
import time
from matplotlib import pyplot as plt

# region ---- CONFIGURATION ----
VIDEO = 0
MANUAL_CONTROL = 0  # enable manual control if 1
RESIZE = 1  # enable resize if 1
CFG_SHOW_FRAMES = 1  # shows frames in windows
CFG_TEST = 0  # test
CFG_RUN = CFG_TEST == 0  # run
# endregion

# region ---- CODE ----
KEY_PRESSED = 1  # controls the frame steps
__HEIGHT__ = 120
__WIDTH__ = 160
__PIXELCOUNT__ = __HEIGHT__ * __WIDTH__
__MUE__ = 0
__SIGMA__ = 1
__OMEGA__ = 2
__DIST_N__ = 10
__DIST_BG__ = 6

if VIDEO == 1:
    cap = cv.VideoCapture("D:\\joci\\EGYETEM\\_PE_MIK\\3_felev\\Szakdoga\\02_data\\01_vid\\square.mp4")
log_file_path = "../02_data/03_logs/time_log.txt"
"""
TASKS:
    - [DONE] Implement background decider M()
    - [DONE] Evaluate omega value
    - [DONE] Evaluate mue and sigma
    - Create B list
"""

# region ---- GLOBAL PARAMETERS ----
omega_g = (1. * np.ones((__PIXELCOUNT__, 3, __DIST_N__))).astype('f')

mue_g = (1 * np.ones((__PIXELCOUNT__, 3, __DIST_N__))).astype(int)
for n in range(__DIST_N__):
    x = n * int(255 / (__DIST_N__-1))
    mue_g[:, :, n] = mue_g[:, :, n] * x

sigma_g = (150 * np.ones((__PIXELCOUNT__, 3, __DIST_N__))).astype(int)

"""distribution_g
    axis 0 :    0 - __PIXELCOUNT__  : pixel identifier
    axis 1 :            0 - 3       : R,G,B channels
    axis 2 :            0 - __DIST_N__       : distribution identifier
    axis 3 :            0 - 3       : mue,sigma,omega parameters
"""
distribution_g = np.stack((mue_g, sigma_g, omega_g), axis=3)

"""foreground_dist_g
    axis 0 :    0 - __PIXELCOUNT__  : pixel identifier
    axis 1 :            0 - 3       : R,G,B channels
    axis 2 :            0 - 1       : distribution identifier
    axis 3 :            0 - 3       : mue,sigma,omega parameters
"""
foreground_dist_g = np.zeros((__PIXELCOUNT__, 3, 1, 3))
alpha_g = .2
ro_g = .7
ro_g_fg = .5


# endregion

def captureImage(folderName, imageStringWithoutNumber, fileFormat, i):
    if True and isinstance(folderName, str) \
            and isinstance(imageStringWithoutNumber, str) \
            and isinstance(fileFormat, str):
        path = str(folderName).replace('\\', '/') + '/'
        genPath = str(path + imageStringWithoutNumber + "%04d" % i + fileFormat)
        image = cv.imread(genPath)
        return image


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


def omega_update(omega_p, alpha_p, M_p):
    """
    Omega updater method
        @omega_p: omega_g shall be this
        @alpha_p: alpha_g shall be this
        @M_p: Marks if distribution matches to current pixel (1 or 0)
            matrix with a size of omega_g
    """
    return (1 - alpha_p) * omega_p + alpha_p * M_p.T


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


imageId = 625
previous_background = np.ones((1, __PIXELCOUNT__))
while (CFG_RUN):
    if VIDEO == 0:
        frame = captureImage("D:\joci\projects\Szakdoga_PE\Szakdoga\Dataset\WallFlower\MovedObject",
                             "b0",
                             ".bmp",
                             imageId)
        imageId = imageId + 1
        ret = True
    else:
        ret, frame = cap.read()

    # region---- control frame on key ----
    if KEY_PRESSED == 0 and MANUAL_CONTROL == 1 and CFG_SHOW_FRAMES == 1:
        k = cv.waitKey() & 0xff
        KEY_PRESSED = 1
        print(f"key pressed: {k}")
    else:
        KEY_PRESSED = 0
        # endregion

        # region---- exit on empty frame ----
        if ret == False:
            print(f"exit on empty frame: ret = {ret}")
            break
        # endregion

        # region---- resize frame ----
        if RESIZE == True:
            frame_r = cv.resize(frame, (__WIDTH__, __HEIGHT__))
        else:
            frame_r = frame
        # endregion
        # region ---- show frames in window ----
        if CFG_SHOW_FRAMES == 1:
            cv.imshow('MOG frame', frame_r)
        # endregion

        # region---- apply algorithms ----
        start = time.time()
        result = []
        long_frame = np.reshape(frame_r, (__PIXELCOUNT__, 3))
        sigma_avg = distribution_g[:, :, :, __SIGMA__].sum(axis=1) / 3
        result = M(long_frame, sigma_avg, distribution_g[:, :, :, __MUE__])
        distribution_g[:, 0, :, __OMEGA__] = omega_update(distribution_g[:, 0, :, __OMEGA__], alpha_g, result)
        distribution_g[:, :, :, __MUE__] = mue_update(ro_g, long_frame, distribution_g[:, :, :, __MUE__], result)
        distribution_g[:, :, :, __SIGMA__] = sigma_updater(ro_g, long_frame, distribution_g[:, :, :, __MUE__],
                                                           distribution_g[:, :, :, __SIGMA__], result)

        omega_rec = 1 / distribution_g[:, 0, :, __OMEGA__]
        B_all = np.einsum('ij,ij->ij', omega_rec,
                          sigma_avg)  # stores the ordering value for every pixel and every distribution
        B_indx = np.argpartition(B_all, (__DIST_BG__))[:, -__DIST_BG__:]  # Returns the top __DIST_BG__ distributions index
        B_indx_min = np.argpartition(B_all, 1)[:, 0]  # Returns the lowest 1 distribution index
        # B_ext = [sigma_avg, B_all]
        # B_extArr = np.asarray(B_ext)
        # B_extArr = np.moveaxis(B_extArr, 0, -1)
        # B_Res = []
        # for distr in B_extArr:
        #     B_Res.append(sorted(distr, key=lambda x: x[1]))
        # B_sorted = np.asarray(B_Res)
        # B = B_sorted[:, :__DIST_BG__, :]

        sigma_top = np.zeros((__PIXELCOUNT__, __DIST_BG__))
        row_n = 0
        for row in sigma_avg:
            a = row[B_indx[row_n]]
            sigma_top[row_n] = a
            row_n = row_n + 1

        mue_top = np.zeros((__PIXELCOUNT__, 3, __DIST_BG__))
        row_n = 0
        for row in distribution_g:
            b = row[:, B_indx[row_n], __MUE__]
            mue_top[row_n] = b
            row_n = row_n + 1
        # check if pixel is part of B:
        background = M(long_frame, sigma_top, mue_top)

        # region foreground model
        background = np.sum(background, axis=0)
        background[background > 0] = 1
        background = background[np.newaxis, ...]
        new_foreground = (previous_background == 1) * (background == 0)
        previous_background = background
        # mue
        foreground_dist_g[..., 0, __MUE__] = (long_frame.T * new_foreground).T
        # sigma
        foreground_dist_g[..., 0, __SIGMA__] = (50 * np.ones((__PIXELCOUNT__, 3)).T * new_foreground).T

        # update parameters
        foreground_dist_g[..., __MUE__] = mue_update(ro_g_fg, long_frame, foreground_dist_g[..., __MUE__],
                                                     np.logical_not(background))
        foreground_dist_g[..., __SIGMA__] = sigma_updater(ro_g_fg, long_frame, foreground_dist_g[..., __MUE__],
                                                          foreground_dist_g[..., __SIGMA__], np.logical_not(background))

        # turn to background only when sigma is low
        low_sigmas = (foreground_dist_g[...].T * (foreground_dist_g[..., __SIGMA__] < 20).T).T

        foreground_dist_g = (foreground_dist_g.T * (foreground_dist_g[..., __SIGMA__] >= 20).T).T

        for row in range(__PIXELCOUNT__):
            if low_sigmas[row, :, 0, __MUE__].any() > 0:
                distribution_g[row, :, B_indx_min[row], __MUE__] = low_sigmas[row, :, 0, __MUE__]
                distribution_g[row, :, B_indx_min[row], __SIGMA__] = low_sigmas[row, :, 0, __SIGMA__]
                distribution_g[row, 0, B_indx_min[row], __OMEGA__] = 1.

        print(np.max(foreground_dist_g[:, 0, 0, __OMEGA__]))
        # plot for test
        fg_image = np.uint8(np.reshape(foreground_dist_g[:, :, 0, __OMEGA__]*255, (3, __HEIGHT__, __WIDTH__)))
        fg_image = np.reshape(fg_image, (__HEIGHT__, __WIDTH__, 3))
        cv.imshow("foreground", fg_image)
        # endregion foreground model

        background = np.reshape(background, (__HEIGHT__, __WIDTH__))

        rr = background * 1.0
        # rr = np.sum(rr,axis=0) # detect if any distribution is 1
        # rr = np.einsum('ijk->jki', rr)
        mueimg = mue_top[:, :, 0]
        mueimg_reshape = np.uint8(np.reshape(mueimg, (__HEIGHT__, __WIDTH__, 3)))

        rrr = np.uint8(rr * 255)

        cv.imshow("1.png", mueimg_reshape)
        cv.imshow("2.png", rrr)
        end = time.time()
        print(end - start)
        # result_shape = np.shape(frame_r)
        # result1 = np.reshape(result[0],result_shape[:2])
        # result1 = result1*1.0
        # cv.imshow('res',result1)
        # endregion

        # region ---- exit on 'esc' key ----
        k = cv.waitKey(30) & 0xff
    if k == 27:
        print(f"exit on 'esc' key: k = {k}")
        break
        # endregion
# endregion


# region ---- TEST ----
i = 0
while (CFG_TEST and i < 5):

    # TODO:
    #    - import video frames
    #    - flatten them
    #    - resize
    #    - redefine the storagematrix to new pixelcount
    #    - apply the algorithm
    #    - save runtime to file
    #    - repeate many time

    ret, frame = cap.read()
    if ret == False:
        print("Empty frame during test")

    start = 0
    stop = 0
    result = []
    pixel_count = int((np.shape(frame)[0] * np.shape(frame)[1]))
    long_frame = np.reshape(frame, (pixel_count, 3))
    meas_n = 60  # number of meas
    print("started")

    mog = cv.createBackgroundSubtractorMOG2()
    for meas_i in range(1, meas_n, 1):
        print("Loop:    ", meas_i)
        pixels = int(((pixel_count / 2) / meas_n) * meas_i)
        test_frame = long_frame[:pixels]
        # dist = distribution_g[:pixels,:,:,:]

        """
        #########################################
        # algorithm
        start = time.time()
        result = M(test_frame, dist[:, :, :, __SIGMA__], dist[:,:,:,__MUE__])
        stop = time.time()
        t1= stop-start

        start = time.time()
        dist[:, 0, :, __OMEGA__] = omega_update(dist[:, 0, :, __OMEGA__], alpha_g, result)
        stop = time.time()
        t2 = stop - start

        start = time.time()
        dist[:, :, :, __MUE__] = mue_update(ro_g, test_frame, dist[:, :, :, __MUE__], result)
        stop = time.time()
        t3 = stop - start

        start = time.time()
        dist[:, :, :, __SIGMA__] = sigma_updater(ro_g, test_frame, dist[:, :, :, __MUE__],
                                                           dist[:, :, :, __SIGMA__], result)
        stop = time.time()
        t4 = stop - start
        """
        start = time.time()
        mask = mog.apply(test_frame)
        stop = time.time()
        t1 = stop - start

        with open(log_file_path, 'a') as f:
            print("{};{};".format(t1, pixels), file=f)
    print("Finished")
    with open(log_file_path, 'a') as f:
        print("0;0;", file=f)
    print(i)
    i = i + 1
"""
    sigma_avg = distribution_g[:, :, :, __SIGMA__].sum(axis=1) / 3
    omega_rec = 1 / distribution_g[:, 0, :, __OMEGA__]
    B_all = np.einsum('ij,ij->ij', omega_rec, sigma_avg)
    B_ext = [sigma_avg, B_all]
    B_extArr = np.asarray(B_ext)
    B_extArr[:, :, :2] = 20
    B_extArr = np.moveaxis(B_extArr, 0, -1)
    B_Res = []
    for distr in B_extArr:
        B_Res.append(sorted(distr, key=lambda x: x[1]))

    B_sorted = np.asarray(B_Res)
    B = B_sorted[:, :4, :]

    # check if pixel is part of B:
    background = M(long_frame, B[:, :, 0])

    background = np.reshape(background, (7, 400, 600))

    rr = background[:1] * 1.0
    rrr = np.einsum('ijk->jki', rr)
    mueimg = mue_g[:, :, 0] / 255.0
    mueimg_reshape = np.reshape(mueimg, (400, 600, 3))

    cv.imshow("1.png", rrr)
    end = time.time()
    print(end - start)
"""

# endregion
if VIDEO == 1:
    cap.release()
cv.destroyAllWindows()
