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

cap = cv.VideoCapture("D:\\joci\\EGYETEM\\_PE_MIK\\3_felev\\Szakdoga\\02_data\\01_vid\\square.mp4")
log_file = open("D:/joci/EGYETEM/_PE_MIK/3_felev/Szakdoga/opencvtest/log.txt", "w")

# region ---- CONFIGURATION ----
MANUAL_CONTROL = 0  # enable manual control if 1
RESIZE = 1  # enable resize if 1
CFG_SHOW_FRAMES = 1  # shows frames in windows
CFG_TEST = 0  # test
CFG_RUN = CFG_TEST == 0  # run
# endregion

# region ---- CODE ----
KEY_PRESSED = 1  # controls the frame steps
__HEIGHT__ = 400
__WIDTH__ = 600
__PIXELCOUNT__ = __HEIGHT__ * __WIDTH__
__MUE__ = 0
__SIGMA__ = 1
__OMEGA__ = 2

"""
TASKS:
    - [DONE] Implement background decider M()
    - [DONE] Evaluate omega value
    - Evaluate mue and sigma
    - Create B list
"""

# region ---- GLOBAL PARAMETERS ----
omega_g = (.5 * np.ones((__PIXELCOUNT__, 3, 7))).astype('f')
mue_g = (5 * np.ones((__PIXELCOUNT__, 3, 7))).astype(int)
sigma_g = (10 * np.ones((__PIXELCOUNT__, 3, 7))).astype(int)

"""distribution_g
    axis 0 :    0 - __PIXELCOUNT__  : pixel identifier
    axis 1 :            0 - 3       : R,G,B channels
    axis 2 :            0 - 7       : distribution identifier
    axis 3 :            0 - 3       : mue,sigma,omega parameters
"""
distribution_g = np.stack((mue_g, sigma_g, omega_g), axis=3)
alpha_g = .6
ro_g = .5


# endregion


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
    distance_sq = np.einsum('ijk,ijk->ijk', distance, distance)
    sigma_p_sq = np.einsum('ijk,ijk->ijk', sigma_p, sigma_p)
    a = (1 - ro_p) * sigma_p_sq
    b = ro_p * distance_sq.T
    c = np.einsum('ijk,ki->ijk', a + b, M_p)
    d = np.einsum('ijk,ki->ijk', sigma_p_sq, (1 - M_p))
    sigma_sq = c + d
    sigma_ret = np.sqrt(sigma_sq)
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
    b = np.einsum('ijk,ij->ijk', a, (ro_p * pixel_p))
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


def M(pixel_p, sigma_p):
    """
    M algorithm
        @pixel_p: 3 element array
        @sigma_p
    """
    sigma_avg = np.mean(sigma_p, axis=1).T

    a_min_b = mue_g.T - pixel_p.T
    b = np.sqrt(np.einsum('ijk,ijk->ik', a_min_b, a_min_b))
    a = b - sigma_avg
    a[a < 0] = 0
    a[a > 0] = 1

    return a.astype(bool)


while (CFG_RUN):
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
            frame_r = cv.resize(frame, (600, 400))
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
        result = M(long_frame, distribution_g[:, :, :, __SIGMA__])
        distribution_g[:, 0, :, __OMEGA__] = omega_update(distribution_g[:, 0, :, __OMEGA__], alpha_g, result)
        distribution_g[:, :, :, __MUE__] = mue_update(ro_g, long_frame, distribution_g[:, :, :, __MUE__], result)
        distribution_g[:, :, :, __SIGMA__] = sigma_updater(ro_g, long_frame, distribution_g[:, :, :, __MUE__],
                                                           distribution_g[:, :, :, __SIGMA__], result)

        # test
        sigma_g[:, :1, :2] = 20
        # test end
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
        # result_shape = np.shape(frame_r)
        # result = np.reshape(result,result_shape[:2])
        # endregion

        # region ---- exit on 'esc' key ----
        k = cv.waitKey(30) & 0xff
    if k == 27:
        print(f"exit on 'esc' key: k = {k}")
        break
        # endregion
# endregion
cap.release()
cv.destroyAllWindows()

# region ---- TEST ----
if (CFG_TEST):
    T_x = [np.arange(100), np.arange(100), np.arange(100)]
    T_x = np.swapaxes(T_x, 0, 1)
    result = []
    result_plot = np.squeeze(result)
    plt.plot(result_plot)  # plots the 1st gaussian
    plt.show()

# endregion
