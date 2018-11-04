# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 10:50:37 2018

@author: koojo
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

cap = cv.VideoCapture("D:\\joci\\EGYETEM\\_PE_MIK\\3_felev\\Szakdoga\\02_data\\01_vid\\square.mp4")
log_file = open("D:/joci/EGYETEM/_PE_MIK/3_felev/Szakdoga/opencvtest/log.txt","w")

#region ---- CONFIGURATION ----
MANUAL_CONTROL = 0 # enable manual control if 1
RESIZE = 0 # enable resize if 1
CFG_SHOW_FRAMES = 1 # shows frames in windows when 'True'
CFG_LOG = False # enable logging to file when 'True'
#endregion

#region ---- CODE ----
KEY_PRESSED = 1 #controls the frame steps

fn_KNN = cv.createBackgroundSubtractorKNN()
fn_MOG2 = cv.createBackgroundSubtractorMOG2()

log_knn = [0]
log_mog = [0]
frame_r2 = np.zeros((2160,3840,3)).astype(np.uint8)

while(1):
    ret, frame = cap.read()
    
    #region---- control frame on key ----
    if  KEY_PRESSED == 0 and MANUAL_CONTROL == 1 and CFG_SHOW_FRAMES == 1:
        k = cv.waitKey() & 0xff
        KEY_PRESSED = 1
        print(f"key pressed: {k}")
    else:
        KEY_PRESSED = 0
    #endregion

        #region---- exit on empty frame ----
        if ret == False:
            print(f"exit on empty frame: ret = {ret}")
            break
        #endregion

        #region---- resize frame ----
        if RESIZE == True:
            frame_r = cv.resize(frame,(600,400))
        else:
            frame_r = frame
        #endregion

        #region---- apply algorithms ----
        fgmask_KNN = fn_KNN.apply(frame_r)
        fgmask_MOG3 = fn_MOG2.apply(frame_r)
        
        fgmask_diff = cv.norm(frame_r,frame_r2,cv.NORM_L1)
        frame_r2 = frame_r
        #fgmask_mix = fgmask_KNN & fgmask_MOG3
        #endregion

        #region ---- show frames in window ----
        if CFG_SHOW_FRAMES == 1:
            cv.imshow('KNN frame',fgmask_KNN)
            cv.imshow('MOG3 frame',fgmask_MOG3)
            #cv.imshow('mixed frame',fgmask_mix)
            cv.imshow('orig_frame',frame_r)
            print(fgmask_diff)
        #endregion

        #region---- log ----
        if CFG_LOG == True:
            #fgmask_mix[fgmask_mix > 0] = 1
            fgmask_KNN[fgmask_KNN > 0] = 1
            fgmask_MOG3[fgmask_MOG3 > 0] = 1

            #frame_sum_mix = sum(sum(fgmask_mix))
            frame_sum_knn = sum(sum(fgmask_KNN))
            frame_sum_mog3 = sum(sum(fgmask_MOG3))

            log_knn.extend([frame_sum_knn])
            log_mog.extend([frame_sum_mog3])

            log_file.write(f"{frame_sum_knn},{frame_sum_mog3}\n")
            print(f"knn = {frame_sum_knn},  mog3 = {frame_sum_mog3}")
        #endregion

        #region ---- exit on 'esc' key ----
        k = cv.waitKey(30) & 0xff
    if k == 27:
        print(f"exit on 'esc' key: k = {k}")
        break
        #endregion
#endregion
log_file.close()
cap.release()
cv.destroyAllWindows()
