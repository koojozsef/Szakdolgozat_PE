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
MANUAL_CONTROL = 1 # enable manual control if 1
RESIZE = 1 # enable resize if 1
CFG_SHOW_FRAMES = 1 # shows frames in windows
CFG_TEST = 0 # test
CFG_RUN = CFG_TEST == 0 # run
#endregion

#region ---- CODE ----
KEY_PRESSED = 1 #controls the frame steps

"""
TASKS:
    - Implement background decider M()
    - Evaluate omega value
    - Evaluate mue and sigma
    - Create B list
"""
#region ---- GLOBAL PARAMETERS ----
omega_g = (.5*np.ones((7))).astype('f')
mue_g = (1*np.ones((7,3))).astype(int)
sigma_g = (2*np.ones((3,3))).astype('f')#
alpha_g = .6
#endregion



"""
Omega updater method
    @omega_p: omega_g shall be this
    @alpha_p: alpha_g shall be this
    @M_p: Marks if distribution matches to current pixel (1 or 0)
        matrix with a size of omega_g
"""
def omega_update(omega_p,alpha_p,M_p):
    return (1-alpha_p)*omega_p + alpha_p*M_p


"""
M algorithm
    @pixel_p: 3 element array
"""
def M(pixel_p):
    return 1

while(CFG_RUN):
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
        #region ---- show frames in window ----
        if CFG_SHOW_FRAMES == 1:
            cv.imshow('MOG frame',frame_r)
        #endregion
        
        #region---- apply algorithms ----
        result = []
        for rows in iter(frame_r):
            for pixel in iter(rows):
                result.append(M(pixel))
        result_shape = np.shape(frame_r)
        result = np.reshape(result,result_shape[:2])
        #endregion
        
        

        #region ---- exit on 'esc' key ----
        k = cv.waitKey(30) & 0xff
    if k == 27:
        print(f"exit on 'esc' key: k = {k}")
        break
        #endregion
#endregion
cap.release()
cv.destroyAllWindows()

#region ---- TEST ----
if(CFG_TEST):
    T_x = [np.arange(100),np.arange(100),np.arange(100)]
    T_x = np.swapaxes(T_x,0,1)
    result = []
    result_plot = np.squeeze(result)
    plt.plot(result_plot) # plots the 1st gaussian
    plt.show()
    
#endregion
