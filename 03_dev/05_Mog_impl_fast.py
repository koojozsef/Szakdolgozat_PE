# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 10:50:37 2018

@author: koojo
"""

import cv2 as cv
import numpy as np
import math
import time
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
    - Implement P(x) and log the results
    - Evaluate omega value
    - Evaluate mue and sigma
    - Create B list
"""
#region ---- GLOBAL PARAMETERS ----
omega_g = (.5*np.ones((7))).astype('f')
mue_g = (1*np.ones((7,3))).astype('f')
sigma_g = (.3*np.ones((3,3))).astype('f')#
alpha_g = .6
#endregion

"""
Gaussian calculation 
    @x_p: image pixel (3,)
    @mue_p: expected value (3,)
    @sigma_p: covariance matrix (must be square and diagonal) (3,3)
    @n: ??? default 1
"""
def eta(x_p,mue_p,sigma_p,n=1):
    
    a=(x_p-mue_p)
    b=sigma_p**-1
    c=x_p-mue_p
    d= b@c
    e= a.T@d
    
    exponent = (-1/2) * e
    
    
    sigma_det= sigma_p[0][0]*sigma_p[1][1]*sigma_p[2][2]
    
    denominator = (2*math.pi)**(n/2) * sigma_det**(1/2) 
    
    return (1/denominator)*math.e**exponent

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
Probabitiy of a pixel
    @x_p: individual pixel 
"""
def P(x_p):
    eta_r=[]
    for mue_i in iter(mue_g):#iterate on gaussians
        eta_r.append(eta(x_p, mue_i, sigma_g))
        
    return (eta_r*omega_g)
    

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
        start_t = time.time()
        
        for component in np.ndindex(frame_r.shape[:2]):
            result.append(P(frame_r[component]))
        
        end_t = time.time()
        print(end_t-start_t)
        #fgmask_mix = fgmask_KNN & fgmask_MOG3
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
    T_x = np.arange(0,5,.01)
    result = []
    for x in iter(T_x):
        result.append(P(x))
    result_plot = np.squeeze(result)
    plt.plot(result_plot[:,0]) # plots the 1st gaussian for R G and B channel
    plt.show()
    
#endregion
