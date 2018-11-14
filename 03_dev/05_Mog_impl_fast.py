# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 10:50:37 2018

@author: koojo
"""

import cv2 as cv
import numpy as np
import time
from matplotlib import pyplot as plt

cap = cv.VideoCapture("D:\\joci\\EGYETEM\\_PE_MIK\\3_felev\\Szakdoga\\02_data\\01_vid\\square.mp4")
log_file = open("D:/joci/EGYETEM/_PE_MIK/3_felev/Szakdoga/opencvtest/log.txt","w")

#region ---- CONFIGURATION ----
MANUAL_CONTROL = 0 # enable manual control if 1
RESIZE = 1 # enable resize if 1
CFG_SHOW_FRAMES = 1 # shows frames in windows
CFG_TEST = 0 # test
CFG_RUN = CFG_TEST == 0 # run
#endregion

#region ---- CODE ----
KEY_PRESSED = 1 #controls the frame steps
__HEIGHT__ = 400
__WIDTH__ = 600
__PIXELCOUNT__ = __HEIGHT__*__WIDTH__

"""
TASKS:
    - [DONE] Implement background decider M()
    - [DONE] Evaluate omega value
    - Evaluate mue and sigma
    - Create B list
"""
#region ---- GLOBAL PARAMETERS ----
omega_g = (.5*np.ones((7,__PIXELCOUNT__))).astype('f')
mue_g = (5*np.ones((__PIXELCOUNT__,3,7))).astype(int)
sigma_g = (10*np.ones((__PIXELCOUNT__,3,7))).astype(int)
alpha_g = .6
ro_g = .5
#endregion

"""
Sigma updater
    @ro_p
    @pixel_p
    @mue_p
    @sigma_p
    @M_p
"""
def sigma_updater(ro_p, pixel_p, mue_p, sigma_p, M_p):
    distance = mue_p.T - pixel_p.T
    distance_sq = np.einsum('ijk,ijk->ijk',distance,distance)
    sigma_p_sq = np.einsum('ijk,ijk->ijk',sigma_p,sigma_p) 
    a = (1-ro_p)*sigma_p_sq
    b = ro_p*distance_sq.T
    c = np.einsum('ijk,ki->ijk',a+b,M_p)
    d = np.einsum('ijk,ki->ijk',sigma_p_sq,(1-M_p))
    sigma_sq = c + d
    sigma_ret = np.sqrt(sigma_sq)
    return sigma_ret

"""
Mue updater
    @ro_p: shall be ro_g
    @pixel_p: image matrix
    @mue_p: shall be mue_g
    @M_p: 
"""
def mue_update(ro_p, pixel_p, mue_p,M_p):
    a = (1-ro_p)*mue_p
    b = np.einsum('ijk,ij->ijk',a,(ro_p*pixel_p))
    mue= np.einsum('ijk,ki->ijk',b,M_p)
    #d= (1-M_p)*mue_p
    c= np.einsum('ij,jki->jki',(1-M_p),mue_p)#M_p*(a.T + b.T) + d
    result = c+mue
    return result


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
    sigma_avg = np.mean(sigma_g,axis=1).T
    
    a_min_b = mue_g.T - pixel_p.T
    b = np.sqrt(np.einsum('ijk,ijk->ik', a_min_b, a_min_b))
    a = b - sigma_avg
    a[a<0] = 0
    a[a>0] = 1
    
    return a.astype(bool)

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
        start = time.time()
        result = []
        long_frame = np.reshape(frame_r,(__PIXELCOUNT__,3))
        result = M(long_frame)
        omega_g = omega_update(omega_g,alpha_g,result)
        mue_g = mue_update(ro_g,long_frame,mue_g,result)
        sigma_g = sigma_updater(ro_g,long_frame,mue_g,sigma_g,result)
        
        result = np.reshape(result,(7,400,600))
        
        rr = result[:1]*1.0
        rrr = np.einsum('ijk->jki',rr)
        mueimg = mue_g[:,:,0]/255.0
        mueimg_reshape= np.reshape(mueimg,(400,600,3))
        
        cv.imshow("1.png",mueimg_reshape)
        end = time.time()
        print(end-start)
        #result_shape = np.shape(frame_r)
        #result = np.reshape(result,result_shape[:2])
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
