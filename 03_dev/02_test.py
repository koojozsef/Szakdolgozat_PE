# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 10:30:50 2018

@author: koojo
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
 
#def main():
path="D:\\joci\\EGYETEM\\_PE_MIK\\3_felev\\Szakdoga\\opencvtest\\src\\lena.jpg"
img = cv.imread(path)
img_hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
img_hsv[...,1] = 255
img_hsv[...,2] = 255
img_hsv = cv.cvtColor(img_hsv,cv.COLOR_HSV2BGR)

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

cv.imshow('hsv', img_hsv)
cv.imshow('Lena', img)
cv.waitKey(0)
cv.destroyAllWindows()

cap = cv.VideoCapture("D:\\joci\\EGYETEM\\_PE_MIK\\3_felev\\Szakdoga\\02_data\\01_vid\\square.mp4")

fgbg = cv.createBackgroundSubtractorKNN()

while(1):
    ret, frame = cap.read()

    frame_r = cv.resize(frame,(600,400))
    fgmask = fgbg.apply(frame_r)
    #mask = np.array(fgmask, fgmask, fgmask)
    #mask.transpose(2,0,1)
    #maskedframe = mask & frame_r

    cv.imshow('frame',fgmask)
    cv.imshow('orig_frame',frame_r)
    #cv.imshow('masked_frame',maskedframe)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()

#if __name__ == "__main__":
#    main()