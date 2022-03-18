from enum import Flag
import cv2
from cv2 import Laplacian
from markupsafe import re
import numpy as np
import matplotlib.pyplot as plt
import itertools

# Getting the local video file
cap = cv2.VideoCapture('G:\Projects\AI Arts\Project 01\movie0.mp4')


while True:
    ret, frame = cap.read()

    # Different effects
    laplacian = cv2.Laplacian( frame, cv2.CV_64F)
    sobelx = cv2.Sobel( frame, cv2.CV_64F, 1, 0, ksize=3 )
    sobely = cv2.Sobel( frame, cv2.CV_64F, 0, 1, ksize=5 )
    edges = cv2.Canny( frame, 200, 200 )

    # Different effects 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([30,150,50])
    upper_red = np.array([255,255,180])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Color changes and adding laplacian effect
    # Ref - https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html
    grayVideo = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    laplacian2 = cv2.Laplacian( grayVideo, cv2.CV_64F)

    tiktoktEffect = cv2.cvtColor(frame, cv2.COLOR_RGB2BGRA) #good x1
    tiktoktEffect2 = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS) #good x1
    tiktoktEffect3 = cv2.cvtColor(frame, cv2.COLOR_HLS2BGR) #good x3 - trippy

    # Accesing pixels
    px = grayVideo[50,50]

    # x start value -> x end value, y start value -> y end value, red value
    # x start value -> x end value, y start value -> y end value, green value
    # x start value -> x end value, y start value -> y end value, blue value 
    grayVideo[50:175, 50:175, 0] = 225
    grayVideo[50:175, 50:175, 1] = 0
    grayVideo[50:175, 50:175, 2] = 0
    print(px)

    # Displaying the retouched video
    if ( ret == True ):
        #cv2.imshow('original', grayVideo)
        #cv2.imshow('laplacian', laplacian)
        cv2.imshow('tiktok', tiktoktEffect)

        #cv2.imshow('sobelx', sobelx)
        #cv2.imshow('sobely', sobely)
        #cv2.imshow('Edges', edges)
        #cv2.imshow('Mask',mask)
    
    # Adding a key to close the window
    if ( cv2.waitKey(25) == ord('q') ):
        break

cap.release()
cv2.destroyAllWindows()
 