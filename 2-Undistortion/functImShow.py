
""" This is the program for checking if file excist in upfolder
"""
import sys
import os
import functInColor
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time

def videoShow(WinName, frame):
    cv2.namedWindow(WinName, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WinName, cv2.WINDOW_NORMAL, cv2.WINDOW_NORMAL)
    cv2.imshow(WinName, frame)



def show_One(video, nW, nH, figureName):
    row, col, channels = video.shape
    print("row: ", row, "col: ", col)
    w, h = int(col / nW), int(row / nH)
    resizedVideos = cv2.resize(video, (w, h))
    cv2.imshow(figureName, resizedVideos)

def show_imagepoint_in2D(imagepoints,outputFigures,image, readFNum):
    imagepoints = np.array(imagepoints)

    N, _, _, _ = imagepoints.shape
    plt.figure('Cyrcular reference paterns', figsize=(20,15))

    plt.imshow(image)
    for i in range(N):
        X = imagepoints[i, :, :, 0]
        Y = imagepoints[i, :, :, 1]
        plt.scatter(X, Y)
        plt.plot(X,Y)
        plt.annotate( str(readFNum[i]),
                     xy=(imagepoints[i, 0, :, 0], imagepoints[i, 0, :, 1]), xycoords='data',
                     xytext=(20, 20), textcoords='offset points',
                      bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3),
                     arrowprops=dict(arrowstyle="->", shrinkB=10,
                                     connectionstyle="angle3", color = 'r'),
                     size=8)
    plt.title('Cyrcular reference paterns', style='normal', fontsize='8')
    S = outputFigures + '/' + 'impoints2D' + time.strftime("%Y%m%d-%H%M%S") +'.jpg'
    plt.savefig(S, bbox_inches='tight')

def show_imagepoint_in3D(objpoints):

    print('----------------')
    objpoints = np.array(objpoints)
    print(objpoints.shape)
    N, _, _ = objpoints.shape
    plt.figure('Cyrcular reference paterns in 3D')

    #plt.imshow(image)
    for i in range(N):
        X = objpoints[i, :, 0]
        Y = objpoints[i, :, 1]
        plt.scatter(X, Y)
        plt.plot(X,Y)
        plt.annotate( str(i),
                     xy=(objpoints[i, 0, 0], objpoints[i, 0, 1]), xycoords='data',
                     xytext=(20, 20), textcoords='offset points',
                      bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3),
                     arrowprops=dict(arrowstyle="->", shrinkB=10,
                                     connectionstyle="angle3", color = 'r'),
                     size=8)
    print(objpoints[0, :, 0])
    print(objpoints[0, :, 1])
    print(objpoints[0, :, 2])

    plt.title('Cyrcular reference paterns in 3D', style='normal', fontsize='8')
    #S = outputFigures + '/' + 'impoints2D' + time.strftime("%Y%m%d-%H%M%S") +'.jpg'
    #plt.savefig(S, bbox_inches='tight')


def show_imagepoint_in3Dtst(objpoints):

    print('----------------')
    objpoints = np.array(objpoints)
    print(objpoints.shape)
    N, _ = objpoints.shape
    plt.figure('Cyrcular reference paterns in 3D')
    ax1 = plt.subplot(111)
    #plt.imshow(image)
    X = objpoints[:, 0]
    Y = objpoints[:, 1]
    ax1.scatter(X, Y)
    ax1.plot(X,Y)
    ax1.annotate( str(1), xy=(objpoints[0, 0], objpoints[0, 1]), xycoords='data',
                xytext=(20, 20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3),
                arrowprops=dict(arrowstyle="->", shrinkB=10, connectionstyle="angle3", color = 'r'),
                size=8)
    ax1.minorticks_on()
    ax1.grid(which='major', linestyle='-', linewidth='0.5', color='blue')
    ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='green')
   # ax1.set_xlim(0, 1000)


    #ax1.set_ylim(0, 600)
    ax1.set_xlabel('X, mm')
    ax1.set_ylabel('Y, mm')

    print(objpoints[0, 0])
    print(objpoints[0, 1])
    print(objpoints[0, 2])

    plt.title('Cyrcular reference paterns in 3D_tst', style='normal', fontsize='8')
    #S = outputFigures + '/' + 'impoints2D' + time.strftime("%Y%m%d-%H%M%S") +'.jpg'
    #plt.savefig(S, bbox_inches='tight')
