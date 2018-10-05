# Extract clibration matrix and distortion coefficients from Left and right videos
# Programm will not work if program detects more then 150 patterns.

import cv2
import os
import json
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from  math  import  sin ,  cos ,  tan ,  pi
from matplotlib.colors import ListedColormap, BoundaryNorm
import threading
import my_motion_detect1 as md
import myClassCalib as mc
from PIL import Image
from pprint import pprint
# tiff files
import math

def Calibration(video, yamlFiles):
    __mtL = mc.Calib(video[0])
    __mtR = mc.Calib(video[1])
    __sh = mc.Show()
    __mtL.findCorners(TypeOfPattern=1)
    __mtR.findCorners(TypeOfPattern=1)
    out = cv2.VideoWriter('CalibCyrcle100.avi',cv2.VideoWriter_fourcc(*'XVID'), 5, (__mtL.width*2, __mtL.height))
    while (1):
        if (__mtL.cap.isOpened) and (__mtR.cap.isOpened):
            retL, __mtL.frames = __mtL.cap.read()
            retR, __mtR.frames = __mtR.cap.read()
            if (retL == True) and (retR == True):
                print("----------------------------------------")
                #gray = cv2.cvtColor(__mtL.frames, cv2.COLOR_BGR2GRAY)
                retCL, circlesL = cv2.findCirclesGrid(__mtL.frames, (4, 11), flags=(cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING), blobDetector = __mtL.detector )
                #__mtL.frames = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                retCR, circlesR = cv2.findCirclesGrid(__mtR.frames, (4, 11), flags=(cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING), blobDetector = __mtR.detector )
                __mtL.fNum = round(__mtL.cap.get(cv2.CAP_PROP_POS_FRAMES))
                __mtR.fNum = round(__mtR.cap.get(cv2.CAP_PROP_POS_FRAMES))
                print("fNum: ", __mtL.fNum, " Left: ", retCL, " Right: ", retCR)
                if retCL == True:
                    __mtL.numGrid.append(__mtL.fNum)
                    __mtL.objpoints.append(__mtL.objp)
                    __mtL.imgpoints.append(circlesL)
                    __mtL.PatNum = __mtL.PatNum + 1
                    cv2.drawChessboardCorners(__mtL.frames, (4, 11), circlesL, retCL)
                    print("Circles found on image " + str(__mtL.fNum) + ".")
                if retCR == True:
                    __mtR.numGrid.append(__mtR.fNum)
                    __mtR.objpoints.append(__mtR.objp)
                    __mtR.imgpoints.append(circlesR)
                    __mtR.PatNum = __mtR.PatNum + 1
                    cv2.drawChessboardCorners(__mtR.frames, (4, 11), circlesR, retCR)
                    print("Circles found on image " + str(__mtR.fNum) + ".")
                __mtL.addTexttoFramePattern(__mtL.frames, (0, 255, 0))
                __mtR.addTexttoFramePattern(__mtR.frames, (0, 0, 255))
                __sh.show(__mtL.frames,__mtR.frames)
                C = cv2.hconcat([__mtL.frames, __mtR.frames])
                out.write(C)

                if __mtL.PatNum != []:
                    if int(__mtL.PatNum) == 150:
                        print("Left calibration")
                        __mtL.calibrate()
                        __mtL.saveToFile(yamlFiles[0])
                if __mtR.PatNum != []:
                    if int(__mtR.PatNum) == 150:
                        print("Right calibration")
                        __mtR.calibrate()
                        __mtR.saveToFile(yamlFiles[1])

        k = cv2.waitKey(30) & 0xff
        if k == 27:  # (ASCII value for ESC is 27 - exit program)
            __mtL.calibrate()
            __mtR.calibrate()
            __mtL.saveToFile(yamlFiles[0])
            __mtR.saveToFile(yamlFiles[1])
            out.release()
            print('ESC')
        elif k == 32:
            while (cv2.waitKey(1) == 'p'):
                out.release()
                print("pause")
    __mtR.calibrate()
    __mtR.saveToFile(yamlFiles[1])
    __mtL.calibrate()
    __mtL.saveToFile(yamlFiles[0])

    out.release()
    __mtL.cap.release()
    __mtR.cap.release()
    print("done")


def main():
    try:
        video = ['cam1_20180623_144000crop1.mp4', 'cam2_20180623_144506crop1.mp4']
        yamlFiles = ["calibCyrcleL200.yaml", "calibCyrcleR200.yaml"]
        Calibration(video, yamlFiles)

        return 0
    except:
        return 1


if __name__ == '__main__':
    main()