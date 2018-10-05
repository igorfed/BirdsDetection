# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 15:22:42 2018

@author: igofed
"""
import json
import cv2
import tkinter
import numpy as np
import matplotlib.pyplot as plt
import threading
import yaml

from  math  import  sin ,  cos ,  tan ,  pi

class POI(object):

    def __init__(self):
        self.frame = []
        self.x = []
        self.y = []
        self.value = []
        self.id = []
        self.dist = []
        self.X3 = []
        self.Y3 = []
        self.Z3 = []


class Calib(object):
    def __init__(self, fileName):
        self.pSize = 3.45E-6
        self.pattern_size = (9, 6)
        self.objpoints = []
        self.imgpoints = []
        self.calc_timestamps = [0.0]
        self.calibration = {}
        self.fileName = fileName
        self.fNum = 0
        self.PatNum = 0
        self.objpoints = []
        self.imgpoints = []
        self.mError = 0
        self.numGrid = []
        self.VideoCap()
        self.mean_error = 0
        self.detector = self.BlobDetector()
    def BlobDetector(self):
        self.params = cv2.SimpleBlobDetector_Params()
        self.params.filterByArea = True
        self.params.minArea = 100
        self.params.maxArea = 20000
        self.params.filterByCircularity = True
        self.params.minCircularity = 0.2
        self.params.filterByInertia = True
        self.params.minInertiaRatio = 0.1
        # blob detection only works with "uint8" images.

        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3:
            print('SimpleBlobDetector')
            return cv2.SimpleBlobDetector(self.params)

        else:
            print('SimpleBlobDetector_create')
            return cv2.SimpleBlobDetector_create(self.params)

    def VideoCap(self):
        self.cap = cv2.VideoCapture(self.fileName)
        if self.cap.isOpened() == False:
            print("Error opening video stream or file")
        else:
            print("Video :", self.fileName, " is opened")
            self.n_frame = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.frames = np.zeros((self.height, self.width, 3), 'uint8')
            print("nframes: ", self.n_frame, " fps: ", " [ ",self.width, " x ", self.height, "]")


    def findCorners(self, TypeOfPattern = 1):
        params = []
        def objParams(TypeOfPattern):
            print (TypeOfPattern)
            if TypeOfPattern == 1:
                param1 = np.zeros((4 * 11, 3), np.float32)
                param2 = np.mgrid[0:4,0:11].T.reshape(-1,2)
                print("Cyrcle")
            elif TypeOfPattern == 0:
                param1 = np.zeros((6 * 9, 3), np.float32)
                param2 = np.mgrid[0:9,0:6].T.reshape(-1,2)
                print("Checkerboard")
            else:
                param1 = np.zeros((4 * 11, 3), np.float32)
                param2 = np.mgrid[0:4,0:11].T.reshape(-1,2)
                print ("Type of pattern unrecognized")
            params.append(param1)
            params.append(param2)
        objParams(TypeOfPattern)
        self.objp = params[0]
        self.objp[:, :2] = params[1]
        objParams(TypeOfPattern)

    def calibrate(self):
        print ("start calibrateCamera")
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints,(self.width, self.height), None, None)
        print ("end calibrateCamera")
        self.error_in_frame = []
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(self.objpoints[i], self.rvecs[i], self.tvecs[i], self.mtx, self.dist)
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            self.mean_error += error
            self.error_in_frame.append(error)
        self.mean_error =self.mean_error / len(self.objpoints)
        print("total error: ", self.mean_error)

        print (type(self.mtx),"mtx: ", self.mtx)
        print('dist', self.dist)

    def addTexttoFramePattern(self, frame, color):
        S = "Pattern detected: " +  str(int(self.PatNum))
        cv2.putText(frame, str(int(self.n_frame)), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4, 8)
        cv2.putText(frame, str(self.fNum), (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4, 8)
        cv2.putText(frame, S, (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4, 8)

    def addTexttoFrame(self, frame, color):
        cv2.putText(frame, str(int(self.n_frame)), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4, 8)
        cv2.putText(frame, str(self.fNum), (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4, 8)

    def saveToFile(self, yamlFiles):
        f = cv2.FileStorage (yamlFiles, cv2.FILE_STORAGE_WRITE)
        f.write("mtx", self.mtx)
        f.write("dist", self.dist)
        f.write("rms", self.ret)
        f.write("mean_error", np.array(self.mean_error))
        f.write("frame", np.array(self.numGrid))
        f.write("error_in_frame", np.array(self.error_in_frame))
        f.write("rvecs", np.array(self.rvecs))
        f.write("tvecs", np.array(self.tvecs))

        print (np.array(self.numGrid))
        print (np.array(self.error_in_frame))

        #f.write("Frames", self.numGrid)
        #f.release()
        print ("Files saved")

def ReadFromFile(yamlFiles):
    print ("Read from: ", yamlFiles)
    fs = cv2.FileStorage(yamlFiles, cv2.FILE_STORAGE_READ)
    Intrinsic = fs.getNode("mtx")
    Distortion = fs.getNode("dist")
    ERROR = fs.getNode("error_in_frame")
    FRAME = fs.getNode("frame")
    K = Intrinsic.mat()
    D = Distortion.mat()
    ER = ERROR.mat()
    frame = FRAME.mat()
    print ("mtx:", ":\n", K)
    print ("dist:\n", D)
    print("dist:\n", ER)
    print("FRAME:\n", frame)
    #rotMatrix = fs.getNode("dist")
    return K, D, ER, frame

def Read2DFile(yamlFiles):
    print ("Read from: ", yamlFiles)
    fs = cv2.FileStorage(yamlFiles, cv2.FILE_STORAGE_READ)
    X = fs.getNode("X")
    Y = fs.getNode("Y")
    x = X.mat()
    x = np.array(x)
    y = Y.mat()
    y = np.array(y)
    print ("x:", x)
    print ("y:", y)
    return x, y




class Show(object):
    def __init__(self):
        self.N = 4
        self.figureName = "Windows"
    def show(self,video1, video2):
        concatVideo = cv2.hconcat([video1,video2])
        row, col, channels = concatVideo.shape
        w, h = int(col / self.N), int(row / self.N)
        resizedVideos = cv2.resize(concatVideo, (w, h))
        cv2.imshow(self.figureName, resizedVideos)


def OpenJSON(JsonFILE):
    Multi_Point = []
    Longitude = []
    Latitude = []
    Height = []
    X = []
    Y = []
    with open(JsonFILE, 'r') as json_file:
        data = json.load(json_file)
        i = 0
        for p in data['features']:
            Multi_Point.append(p['geometry']['coordinates'])
            i = i + 1

    numrows = len(Multi_Point)
    size = np.shape(Multi_Point)
    r = 0
    col = []
    grey = 0
    for row in range(size[0]):
        Longitude.append(Multi_Point[row][0])
        Latitude.append(Multi_Point[row][1])
        Height.append(Multi_Point[row][2])
        N, W = lattoXY(Longitude[row], Latitude[row])
        X.append(N)
        Y.append(W)
    print (X)
    print (Y)
    print (Height)

    return Longitude, Latitude, Height, X, Y


def lattoXY(dLon, dLat):
    zone  =  int ( dLon / 6.0 + 1 )
    a  =  6378245.0           # Large (equatorial) semiaxis
    b  =  6356863.019         # Small (polar) semiaxis
    e2  =  ( a ** 2 - b ** 2 ) / a ** 2   # Eccentricity
    n  =  ( a - b ) / ( a + b )         # The flatness
    # Parameters of the Gauss-Krueger zone
    F  =  1.0                    # Scale factor
    Lat0  =  0.0                 # Parallel (in radians)
    Lon0  =  ( zone * 6 - 3 ) * pi / 180   # Central meridian (in radians)
    N0  =  0.0                   # Conditional north offset for initial parallel
    E0  =  zone * 1e6 + 500000.0     # Nominal offset for east central meridian
    # Translation of latitude and longitude into radians
    Lat  =  dLat * pi / 180.0
    Lon  =  dLon * pi / 180.0
    # Calculation of variables for the transformation
    v  =  a * F * ( 1 - e2 * ( sin ( Lat ) ** 2 )) ** - 0.5
    p  =  a * F * ( 1 - e2 ) * ( 1 - e2 * ( sin ( Lat ) ** 2 )) ** - 1.5
    n2  =  v / p - 1
    M1  =  ( 1 + n + 5.0 / 4.0 * n ** 2 + 5.0 / 4.0 * n ** 3 ) * ( Lat - Lat0 )
    M2  =  ( 3 * n + 3 * n ** 2 + 21.0 / 8.0 * n ** 3 ) * sin ( Lat - Lat0 ) * cos( Lat + Lat0 )
    M3  =  ( 15.0 / 8.0 * n ** 2 + 15.0 / 8.0 * n ** 3 ) * sin ( 2 * ( Lat - Lat0 )) * cos ( 2 * ( Lat + Lat0 ))
    M4  =  35.0 / 24.0 * n ** 3 * sin ( 3 *( Lat - Lat0 )) * cos ( 3 * ( Lat + Lat0 ))
    M  =  b * F * ( M1 - M2 + M3 - M4 )
    I  =  M + N0
    II  =  v / 2 * sin ( Lat ) * cos ( Lat )
    III  =  v / 24 * sin (Lat ) * ( cos ( Lat )) ** 3 * ( 5 - ( tan ( Lat ) ** 2 ) + 9 * n2 )
    IIIA  =  v / 720 * sin ( Lat ) * ( cos ( Lat ) ** 5 ) * ( 61 - 58 * ( tan ( Lat ) **2 ) + ( tan ( Lat ) ** 4 ))
    IV  =  v * cos ( Lat )
    V  =  v / 6 * ( cos ( Lat ) ** 3 ) * ( v / p - ( tan ( Lat ) ** 2 ))
    VI  =  v / 120 * ( cos ( Lat ) **5 ) * ( 5 - 18 * ( tan ( Lat ) ** 2 ) + ( tan ( Lat ) ** 4 ) + 14 * n2 - 58 * ( tan ( Lat ) ** 2 ) * n2 )
    # Calculation of the north and east offset (in meters)
    N  =  I + II * ( Lon - Lon0 ) ** 2 + III * ( Lon - Lon0 ) ** 4 + IIIA * ( Lon - Lon0 ) ** 6
    E  =  E0 + IV * ( Lon - Lon0 ) + V * ( Lon - Lon0 ) **3 + VI * ( Lon - Lon0 ) ** 5
    print  ( 'Latitude:' ,  dLat )
    print  ( 'Longitude:' ,  dLon )
    print  ( 'North offset, [m]:' ,  N )
    print  ( 'East offset, [m]:' ,  E )
    return N, E






    def undistPoint(self):
        # Copied Peter Kovesi
        k = self.dist
        # radial lense distortion
        k1, k2, k3 = k[0,0], k[0,1], k[0,4]
        # tangential lens distortion
        p1 = k[0,2]
        p2 = k[0,3]
        # normalised pixels with the origin at the principal point (defined in pix)
        # normalized coordinates corresponding to z = 1
        pxn = (self.px -self.Cc[0])/self.Fc[0]
        pyn = (self.py -self.Cc[1])/self.Fc[1]
        # squared normalized radius
        r2 = pxn**2 + pyn**2
        # distortion scaling factor
        dr = k1*r2 + k2*r2**2 + k3*r2**3
        # Tangential distortion component
        dtx =    2*p1*pxn*pyn + p2*(r2 + 2*pxn**2)
        dty = p1*(r2 + 2*pyn**2) + 2*p2*pxn*pyn
        #Apply the radial and tangential distortion components to x and y
        pxn_ = pxn - dr*pxn - dtx;
        pyn_ = pyn - dr*pyn - dty;
        #Now rescale by f and add the principal point back
        self.pxu = pxn_*self.Fc[0] + self.Cc[0]
        self.pyu = pyn_*self.Fc[1] + self.Cc[1]


    def undistPoint(self):
        # Copied Peter Kovesi
        k = self.dist
        # radial lense distortion
        k1, k2, k3 = k[0,0], k[0,1], k[0,4]
        # tangential lens distortion
        p1 = k[0,2]
        p2 = k[0,3]
        # normalised pixels with the origin at the principal point (defined in pix)
        # normalized coordinates corresponding to z = 1
        pxn = (self.px -self.Cc[0])/self.Fc[0]
        pyn = (self.py -self.Cc[1])/self.Fc[1]
        # squared normalized radius
        r2 = pxn**2 + pyn**2
        # distortion scaling factor
        dr = k1*r2 + k2*r2**2 + k3*r2**3
        # Tangential distortion component
        dtx =    2*p1*pxn*pyn + p2*(r2 + 2*pxn**2)
        dty = p1*(r2 + 2*pyn**2) + 2*p2*pxn*pyn
        #Apply the radial and tangential distortion components to x and y
        pxn_ = pxn - dr*pxn - dtx;
        pyn_ = pyn - dr*pyn - dty;
        #Now rescale by f and add the principal point back
        self.pxu = pxn_*self.Fc[0] + self.Cc[0]
        self.pyu = pyn_*self.Fc[1] + self.Cc[1]
