
""" This is the program for checking if file excist in upfolder
"""
import sys
import os
import functInColor
import numpy as np
import matplotlib.pyplot as plt
import cv2
def ifFileExistBot(filename):

    # find a file in up folder
    #video = os.path.dirname(os.getcwd()) + '/' + filename
    try:
        myFile = open(filename)
        #os.path.isfile(video)
        #print(os.path.isfile(video))
        print(functInColor.color.GREEN + functInColor.color.BOLD + "File {}: EXIST".format(filename) + functInColor.color.END)
    except OSError:
        #print(color.RED + 'Hello World !' + color.END)
        print (functInColor.color.RED + "File {}: NOT FOUND".format(filename)  + functInColor.color.END)

        sys.exit()
    return video
	
	

def ifFileExist(filename):

    # find a file in up folder
    video = os.path.dirname(os.getcwd()) + '/' + filename
    try:
        myFile = open(video)
        #os.path.isfile(video)
        #print(os.path.isfile(video))
        print(functInColor.color.GREEN + functInColor.color.BOLD + "File {}: EXIST".format(filename) + functInColor.color.END)
    except OSError:
        #print(color.RED + 'Hello World !' + color.END)
        print (functInColor.color.RED + "File {}: NOT FOUND".format(filename)  + functInColor.color.END)

        sys.exit()
    return video

def mkDir(dirName = "Output"):
    # create directory for output files
    try:
        os.mkdir(dirName)
        print(functInColor.color.GREEN + functInColor.color.BOLD + "Directory {}: Created".format(dirName) + functInColor.color.END)
    except FileExistsError:
        print(functInColor.color.BLUE + "Directory {}: ALREADY EXIST".format(dirName) + functInColor.color.END)
    return dirName

def readFNumInTXT(filename):

    readFNum = np.loadtxt(filename, delimiter=' ', unpack=True)
    readFNum =[int(i) for i in readFNum]
    #if f.mode == "r":
     #   content = f.read()
      #  for row in content:
    #POI.frame = frame
    #P#OI.x = x
    #POI.y = y
    print (functInColor.color.BLUE + "{} frames selected".format(len(readFNum)) + functInColor.color.END)
    print(functInColor.color.BLUE, readFNum, functInColor.color.END)
    return readFNum

def SaveToFile(fileName, TotalNumberPatterns, numGrid, K, D, rms, rvecs, tvecs):
    fs = cv2.FileStorage (fileName, cv2.FILE_STORAGE_WRITE)

    #with open(fileName, 'w') as fs:
        #fs.write("intrinsic", K)
        #fs.write("distCoeff", D)
        #fs.write("RMS", rms)
    print("D", D)
    print("RVECS", type(rvecs))
    print("TVECS", tvecs)

    #fs.write('---\n')
    #fs.write('distortion_coefficients :\n')
    #fs.write('  -\n')
    fs.write("intrinsic", K)

    fs.write('fx',K[0][0])
    fs.write('fy',K[1][1])
    fs.write('cx',K[0][2])
    fs.write('cy',K[1][2])

    fs.write("distCoeff", D)
    fs.write('k1', D[0][0])
    fs.write('k2', D[0][1])
    fs.write('p1', D[0][2])
    fs.write('p2', D[0][3])
    fs.write('k3', D[0][4])

    fs.write("rms", rms)
    R = np.array(rvecs)
    T = np.array(tvecs)
    numGrid = np.array(numGrid)
    #for i in (rvecs):
    ##    i = list(i)
     #   R.append(i)

    fs.write("frames", TotalNumberPatterns)
    fs.write("numGrid", numGrid)
    print ("R", R)
    fs.write("R", R)
    fs.write("T", T)

    #    fs.write('camera_matrix :\n')
    #    fs.write('  -\n')
    #fs.write('    fx : ' + repr(K[0][0]) + '\n')
    #fs.write('    fy : ' + repr(K[1][1]) + '\n')
    #fs.write('    cx : ' + repr(K[0][2]) + '\n')
    #fs.write('    cy : ' + repr(K[1][2]) + '\n')

        #fs.write("rotMatrix", rvecs[0])
        #fs.write("rotMatrix", rvecs[0])
    #fs.write("transVect", tvecs)
    fs.release()
    print ("Files saved")

#video = ifFileExist("calib_c1_20180623_224103New_crop.mp4")