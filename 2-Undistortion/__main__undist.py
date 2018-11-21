
""" This is the program for camera calibration based on cyrcular reference pattern

Developed for prototiping of MOVEMENTS project
"""

__version__ = '0.1'
__author__ = 'Igofed'

import argparse
import functInFiles
import functInCalib
import functImShow
import functInColor
import numpy as np
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import pickle

def Calibration(args):
    __mt = functInCalib.Calib(fileName = args.video,
                              pattern_type = args.pattern_type,
                              undistort_path = args.outputUNDISTORTED)
    __mt.findCorners(TypeOfPattern=1)
    print (args)
    print(args.framesNumber)
    ret0 = []
    working_images = []
    #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    readFNum= functInFiles.readFNumInTXT(args.framesNumber)
    if (__mt.cap.isOpened()):
        for i,val in enumerate(readFNum):
            __mt.cap.set(1, val -1)
            #if i <= len(readFNum):
            if i <= 1:

                ret, frame = __mt.cap.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                fNum = round(__mt.cap.get(cv2.CAP_PROP_POS_FRAMES))
                found, circles = __mt._circulargrid_image_points(gray, (cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING), __mt.detector)
                if found ==True:
                    print(i, "fNum: ", fNum, "in: ", __mt.n_frame, " _ ", "Circle found: ", functInColor.color.GREEN + functInColor.color.BOLD + str(found) + functInColor.color.END)
                    S = str(val)+'.jpg'
                    working_images.append(S)
                    cv2.imwrite(S, frame)
                    if __mt.subpixel_refinement:
                        circles2 = cv2.cornerSubPix(gray, circles, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                    else:
                        circles2 = circles.copy()

                    __mt.numGrid.append(fNum)
                    #__mt.objpoints.append(__mt.objp)
                    __mt.objpoints.append(__mt.pattern_points)
                    __mt.imgpoints.append(circles)
                    __mt.PatNum = __mt.PatNum + 1
                    __mt.TotalNumberPatterns = __mt.TotalNumberPatterns +1
                    vis = cv2.drawChessboardCorners(frame.copy(), (__mt.pattern_columns, __mt.pattern_rows), circles2, found)
                    cv2.putText(vis, os.path.basename(args.video), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, 6)
                    name = str(fNum) + '.jpg'
                    fname = args.outputCircles + '/' + name
                    cv2.putText(vis, str(name), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, 6)
                    S1 = 'fNum: ' + str(fNum) + "in: " + str(__mt.n_frame)
                    cv2.putText(vis, S1, (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, 6)
                    cv2.imwrite(fname, vis)

                    # combine it to a dataframe


                else:
                    print(i, "fNum: ", fNum, "in: ", __mt.n_frame, " _ ", "Circle found: ",
                          functInColor.color.RED + functInColor.color.BOLD + str(found) + functInColor.color.END)
            else:
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    #print(__mt.objpoints)
    functImShow.show_imagepoint_in3D(__mt.objpoints)

    __mt.calibration_df = pd.DataFrame({"image_names": working_images,
                                        "img_points": __mt.imgpoints,
                                        "obj_points": __mt.objpoints,
                                        })
    __mt.calibration_df.sort_values("image_names")
    __mt.calibration_df = __mt.calibration_df.reset_index(drop=True)

    print (__mt.calibration_df)
    result_dictionary = __mt.calibrate(frame)
    yamlfiles = args.video + '.yaml'
    print(yamlfiles)
    functInFiles.SaveToFile(yamlfiles, __mt.TotalNumberPatterns, __mt.numGrid, __mt.mtx, __mt.dist, __mt.rms, __mt.rvecs, __mt.tvecs)
    npzlfiles = args.video + '.npz'

    np.savez(npzlfiles, ret=__mt.rms, mtx=__mt.mtx, dist=__mt.dist, rvecs=__mt.rvecs, tvecs=__mt.tvecs)

    with open("./results.pickle","wb") as f:
        pickle.dump(result_dictionary,f)

    functImShow.show_imagepoint_in2D(__mt.imgpoints, args.outputFigures, frame, readFNum)
    __mt.visualize_calibration_boards(cam_width=1,cam_height=1)
    plt.show()
    __mt.cap.release()
    cv2.destroyAllWindows()



    print('Calibration done')

#import Calibration

#def Stich (**args):
#    HL = [-1.35663654e+02, 3.98855246e+02, 8.65938313e+01, 2.15566647e+03,
#          1.70440261e+02, 2.23504549e+02, -2.94372885e+02, 2.13367647e+03,
#          8.16111638e-02, 1.05554248e-01, 3.91167462e-02, 1.00000000e+00]

#    HR = [-1.55728324e+02, 3.92864326e+02, 1.00237827e+02, 2.64696687e+03,
#          1.59833471e+02, 2.31089015e+02, -2.69160860e+02, 2.12964697e+03,
#          7.49548433e-02, 1.09989897e-01, 5.45584058e-02, 1.00000000e+00]
#    val = list(args.values())
    #print(val)
#    __mc= Calibration.Calibration(val[0], val[1], val[2], val[3], HL, HR)

    #mf.imshow('Source images', __mf.image0, __mf.image1, None)

    #plt.show()
 #   print ('done')



def argParse():

    # check if video excist in a folder
    video = functInFiles.ifFileExist("cam1_20180623_144000crop1.mp4")



    #video = functInFiles.ifFileExist("cam2_20180623_144506crop1.mp4")

    #video = functInFiles.ifFileExist("video_name.mp4")

    #video = functInFiles.ifFileExist("video_webcam.mp4")

    # video by defoult search in up one directory

    #framesNumber = functInFiles.ifFileExist("fNumInVideoCalibC1.txt")
    #framesNumber = functInFiles.ifFileExist("fNumInVideoCalibC1__.txt")
    #framesNumber = functInFiles.ifFileExist("fNumInVideoCalibC1_small_pattern.txt")
    framesNumber = functInFiles.ifFileExist("common_pattern.txt")
   # framesNumber = functInFiles.ifFileExist("smal.txt")

    # create directory for output files
    outputCircles = functInFiles.mkDir(dirName="CALIB_CB_ASYMMETRIC_GRID")

    # create directory for output figures
    outputFigures = functInFiles.mkDir(dirName="FIGURES")

    outputUNDISTORTED = functInFiles.mkDir(dirName="UNDISTORTED")
#######################################################
#    outPath = Path.abspath(Path.join(__file__, dirName))
#    timestr = time.strftime("%Y%m%d-%H%M%S")
#    outPath = os.getcwd() + '/Output'
#    imageOutput = outPath + "/" + "out" + timestr +".jpg"
#######################################################
    pattern_type =  "asymmetric_circles"

    parser = argparse.ArgumentParser(description="Calculate calibration matrix and distortion.")
    parser.add_argument("-video", "--video",
                        required=False,
                        type=str,
                        default= video,
                        help='input video with reference patterns')
    parser.add_argument("-framesNumber",
                        "--framesNumber",
                        required=False,
                        type=str,
                        default= framesNumber,
                        help='file with a frames numbers in a video')
    parser.add_argument("-outputCircles",
                        "--outputCircles",
                        required=False,
                        type=str,
                        default= outputCircles,
                        help='file with a frames numbers in a video')
    parser.add_argument("-pattern_type",
                        "--pattern_type",
                        required=False,
                        type=str,
                        default= pattern_type,
                        help='file with a frames numbers in a video')
    parser.add_argument("-outputFigures",
                        "--outputFigures",
                        required=False,
                        type=str,
                        default= outputFigures,
                        help='file with a frames numbers in a video')

    parser.add_argument("-outputUNDISTORTED",
                        "--outputUNDISTORTED",
                        required=False,
                        type=str,
                        default= outputUNDISTORTED)


    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argParse()

    Calibration(args)

  #  args = mf.main()
  #  Stich(**vars(args))
