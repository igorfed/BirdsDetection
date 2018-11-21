
""" This is the program for checking if file excist in upfolder
"""
import sys
import os
import functInColor
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import functImShow
from matplotlib import cm
import timeit
from numpy import linspace
class Calib(object):
    def __init__(self, fileName, pattern_type, undistort_path):
        self.pSize = 3.45E-6
        #self.pattern_size = (9, 6)
        self.undistort_path = undistort_path
        self.pattern_columns = 4
        self.pattern_rows = 11

        self.imgpoints = []
        self.objpoints = []

        self.calc_timestamps = [0.0]
        self.calibration = {}
        self.fileName = fileName
        self.fNum = 0
        self.PatNum = 0
        self.objpoints = []
        self.imgpoints = []
        self.figsize = (8,8)
        self.calibration_df = []
        self.mError = 0
        self.numGrid = []
        self.VideoCap()
        self.mean_error = 0
        self.TotalNumberPatterns = 0
        self.NumberPatterns = []
        assert pattern_type in ["chessboard", "asymmetric_circles"], "Unexpected type of pattern {}".format(["chessboard", "asymmetric_circles"])
        self.pattern_type = pattern_type
        self.subpixel_refinement = False #turn on or off subpixel refinement
        self.distance_in_world_units = 95.0 ##mm



        if self.pattern_type == "asymmetric_circles":
            self.detector = self.BlobDetector(filetrByArea = True,
                                              minArea = 100,
                                              maxArea = 20000,
                                              filterByCircularity = True,
                                              minCircularity = 0.2,
                                              filterByInertia = True,
                                              minInertiaRatio = 0.1)
            self.double_count_in_column = False
        self.pattern_points = self._asymmetric_world_points() * self.distance_in_world_units

        print(functInColor.color.GREEN + functInColor.color.BOLD + 'Calibration initialized' + functInColor.color.END)
        print(self.pattern_points)
        functImShow.show_imagepoint_in3Dtst(self.pattern_points)
        plt.show()

    def _asymmetric_world_points(self):
        pattern_points = []
        if self.double_count_in_column:
            for i in range(self.pattern_rows):
                for j in range(self.pattern_columns):
                    x = j / 2
                    if j % 2 == 0:
                        y = i
                    else:
                        y = i + 0.5
                    pattern_points.append((x, y))
        else:
            for i in range(self.pattern_rows):
                for j in range(self.pattern_columns):
                    y = i / 2
                    if i % 2 == 0:
                        x = j
                    else:
                        x = j + 0.5

                    pattern_points.append((x, y))

        pattern_points = np.hstack((pattern_points, np.zeros((self.pattern_rows * self.pattern_columns, 1)))).astype(
            np.float32)
        return (pattern_points)

    def _calc_reprojection_error(self, figure_size=(8, 8), save_dir=None):
        reprojection_error = []
        for i in range(len(self.calibration_df)):
            imgpoints2, _ = cv2.projectPoints(self.calibration_df.obj_points[i], self.calibration_df.rvecs[i], self.calibration_df.tvecs[i], self.mtx, self.dist)
            temp_error = cv2.norm(self.calibration_df.img_points[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            reprojection_error.append(temp_error)
        self.calibration_df['reprojection_error'] = pd.Series(reprojection_error)
        avg_error = np.sum(np.array(reprojection_error))/len(self.calibration_df.obj_points)
        x = [os.path.basename(p) for p in self.calibration_df.image_names]
        y_mean = [avg_error]*len(self.calibration_df.image_names)
        fig, ax = plt.subplots()
        fig.set_figwidth(figure_size[0])
        fig.set_figheight(figure_size[1])
        # Plot the data
        ax.scatter(x, reprojection_error, label='Reprojection error', marker='o')  # plot before
        # Plot the average line
        ax.plot(x, y_mean, label='Mean Reprojection error', linestyle='--')
        # Make a legend
        ax.legend(loc='upper right')
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
        # name x and y axis
        ax.set_title("Reprojection_error plot")
        ax.set_xlabel("Image_names")
        ax.set_ylabel("Reprojection error in pixels")

        if save_dir:
            plt.savefig(os.path.join(save_dir, "reprojection_error.png"))


        print("The Mean Reprojection Error in pixels is:  {}".format(avg_error))

    def VideoCap(self):
        self.cap = cv2.VideoCapture(self.fileName)
        if self.cap.isOpened() == False:
            print(functInColor.color.RED + "Error opening video stream or file" + functInColor.color.END)
        else:
            print(functInColor.color.GREEN + functInColor.color.BOLD +"Video :", self.fileName, " is opened" + functInColor.color.END)
            self.n_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.frames = np.zeros((self.height, self.width, 3), 'uint8')
            print("nframes: ", self.n_frame, " fps: ", " [ ", self.width, " x ", self.height, "]")

    def BlobDetector(self, filetrByArea = True, minArea = 100, maxArea = 200000,
                     filterByCircularity = True,
                     minCircularity = 0.2,
                     filterByInertia = True,
                     minInertiaRatio = 0.1 ):

        self.params = cv2.SimpleBlobDetector_Params()
        self.params.filterByArea = filetrByArea
        self.params.minArea = minArea
        self.params.maxArea = maxArea
        self.params.filterByCircularity = filterByCircularity
        self.params.minCircularity = minCircularity
        self.params.filterByInertia = filterByInertia
        self.params.minInertiaRatio = minInertiaRatio

        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3:
            print('SimpleBlobDetector')
            return cv2.SimpleBlobDetector(self.params)
        else:
            print(functInColor.color.GREEN + functInColor.color.BOLD + 'SimpleBlobDetector_create' + functInColor.color.END)
            return cv2.SimpleBlobDetector_create(self.params)

    def findCorners(self, TypeOfPattern = 1):
        params = []
        def objParams(TypeOfPattern):
            if TypeOfPattern == 1:
                param1 = np.zeros((4 * 11, 3), np.float32)
                param2 = np.mgrid[0:4,0:11].T.reshape(-1,2)
                print("Cyrcle Calibration Pattern")
            elif TypeOfPattern == 0:
                param1 = np.zeros((6 * 9, 3), np.float32)
                param2 = np.mgrid[0:9,0:6].T.reshape(-1,2)
                print("Checkerboard Calibration Pattern")
            else:
                param1 = np.zeros((4 * 11, 3), np.float32)
                param2 = np.mgrid[0:4,0:11].T.reshape(-1,2)
                print ("Type of pattern unrecognized")
            params.append(param1)
            params.append(param2)
        objParams(TypeOfPattern)
        self.objp = params[0]
        self.objp[:, :2] = params[1]
        #objParams(TypeOfPattern)



    def get_frames(self): # return left, right
        if self.zed:

            # grab single frame from camera (read = grab/retrieve)
            # and split into Left and Right

            _, frame = self.camZED.read()
            height,width, channels = frame.shape
            frameL= frame[:,0:int(width/2),:]
            frameR = frame[:,int(width/2):width,:]
        else:
            # grab frames from camera (to ensure best time sync.)

            self.camL.grab();
            self.camR.grab();

            # then retrieve the images in slow(er) time
            # (do not be tempted to use read() !)

            _, frameL = self.camL.retrieve();
            _, frameR = self.camR.retrieve();
#
        return frameL, frameR

    def _circulargrid_image_points(self, frame, flags, blobDetector):
        found, corners = cv2.findCirclesGrid(frame,
                                             (self.pattern_columns, self.pattern_rows),
                                             flags=flags,
                                             blobDetector=blobDetector)
        return (found, corners)


        return (found, corners)

    def visualize_calibration_boards(self,
                                     cam_width=20.0,
                                     cam_height=10.0,
                                     scale_focal=40):
        def _create_camera_model(camera_matrix, width, height, scale_focal, draw_frame_axis=False):
            # util function
            fx = camera_matrix[0, 0]
            fy = camera_matrix[1, 1]
            focal = 2 / (fx + fy)
            f_scale = scale_focal * focal

            # draw image plane
            X_img_plane = np.ones((4, 5))
            X_img_plane[0:3, 0] = [-width, height, f_scale]
            X_img_plane[0:3, 1] = [width, height, f_scale]
            X_img_plane[0:3, 2] = [width, -height, f_scale]
            X_img_plane[0:3, 3] = [-width, -height, f_scale]
            X_img_plane[0:3, 4] = [-width, height, f_scale]

            # draw triangle above the image plane
            X_triangle = np.ones((4, 3))
            X_triangle[0:3, 0] = [-width, -height, f_scale]
            X_triangle[0:3, 1] = [0, -2 * height, f_scale]
            X_triangle[0:3, 2] = [width, -height, f_scale]

            # draw camera
            X_center1 = np.ones((4, 2))
            X_center1[0:3, 0] = [0, 0, 0]
            X_center1[0:3, 1] = [-width, height, f_scale]

            X_center2 = np.ones((4, 2))
            X_center2[0:3, 0] = [0, 0, 0]
            X_center2[0:3, 1] = [width, height, f_scale]

            X_center3 = np.ones((4, 2))
            X_center3[0:3, 0] = [0, 0, 0]
            X_center3[0:3, 1] = [width, -height, f_scale]

            X_center4 = np.ones((4, 2))
            X_center4[0:3, 0] = [0, 0, 0]
            X_center4[0:3, 1] = [-width, -height, f_scale]

            # draw camera frame axis
            X_frame1 = np.ones((4, 2))
            X_frame1[0:3, 0] = [0, 0, 0]
            X_frame1[0:3, 1] = [f_scale / 2, 0, 0]

            X_frame2 = np.ones((4, 2))
            X_frame2[0:3, 0] = [0, 0, 0]
            X_frame2[0:3, 1] = [0, f_scale / 2, 0]

            X_frame3 = np.ones((4, 2))
            X_frame3[0:3, 0] = [0, 0, 0]
            X_frame3[0:3, 1] = [0, 0, f_scale / 2]

            if draw_frame_axis:
                return [X_img_plane, X_triangle, X_center1, X_center2, X_center3, X_center4, X_frame1, X_frame2,
                        X_frame3]
            else:
                return [X_img_plane, X_triangle, X_center1, X_center2, X_center3, X_center4]

        def _create_board_model(extrinsics, board_width, board_height, square_size, draw_frame_axis=False):
            # util function
            width = board_width * square_size
            height = board_height * square_size

            # draw calibration board
            X_board = np.ones((4, 5))
            # X_board_cam = np.ones((extrinsics.shape[0],4,5))
            X_board[0:3, 0] = [0, 0, 0]
            X_board[0:3, 1] = [width, 0, 0]
            X_board[0:3, 2] = [width, height, 0]
            X_board[0:3, 3] = [0, height, 0]
            X_board[0:3, 4] = [0, 0, 0]

            # draw board frame axis
            X_frame1 = np.ones((4, 2))
            X_frame1[0:3, 0] = [0, 0, 0]
            X_frame1[0:3, 1] = [height / 2, 0, 0]

            X_frame2 = np.ones((4, 2))
            X_frame2[0:3, 0] = [0, 0, 0]
            X_frame2[0:3, 1] = [0, height / 2, 0]

            X_frame3 = np.ones((4, 2))
            X_frame3[0:3, 0] = [0, 0, 0]
            X_frame3[0:3, 1] = [0, 0, height / 2]

            if draw_frame_axis:
                return [X_board, X_frame1, X_frame2, X_frame3]
            else:
                return [X_board]

        def _inverse_homogeneoux_matrix(M):
            # util_function
            R = M[0:3, 0:3]
            T = M[0:3, 3]
            M_inv = np.identity(4)
            M_inv[0:3, 0:3] = R.T
            M_inv[0:3, 3] = -(R.T).dot(T)

            return M_inv

        def _transform_to_matplotlib_frame(cMo, X, inverse=False):
            # util function
            M = np.identity(4)
            M[1, 1] = 0
            M[1, 2] = 1
            M[2, 1] = -1
            M[2, 2] = 0

            if inverse:
                return M.dot(_inverse_homogeneoux_matrix(cMo).dot(X))
            else:
                return M.dot(cMo.dot(X))

        def _draw_camera_boards(ax, camera_matrix, cam_width, cam_height, scale_focal,
                                extrinsics, board_width, board_height, square_size,
                                patternCentric):
            # util function
            min_values = np.zeros((3, 1))
            min_values = np.inf
            max_values = np.zeros((3, 1))
            max_values = -np.inf

            if patternCentric:
                X_moving = _create_camera_model(camera_matrix, cam_width, cam_height, scale_focal)
                X_static = _create_board_model(extrinsics, board_width, board_height, square_size)
            else:
                X_static = _create_camera_model(camera_matrix, cam_width, cam_height, scale_focal, True)
                X_moving = _create_board_model(extrinsics, board_width, board_height, square_size)

            cm_subsection = linspace(0.0, 1.0, extrinsics.shape[0])
            colors = [cm.jet(x) for x in cm_subsection]

            for i in range(len(X_static)):
                X = np.zeros(X_static[i].shape)
                for j in range(X_static[i].shape[1]):
                    X[:, j] = _transform_to_matplotlib_frame(np.eye(4), X_static[i][:, j])
                ax.plot3D(X[0, :], X[1, :], X[2, :], color='r')
                min_values = np.minimum(min_values, X[0:3, :].min(1))
                max_values = np.maximum(max_values, X[0:3, :].max(1))

            for idx in range(extrinsics.shape[0]):
                R, _ = cv2.Rodrigues(extrinsics[idx, 0:3])
                cMo = np.eye(4, 4)
                cMo[0:3, 0:3] = R
                cMo[0:3, 3] = extrinsics[idx, 3:6]
                for i in range(len(X_moving)):
                    X = np.zeros(X_moving[i].shape)
                    for j in range(X_moving[i].shape[1]):
                        X[0:4, j] = _transform_to_matplotlib_frame(cMo, X_moving[i][0:4, j], patternCentric)
                    ax.plot3D(X[0, :], X[1, :], X[2, :], color=colors[idx])
                    min_values = np.minimum(min_values, X[0:3, :].min(1))
                    max_values = np.maximum(max_values, X[0:3, :].max(1))

            return min_values, max_values

        def visualize_views(camera_matrix,
                            rvecs,
                            tvecs,
                            board_width,
                            board_height,
                            square_size,
                            cam_width=64 / 2,
                            cam_height=48 / 2,
                            scale_focal=40,
                            patternCentric=False,
                            figsize=(8, 8),
                            save_dir=None
                            ):
            """
            Visualizes the pattern centric or the camera centric views of chess board
            using the above util functions

            Keyword Arguments

            camera_matrix --numpy.array: intrinsic camera matrix (No default)
            rvecs : --list of rvecs from cv2.calibrateCamera()
            tvecs : --list of tvecs from cv2.calibrateCamera()

            board_width --int: the chessboard width (no default)
            board_height --int: the chessboard height (no default)
            square_size --int: the square size of each chessboard square in mm
            cam_width --float: Width/2 of the displayed camera (Default 64/2)
                               it is recommended to leave this argument to default
            cam_height --float: Height/2 of the displayed camera (Default (48/2))
                                it is recommended to leave this argument to default
            scale_focal --int: Value to scale the focal length (Default 40)
                               it is recommended to leave this argument to default

            pattern_centric --bool: Whether to visualize the pattern centric or the
                                    camera centric (Default False)
            fig_size --tuple: The size of figure to display (Default (8,8))
                              it is recommended to leave this argument to default

            save_dir --str: optional path to a saving directory to save the
                            generated plot (Default None)

            Does not return anything
            """
            i = 0
            extrinsics = np.zeros((len(rvecs), 6))
            for rot, trans in zip(rvecs, tvecs):
                extrinsics[i] = np.append(rot.flatten(), trans.flatten())
                i += 1
            # The extrinsics  matrix is of shape (N,6) (No default)
            # Where N is the number of board patterns
            # the first 3  columns are rotational vectors
            # the last 3 columns are translational vectors

            fig = plt.figure(figsize=figsize)
            ax = fig.gca(projection='3d')
            ax.set_aspect("equal")

            min_values, max_values = _draw_camera_boards(ax, camera_matrix, cam_width, cam_height,
                                                         scale_focal, extrinsics, board_width,
                                                         board_height, square_size, patternCentric)

            X_min = min_values[0]
            X_max = max_values[0]
            Y_min = min_values[1]
            Y_max = max_values[1]
            Z_min = min_values[2]
            Z_max = max_values[2]
            max_range = np.array([X_max - X_min, Y_max - Y_min, Z_max - Z_min]).max() / 2.0

            mid_x = (X_max + X_min) * 0.5
            mid_y = (Y_max + Y_min) * 0.5
            mid_z = (Z_max + Z_min) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

            ax.set_xlabel('x')
            ax.set_ylabel('z')
            ax.set_zlabel('-y')
            if patternCentric:
                ax.set_title('Pattern Centric View')
                if save_dir:
                    plt.savefig(os.path.join(save_dir, "pattern_centric_view.png"))
            else:
                ax.set_title('Camera Centric View')
                if save_dir:
                    plt.savefig(os.path.join(save_dir, "camera_centric_view.png"))
            plt.show()

        """
        User facing method to visualize the calibration board orientations in 3-D
        Plots both the pattern centric and the camera centric views

        Keyword Arguments:
        cam_width --float: width of cam in visualization (Default 20.0)
        cam_height --float: height of cam in visualization (Default 10.0)
        scale_focal --int: Focal length is scaled accordingly (Default 40)

        Output:
            Plots the camera centric and pattern centric views of the chessboard in 3-D using matplotlib
            Optionally saves these views in the debug directory if the constructor is initialized with
            debug directory

        TIP: change the values of cam_width, cam_height for better visualizations
        """

        # Plot the camera centric view
        visualize_views(camera_matrix=self.mtx,
                        rvecs=self.calibration_df.rvecs,
                        tvecs=self.calibration_df.tvecs,
                        board_width=self.pattern_columns,
                        board_height=self.pattern_rows,
                        square_size=self.distance_in_world_units,
                        cam_width=cam_width,
                        cam_height=cam_height,
                        scale_focal=scale_focal,
                        patternCentric=False,
                        figsize=self.figsize,
                        save_dir=None
                        )
        # Plot the pattern centric view
        visualize_views(camera_matrix=self.mtx,
                        rvecs=self.calibration_df.rvecs,
                        tvecs=self.calibration_df.tvecs,
                        board_width=self.pattern_columns,
                        board_height=self.pattern_rows,
                        square_size=self.distance_in_world_units,
                        cam_width=cam_width,
                        cam_height=cam_height,
                        scale_focal=scale_focal,
                        patternCentric=True,
                        figsize=self.figsize,
                        save_dir=None
                        )

        #        self._calc_reprojection_error(figure_size=self.figsize, save_dir=self.debug_dir)
    def undistortFrame(self, frame):

        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx,
                                                          self.dist,
                                                          (self.width, self.height),
                                                          1,
                                                          (self.width, self.height))
        imageUndist = cv2.undistort(frame,
                                    self.mtx,
                                    self.dist,
                                    None,
                                    newcameramtx)
        S = self.undistort_path + '/' + 'undist_' + time.strftime("%Y%m%d-%H%M%S") + '.jpg'
        cv2.imwrite(S, imageUndist)
        S = self.undistort_path + '/' + 'source.jpg'
        cv2.imwrite(S, frame)

        #img = cv2.imread('SourceFile2.jpg', 0)
        #imageUndist = cv2.undistort(img, self.mtx, self.dist, None, newcameramtx)
        #cv2.imwrite('SourceFile1Undist.jpg', imageUndist)

        print('saved')
    def calibrate(self, frame):
        start = timeit.default_timer()
        print("start calibrateCamera: ",start )

        self.rms, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(self.calibration_df.obj_points,
                                                                          self.calibration_df.img_points,
                                                                          (self.width, self.height),
                                                                          None,
                                                                          None)
        self.calibration_df['rvecs'] = pd.Series(rvecs)
        self.calibration_df['tvecs'] = pd.Series(tvecs)
        self.rvecs = rvecs
        self.tvecs = tvecs
        print('RMS:\n ', self.rms)
        print('Cam_mat: ', self.mtx)

       # self.mtx[0][0] = 2.3235440343394721e+03
       # self.mtx[1][1] = 2.3235440343394721e+03
       # self.mtx[0][2] = 1.8392201597271251e+03
       # self.mtx[1][2] = 1.0264765716380098e+03
       ## self.dist[0][0] = 1.8788805085284310e-02
       # self.dist[0][1] = -7.1286607829337517e-02
       # self.dist[0][2] = -6.6373990738412226e-03
       # self.dist[0][3] = -1.6133810429510315e-03
    #self.dist[0][4] = 2.0784571417338218e-02

        print('Dist: \n', self.dist.ravel())
        print('k1: \n', self.dist[0][0])
        print('k2: \n', self.dist[0][1])
        print('p1: \n', self.dist[0][2])
        print('p2: \n', self.dist[0][3])
        print('k3: \n', self.dist[0][4])
        print('R: \n', self.rvecs)
        print('T: \n', self.tvecs)
        print('fx ={}, fy = {}'.format(self.mtx[0][0], self.mtx[1][1]))
        print('fx ={} mm, fy = {} mm'.format(self.mtx[0][2]*13.2/self.width, self.mtx[1][2]*7.5/self.height))
        print('cx ={}, cy = {}'.format(self.mtx[0][2], self.mtx[1][2]))

#        self._calc_reprojection_error(figure_size=self.figsize, save_dir=self.debug_dir)
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (self.width, self.height), 1, (self.width, self.height))

        stop = timeit.default_timer()

        print('Time: ', stop - start)

        imageUndist = cv2.undistort(frame, self.mtx, self.dist, None, newcameramtx)
        cv2.imwrite('undistorted.jpg', imageUndist)
        cv2.imwrite('source.jpg', frame)
        img = cv2.imread('53.jpg')
        imageUndist = cv2.undistort(img, self.mtx, self.dist, None, newcameramtx)
        cv2.imwrite('53Undist.jpg', imageUndist)
        print('saved')

        #x, y, w, h = roi
        # plot the reprojection error graph


        self._calc_reprojection_error(figure_size=self.figsize, save_dir=None)
        result_dictionary = {
                             "rms":self.rms,
                             "intrinsic_matrix":self.mtx,
                             "distortion_coefficients":self.dist,
                             "R": self.rvecs,
                             "T": self.tvecs}


        # Fx = fx * W/w
        # fx - focal length in pixel unit
        # W - sesors width in mm
        # w - image width in pixels


        #undistortion
        #newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (self.width, self.height), 1, (self.width, self.height))

        #imageUndist = imageUndist[y:y + h, x:x + w]

        impoints2D = plt.figure('Undistort', figsize=(20, 15))

#        np.savetxt('newcameramtx.out', newcameramtx)
        np.savetxt('mtx.out', self.mtx)
        np.savetxt('dist.out', self.dist)

        # self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints,self.img.shape[::-1], None, None)
        print('Dist: ', self.dist)
        print('Cam_mat: ', self.mtx)
        print('newcammatr: ', self.mtx)
        #self.dist[0][2] = 0
        #self.dist[0][3] = 0
        #self.dist[0][4] = 0
        #print('Dist: ', self.dist)

#        self.error_in_frame = []
 #       for i in range(len(self.objpoints)):
  ##          imgpoints2, _ = cv2.projectPoints(self.objpoints[i], self.rvecs[i], self.tvecs[i], self.mtx, self.dist)
  #          error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
  #          self.mean_error += error
  #          self.error_in_frame.append(error)
  #      self.mean_error = self.mean_error / len(self.objpoints)
  #      print("total error: ", self.mean_error)
       # print(type(self.mtx), "mtx: ", self.mtx, " ----------------------------------> [", self.mtx.shape, "]")
       # print('dist', self.dist, " ----------> [", self.dist.shape, "]")
        #imageFrame = cv2.imread('C:/Working/Skagen/2-Point2D/R3.jpg',0)
        print("end calibrateCamera")



        plt.imshow(imageUndist)
        return (result_dictionary)