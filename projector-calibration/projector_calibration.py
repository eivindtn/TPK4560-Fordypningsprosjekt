#!/usr/bin/env python

import cv2
import numpy as np
import os
import glob
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pptk
from generated_pattern.gen_pattern import *
import zivid
from scipy.interpolate import griddata
import open3d
 
#Open3d
np.set_printoptions(suppress=True)
#depth scale set to 1 because of unit in mm 
def pixel_to_point(depth, point, intrinsics, depth_scale=1000):

    """ Get 3D world coordinates of 2D image cordinates in camera

    Parameters:
        point: pixel cordinate u,v
        depth
        np.array - depth camera intrinsics matrix:
                [[fx, 0, ppx],
                 [0, fy, ppy],
                 [0, 0, 1]]

    Returns:
        x, y, z: coordinates in world, might want to round
    """
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    ppx = intrinsics[0, 2]
    ppy = intrinsics[1, 2]
    u, v = point
    x = (u - ppx) / fx
    y = (v - ppy) / fy
    z = depth / depth_scale
    x = x * z
    y = y * z
    return (x, y, z)

def point_to_xy(points, intrinsics):
    """ Get 2D image coordinates of 3D point in camera

    Parameters:
        point: x,y,z vector
        np.array - depth camera intrinsics matrix:
                [[fx, 0, ppx],
                 [0, fy, ppy],
                 [0, 0, 1]]

    Returns:
        x, y: coordinates in image, might want to round
    """

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    ppx = intrinsics[0, 2]
    ppy = intrinsics[1, 2]
    
    xy_points=[]
    for p in points:
        m = p[0] / p[2]
        n = p[1] / p[2]

        x = m * fx + ppx
        y = n * fy + ppy
        xy = [int(x), int(y)]
        xy_points.append(xy)
    print(len(xy_points))
    xy_points = np.unique(xy_points, axis=0)
    print(len(xy_points))
    return np.array(xy_points)

def visualizepointcloud(xyzar,windowname):
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(xyzar)
    coord = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    open3d.visualization.draw_geometries([point_cloud, coord],window_name=windowname,width=1000, height=1080)
def interpolatexyz(array, u,v, px):
    xyz = array[v-px:v+px+1, u-px:u+px+1]
    x_indx,y_indx,z_indx = np.meshgrid(np.arange(0, np.shape(xyz)[0]),np.arange(0,np.shape(xyz)[1]), np.arange(0,np.shape(xyz)[2]))
        
    array_masked = np.ma.masked_invalid(xyz)
    valid_xs = x_indx[~array_masked.mask]
    valid_ys = y_indx[~array_masked.mask]
    valid_zs = z_indx[~array_masked.mask]
    validarr = array_masked[~array_masked.mask]
        
    xyz_interp = griddata((valid_xs, valid_ys, valid_zs), validarr.ravel(),
                                    (x_indx, y_indx, z_indx), method='linear')
    return xyz_interp[px][px]





def main():
    output = r"C:\Users\eivin\Desktop\NTNU master-PUMA-2019-2021\3.studhalvår\TPK4560-Specalization-Project\projector-calibration\generated_pattern\chessboard.svg"
    columns = 10
    rows = 7
    p_type = "checkerboard"
    units = "px"
    square_size = 90
    page_size = "CUSTOM"
    # page size dict (ISO standard, mm) for easy lookup. format - size: [width, height]
    page_sizes = {"CUSTOM": [1024, 768], "A1": [594, 840], "A2": [420, 594], "A3": [297, 420], "A4": [210, 297],
                  "A5": [148, 210]}
    page_width = page_sizes[page_size.upper()][0]
    page_height = page_sizes[page_size.upper()][1]
    pm = PatternMaker(columns, rows, output, units, square_size, page_width, page_height)
    
    fx  = 2767.193359
    fy  = 2766.449119
    ppx = 942.942505
    ppy = 633.898
    
    app = zivid.Application()

    intrinsics = np.array([[fx, 0 , ppx], 
                       [0, fy, ppy],
                       [0, 0, 1]])

    distcoeffszivid = np.array([[-0.26513, 0.282772, 0.000767328, 0.000367037,0]])

    



    # Defining the dimensions of checkerboard, colums by rows
    CHECKERBOARD = (9,6)

    # OpenCV optimizes the camera calibration using the Levenberg-Marquardt 
    # algorithm (Simply put, a combination of Gauss-Newton and Gradient Descent)
    # This defines the stopping criteria.
    # The first parameter states that we limit the algorithm both in terms of
    # desired accuracy, as well as number of iterations. 
    # Here we limit it to 30 iterations or an accuracy of 0.001
    # The criteria is also used by the sub-pixel estimator.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    P_p_2d = []  #Saved 2D cordinates of the i-th corner in the projector image space. This cordinates are imported from the script gen_pattern.py
    P_c_2d = []  #2D cordinates of the i-th corner in the camera image space found by OPENCV findChessboardCorners.
    P_3d   = []  #These cordinates are obatained from mapping the 2D cordinates of the image space to zivid depth camera view space. 
                #This is quiered from using the Zivid camera intrinsics to map from pixel cordinates to point in the space.
    P_3d_obj = []
    

    # Extracting path of individual image stored in a given directory
    images = glob.glob('captured_images/*.png')
    zdf    = glob.glob('zdf/*.zdf')

    for fname in range (0,len(images)):
        img = cv2.imread(images[fname]) #Read the frame name for the images stored in the captured_images folder
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        
        frame = zivid.Frame(zdf[fname])
        point_cloud = frame.get_point_cloud().to_array()
        xyz = np.dstack([point_cloud["x"], point_cloud["y"], point_cloud["z"]])

        #x_indx,y_indx,z_indx = np.meshgrid(np.arange(0, 1920),np.arange(0,1200), np.arange(0,3))
        
        #array_masked = np.ma.masked_invalid(xyz)
        #valid_xs = x_indx[~array_masked.mask]
        #valid_ys = y_indx[~array_masked.mask]
        #valid_zs = z_indx[~array_masked.mask]
        #validarr = array_masked[~array_masked.mask]
        
        #xyz_interp = griddata((valid_xs, valid_ys, valid_zs), validarr.ravel(),
        #                            (x_indx, y_indx, z_indx), method='nearest')
    
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """
        if ret == True:
            P_3d = []
            P_p_2d.append(pm.save_2d_projector_corner_cordinates())
            corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
            P_c_2d.append(corners2)
            #print("Cordinates found in camera fram from findchessboardcorners: \n",corners2[0:10])
            for i in range(len(corners2)):
                if np.isnan(xyz[int(corners2[i][0][1])][int(corners2[i][0][0])]).any() == True:
                    P_3d.append(interpolatexyz(xyz,int(corners2[i][0][0]),int(corners2[i][0][1]), 20))
                else:
                    P_3d.append(xyz[int(corners2[i][0][1])][int(corners2[i][0][0])])
                #P_3d.append(xyz_interp[int(corners2[i][0][1])][int(corners2[i][0][0])])
            
            c_3 = np.identity(3)-(1/3)*np.ones((3,3))
            P_3d_t = np.transpose(P_3d)
            P_3d_c = (P_3d_t.T-np.average(P_3d_t,axis = 1)).T   
            u, s, vh = np.linalg.svd(P_3d_c)
            P_3d_rotate_t = np.dot(np.transpose(u), P_3d_c)
            #P_3d_rotate_t = np.transpose(P_3d_rotate)
            P_3d_obj.append(P_3d_rotate_t.T)
    
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

            #cv2.imshow(images[fname],img)
            #plt.imshow(img)
            #plt.show()
            #cv2.waitKey(0)
            print('Cordinates in frame ', fname, ':\n File location \n ', images[fname], '\n', zdf[fname])
            str = ('``` '
                ' \n'
                'Projector image cordinates     Camera image cordinates    3D cordinates \n'
                #this line is the part I need help with: list comprehension for the 3 lists to output the values as shown below
                '```')

            cordinates = "\n".join("{} {} {}".format(x, y, z) for x, y, z in zip(P_p_2d[fname], corners2, P_3d))
            print(str)
            print(cordinates)  
                        
            #visualizepointcloud(P_3d,zdf[fname]+"3D Cordinates")
            #visualizepointcloud(P_3d_c.T,zdf[fname]+"3D Cordinates centered")
            #visualizepointcloud(P_3d_rotate_t.T,zdf[fname]+"3D Cordinates rotated")          
        
        
    cv2.destroyAllWindows()

    P_3d_obj = np.asarray(P_3d_obj)
    P_3d_obj[:,:,2] = 0
    P_3d_obj = P_3d_obj.astype('float32') 
    P_p_2d = np.asarray(P_p_2d)
    P_p_2d = P_p_2d.astype('float32')
    size=(1024,768)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(P_3d_obj, P_p_2d,size , None, None)

    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(P_3d_obj, P_p_2d, P_c_2d, mtx, dist, intrinsics, distcoeffszivid, size)

    print("Projector instrinsic matrix : \n")
    print(mtx)
    print("Projector disortion coeffsients : \n")
    print(dist)
    #print("rvecs : \n")
    #print(rvecs)
    #print("tvecs : \n")
    #print(tvecs)

    T_C_P = np.column_stack((R,T))
    print("Transformation matrix from camera to projector frame:\n")
    print(T_C_P)
    
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(P_3d)
    c_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=200, origin=[0, 0, 0])
    p_frame = copy.deepcopy(c_frame)
    p_frame.rotate(R, center =(0,0,0))
    p_frame = copy.deepcopy(p_frame).translate((T[0][0], T[1][0], T[2][0]))
    open3d.visualization.draw_geometries([point_cloud ,c_frame, p_frame])



    
if __name__ == "__main__":
    main()

