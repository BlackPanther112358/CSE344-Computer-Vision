import numpy as np
import matplotlib.pyplot as plt
import os, sys
import cv2

DIR_PATH = r"datasets/chessboard/"

image_names = os.listdir(DIR_PATH)
image_names = [DIR_PATH + name for name in image_names]

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((4*6,3), np.float32)
objp[:,:2] = np.mgrid[0:6,0:4].T.reshape(-1,2)

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

for image_name in image_names:
    print(image_name)
    img = cv2.imread(image_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (6,4), None)

    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        img = cv2.drawChessboardCorners(img, (7,5), corners2, ret)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

cv2.destroyAllWindows()

# Calibrating the camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Extracting the intrinsic parameters

fx = mtx[0, 0]
fy = mtx[1, 1]
cx = mtx[0, 2]
cy = mtx[1, 2]
s = mtx[0, 1]
print("Camera matrix : \n")
print(mtx)
print("fx = ", fx)
print("fy = ", fy)
print("cx = ", cx)
print("cy = ", cy)
print("s = ", s)

# Extrinsics parameters for the selected images

for i in range(len(objpoints)):
    rotation_matrix, _ = cv2.Rodrigues(rvecs[i])
    print(f'{image_names[i]}:')
    print('Rotation Matrix:')
    print(rotation_matrix)
    print('Translation Vector:')
    print(tvecs[i])
    print()

print(f'Distortion coefficients: {dist}')

# Undistorting the first 5 images
for i in range(5):
    img = cv2.imread(image_names[i])
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    # Plot the images side by side
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image')
    ax2.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    ax2.set_title('Undistorted Image')
    plt.show()

for i in range(len(objpoints)):
    img = cv2.imread(image_names[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib

    # Draw the detected corners
    img_detected = cv2.drawChessboardCorners(img.copy(), (7,6), imgpoints[i], True)

    # Reproject the corners
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    imgpoints2 = imgpoints2.reshape(-1, 2)
    img_reprojected = cv2.drawChessboardCorners(img.copy(), (7,6), imgpoints2, True)

    # Plot the images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_detected)
    plt.title('Detected Corners')
    plt.subplot(1, 2, 2)
    plt.imshow(img_reprojected)
    plt.title('Reprojected Corners')
    plt.show()

for i in range(len(objpoints)):
    # Convert rotation vectors to rotation matrices
    rotation_matrix, _ = cv2.Rodrigues(rvecs[i])

    # The third column is the normal of the plane
    normal = rotation_matrix[:, 2]

    print(f'{image_names[i]}:')
    print(f'Plane Normal: {normal}')

