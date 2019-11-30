#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
from collections import deque
import matplotlib._png as png


# class Camera():
#     def __init__(self):
#         # Stores the source
#         self.ret = None
#         self.mtx = None
#         self.dist = None
#         self.rvecs = None
#         self.tvecs = None
#
#         self.objpoints = []  # 3D points in real space
#         self.imgpoints = []  # 2D points in img space
#
#     def calibrate_camera(self, imgList):
#         counter = 0
#         for img in imgList:
#             # Prepare object points (0,0,0), (1,0,0), etc.
#             objp = np.zeros((nx * ny, 3), np.float32)
#             objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
#
#             # Converting to grayscale
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#             # Finding chessboard corners
#             ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
#             if ret == True:
#                 self.imgpoints.append(corners)
#                 self.objpoints.append(objp)
#                 counter += 1
#         self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints,
#                                                                                     gray.shape[::-1], None, None)
#         return self.mtx, self.dist
#
#     def undistort(self, img):
#         return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)


def pers_transform(img, nx=9, ny=6):
    # Grab the image shape
    img_size = (img.shape[1], img.shape[0])
    # src = np.float32([[190, 720], [582, 457], [701, 457], [1145, 720]])
    src = np.float32([[img_size[0]/3, img_size[1]/2], [0, img_size[1]], [img_size[0], 0], [img_size[0]/2, img_size[1]/2]])
    offset = [150, 0]
    # dst = np.float32([src[0] + offset, np.array([src[0, 0], 0]) + offset,
    #                   np.array([src[3, 0], 0]) - offset, src[3] - offset])

    dst = np.float32([[0, 0], [0, img_size[1]], [img_size[0], 0], [img_size[0], img_size[1]]])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)
    # Return the resulting image and matrix
    Minv = cv2.getPerspectiveTransform(dst, src)

    return warped, M, Minv


def hls_thresh(img, thresh_min=200, thresh_max=255):
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 1]

    # Creating image masked in S channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh_min) & (s_channel <= thresh_max)] = 1
    return s_binary


def sobel_thresh(img, sobel_kernel=3, orient='x', thresh_min=20, thresh_max=100):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)  # Take the derivative in x
        abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    else:
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)  # Take the derivative in x
        abs_sobely = np.absolute(sobely)  # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255 * abs_sobely / np.max(abs_sobely))

    # Creathing img masked in x gradient
    grad_bin = np.zeros_like(scaled_sobel)
    grad_bin[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    return grad_bin


def mag_thresh(img, sobel_kernel=3, thresh_min=100, thresh_max=255):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh_min) & (gradmag <= thresh_max)] = 1

    # Return the binary image
    return binary_output


def dir_thresh(img, sobel_kernel=3, thresh_min=0, thresh_max=np.pi / 2):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh_min) & (absgraddir <= thresh_max)] = 1

    # Return the binary image
    return binary_output


def lab_b_channel(img, thresh=(190, 255)):
    # Normalises and thresholds to the B channel
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    lab_b = lab[:, :, 2]
    # Don't normalize if there are no yellows in the image
    if np.max(lab_b) > 175:
        lab_b = lab_b * (255 / np.max(lab_b))
    #  Apply a threshold
    binary_output = np.zeros_like(lab_b)
    binary_output[((lab_b > thresh[0]) & (lab_b <= thresh[1]))] = 1
    return binary_output

def mask_image(image):
    masked_image = np.copy(image)
    mask = np.zeros_like(masked_image)
    vertices = np.array([[source[0], source[1], source[3], source[2]]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, (1, 1, 1))
    # masked_edges = cv2.bitwise_and(masked_image, mask)
    masked_edges = mask * image

    return masked_edges


def test_process(img):
    # Undistorting image
    # undist = camera.undistort(img)
    undist = img

    # Masking image
    masked = mask_image(undist)

    # Perspective transform image
    # warped, M, Minv = pers_transform(undist)
    warped = masked

    # Colour thresholding in S channel
    s_bin = hls_thresh(warped)

    # Colour thresholding in B channel of LAB
    b_bin = lab_b_channel(warped, thresh=(185, 255))

    # Gradient thresholding with sobel x
    x_bin = sobel_thresh(warped, orient='x', thresh_min=20, thresh_max=100)

    # Gradient thresholding with sobel y
    y_bin = sobel_thresh(warped, orient='y', thresh_min=50, thresh_max=150)

    # Magnitude of gradient thresholding
    mag_bin = mag_thresh(warped, thresh_min=0, thresh_max=255)

    # Direction of gradient thresholding
    dir_bin = dir_thresh(warped, thresh_min=0, thresh_max=np.pi / 2)

    # Combining both thresholds
    combined = np.zeros_like(x_bin)
    combined[(s_bin == 1) | (b_bin == 1)] = 1

    return combined, warped

    # this line never reached
    return combined, warped, Minv

if __name__ == "__main__":

    # # Needed to edit/save/watch video clips
    # from moviepy.editor import VideoFileClip
    # from IPython.display import HTML

    # Importing videos for processing
    # project_clip = VideoFileClip("project_video.mp4")
    #
    # # Importing calibration images
    # cal_filenames = glob.glob('camera_cal/*.jpg')
    # cal_images = np.array([np.array(plt.imread(img)) for img in cal_filenames])
    #
    # # Importing test images
    # test_filenames = glob.glob('test_images/*.jpg')
    # test_images = np.array([np.array(plt.imread(img)) for img in test_filenames])
    #
    # # Chessboard edges
    # nx = 9
    # ny = 6

    test_images = np.array([
        png.read_png_int('./../../kitti/data_road/training/image_2/um_000000.png'),
        png.read_png_int('./../../kitti/data_road/training/image_2/um_000001.png'),
        png.read_png_int('./../../kitti/data_road/training/image_2/um_000002.png'),
    ])
    img = test_images[0]
    img_y, img_x = (img.shape[0], img.shape[1])
    offset = 50

    # Lane masking and coordinates for perspective transform
    # source = np.float32([  # MASK
    #     [img_y - offset, offset],  # bottom left
    #     [img_y - offset, img_x - offset],  # bottom right
    #     [offset, offset],  # top left
    #     [offset, img_x - offset]])  # top right

    source = np.float32([
        [0, img_y], # bottom left
        [img_x, img_y], # bottom right
        [img_x/3, img_y/2], # top left
        [img_x*(2/3), img_y/2]  # top right
    ])

    # dest = np.float32([  # DESTINATION
    #     [300, 720],  # bottom left
    #     [950, 720],  # bottom right
    #     [300, 0],  # top left
    #     [950, 0]])  # top right

    # camera = Camera()
    # Calibrating for the given camera
    # mtx, dist = camera.calibrate_camera(cal_images)
    #
    # img = cv2.imread('camera_cal/calibration1.jpg')
    # plt.figure(figsize=(12, 7))
    # plt.subplot(2, 2, 1)
    # plt.imshow(img)
    # plt.title("Original Image")
    # plt.subplot(2, 2, 2)
    # plt.imshow(cv2.undistort(img,mtx,dist,None,mtx))
    # plt.title("Undistorted Image")
    # img = cv2.imread('test_images/test1.jpg')
    # plt.subplot(2, 2, 3)
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.title("Original Image")
    # plt.subplot(2, 2, 4)
    # plt.imshow(cv2.cvtColor(cv2.undistort(img,mtx,dist,None,mtx), cv2.COLOR_BGR2RGB))
    # plt.title("Undistorted Image")
    # plt.savefig("output_images/undist_img.jpg")

    test_images_thresholded = []

    fig, ax = plt.subplots(len(test_images), 2)
    fig.subplots_adjust(hspace=0.1, wspace=0)
    ax[0][0].set_title('Original', fontsize=15)
    ax[0][1].set_title('Thresholded', fontsize=15)

    for i in range(len(test_images)):
        ax[i][0].imshow(test_images[i])
        # thresholded_img, warped, Minv = test_process(img[i])
        thresholded_img, warped = test_process(test_images[i])
        test_images_thresholded.append(thresholded_img)
        ax[i, 1].imshow(thresholded_img, cmap='gray')

    for axes in ax.flatten():
        axes.axis('off')
    plt.show()
    # plt.savefig("output_images/warp_processing_check.jpg")