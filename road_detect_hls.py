import matplotlib.pyplot as plt
import cv2
import os, glob
import numpy as np
import matplotlib._png as png
from moviepy.editor import VideoFileClip

#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'


def show_images(images, cmap=None):
    cols = 2
    rows = (len(images) + 1) // cols

    plt.figure(figsize=(10, 11))
    for i, image in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        # use gray scale color map if there is only one channel
        cmap = 'gray' if len(image.shape) == 2 else cmap
        plt.imshow(image, cmap=cmap)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()

#test_images = [plt.imread(path) for path in glob.glob('test_images/*.jpg')]

#show_images(test_images)

# image is expected be in RGB color space
def select_rgb_white_yellow(image):
    # white color mask
    lower = np.uint8([200, 200, 200])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower, upper)
    # yellow color mask
    lower = np.uint8([190, 190,   0])
    upper = np.uint8([255, 255, 255])
    yellow_mask = cv2.inRange(image, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked = cv2.bitwise_and(image, image, mask = mask)
    return masked

#show_images(list(map(select_rgb_white_yellow, test_images)))

def convert_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

#show_images(list(map(convert_hsv, test_images)))

def convert_hls(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

#show_images(list(map(convert_hls, test_images)))

def select_white_yellow(image):
    converted = convert_hls(image)
    # white color mask
    lower = np.uint8([  0, 200,   0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted, lower, upper)
    # yellow color mask
    lower = np.uint8([ 10,   0, 100])
    upper = np.uint8([ 40, 255, 255])
    yellow_mask = cv2.inRange(converted, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask = mask)

#mask detect only white not yellow


#white_yellow_images = list(map(select_white_yellow, test_images))

#show_images(white_yellow_images)

def convert_gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def apply_smoothing(image, kernel_size=15):
    """
    kernel_size must be postivie and odd
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def detect_edges(image, low_threshold=50, high_threshold=150):
    return cv2.Canny(image, low_threshold, high_threshold)


def filter_region(image, vertices):
    """
    Create the mask using the vertices and apply it to the input image
    """
    mask = np.zeros_like(image)
    if len(mask.shape) == 2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,) * mask.shape[2])  # in case, the input image has a channel dimension
    return cv2.bitwise_and(image, mask)


def select_region(image):
    """
    It keeps the region surrounded by the `vertices` (i.e. polygon).  Other area is set to 0 (black).
    """
    # first, define the polygon by vertices
    rows, cols = image.shape[:2]
    bottom_left = [cols * 0.1, rows * 0.95]
    top_left = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right = [cols * 0.6, rows * 0.6]
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return filter_region(image, vertices)


def hough_lines(image):
    """
    `image` should be the output of a Canny transform.

    Returns hough lines (not the image with lines)
    """
    return cv2.HoughLinesP(image, rho=1, theta=np.pi / 180, threshold=20, minLineLength=40, maxLineGap=20)

def draw_lines(image, lines, color=[255, 0, 0], thickness=2, make_copy=True):
    # the lines returned by cv2.HoughLinesP has the shape (-1, 1, 4)
    if make_copy:
        image = np.copy(image) # don't want to modify the original
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image


def average_slope_intercept(lines):
    left_lines = []  # (slope, intercept)
    left_weights = []  # (length,)
    right_lines = []  # (slope, intercept)
    right_weights = []  # (length,)

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                continue  # ignore a vertical line
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            if slope < 0:  # y is reversed in image
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))

    # add more weight to longer lines
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None

    return left_lane, right_lane  # (slope, intercept), (slope, intercept)


def make_line_points(y1, y2, line):
    """
    Convert a line represented in slope and intercept into pixel points
    """
    if line is None:
        return None

    slope, intercept = line

    # make sure everything is integer as cv2.line requires it
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)

    return ((x1, y1), (x2, y2))


def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)

    y1 = image.shape[0]  # bottom of the image
    y2 = y1 * 0.6  # slightly lower than the middle

    left_line = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)

    return left_line, right_line


def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=20):
    # make a separate image to draw lines and combine with the orignal later
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, color, thickness)
    # image1 * α + image2 * β + λ
    # image1 and image2 must be the same shape.
    return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0)

def shadow(img):
    or_img = img
    #or_img = cv2.imread('./data_road/training/image_2/um_000007.png')

    # covert the BGR image to an YCbCr image
    y_cb_cr_img = cv2.cvtColor(or_img, cv2.COLOR_BGR2YCrCb)

    # copy the image to create a binary mask later
    binary_mask = np.copy(y_cb_cr_img)

    # get mean value of the pixels in Y plane
    y_mean = np.mean(cv2.split(y_cb_cr_img)[0])

    # get standard deviation of channel in Y plane
    y_std = np.std(cv2.split(y_cb_cr_img)[0])

    # classify pixels as shadow and non-shadow pixels
    for i in range(y_cb_cr_img.shape[0]):
        for j in range(y_cb_cr_img.shape[1]):

            if y_cb_cr_img[i, j, 0] < y_mean - (y_std / 3):
                # paint it white (shadow)
                binary_mask[i, j] = [255, 255, 255]
            else:
                # paint it black (non-shadow)
                binary_mask[i, j] = [0, 0, 0]

    # Using morphological operation
    # The misclassified pixels are
    # removed using dilation followed by erosion.
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(binary_mask, kernel, iterations=1)

    # sum of pixel intensities in the lit areas
    spi_la = 0

    # sum of pixel intensities in the shadow
    spi_s = 0

    # number of pixels in the lit areas
    n_la = 0

    # number of pixels in the shadow
    n_s = 0

    # get sum of pixel intensities in the lit areas
    # and sum of pixel intensities in the shadow
    for i in range(y_cb_cr_img.shape[0]):
        for j in range(y_cb_cr_img.shape[1]):
            if erosion[i, j, 0] == 0 and erosion[i, j, 1] == 0 and erosion[i, j, 2] == 0:
                spi_la = spi_la + y_cb_cr_img[i, j, 0]
                n_la += 1
            else:
                spi_s = spi_s + y_cb_cr_img[i, j, 0]
                n_s += 1

    # get the average pixel intensities in the lit areas
    average_ld = spi_la / n_la

    # get the average pixel intensities in the shadow
    average_le = spi_s / n_s

    # difference of the pixel intensities in the shadow and lit areas
    i_diff = average_ld - average_le

    # get the ratio between average shadow pixels and average lit pixels
    ratio_as_al = average_ld / average_le

    # added these difference
    for i in range(y_cb_cr_img.shape[0]):
        for j in range(y_cb_cr_img.shape[1]):
            if erosion[i, j, 0] == 255 and erosion[i, j, 1] == 255 and erosion[i, j, 2] == 255:
                y_cb_cr_img[i, j] = [y_cb_cr_img[i, j, 0] + i_diff, y_cb_cr_img[i, j, 1] + ratio_as_al,
                                     y_cb_cr_img[i, j, 2] + ratio_as_al]

    # covert the YCbCr image to the BGR image
    return cv2.cvtColor(y_cb_cr_img, cv2.COLOR_YCR_CB2BGR)
def thresholding(img):
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

if  __name__=="__main__":
    test_images = np.array([
        png.read_png_int('./data_road/training/image_2/um_000001.png'),
        png.read_png_int('./data_road/training/image_2/um_000002.png'),
        png.read_png_int('./data_road/training/image_2/um_000003.png'),
        png.read_png_int('./data_road/training/image_2/um_000004.png'),
        png.read_png_int('./data_road/training/image_2/um_000005.png'),
        png.read_png_int('./data_road/training/image_2/um_000006.png'),
    ])
    #show_images(test_images)
    white_yellow_images = list(map(select_white_yellow, test_images))
    #show_images(white_yellow_images)
    gray_images = list(map(convert_gray_scale, white_yellow_images))
    #show_images(gray_images)
    #gray_images = list(map(thresholding, gray_images))

    blurred_images = list(map(lambda image: apply_smoothing(image), gray_images))
    #show_images(blurred_images)
    edge_images = list(map(lambda image: detect_edges(image), blurred_images))
    #show_images(edge_images)
    roi_images = list(map(select_region, edge_images))
    #show_images(roi_images)
    list_of_lines = list(map(hough_lines, roi_images))
    lane_images = []
    for image, lines in zip(test_images, list_of_lines):
        lane_images.append(draw_lane_lines(image, lane_lines(image, lines)))
    show_images(lane_images)

