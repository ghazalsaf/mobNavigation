import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib._png as png
import numpy as np
import cv2


def region_of_interest(img, vertices):
    # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(img)
    # Retrieve the number of color channels of the image.
    channel_count = img.shape[2]
    # Create a match color with the same color channel counts.
    match_mask_color = (1,) * channel_count

    # Fill inside the polygon
    cv2.fillPoly(mask, vertices, match_mask_color)

    # Returning the image only where mask pixels match
    # masked_image = cv2.bitwise_and(img, mask)
    masked_image = img*mask

    return masked_image


def region_of_interest_gray(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 1  # <-- This line altered for grayscale.

    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    # If there are no lines to draw, exit.
    if lines is None:
        return
    # Make a copy of the original image.
    img = np.copy(img)
    # Create a blank image that matches the original in size.
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8,
    )
    # Loop over all lines and draw them on the blank image.
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    # Merge the image with the lines onto the original.
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    # Return the modified image.
    return img


if __name__=="__main__":

    image = png.read_png_int('/Users/junho/PycharmProjects/road_detection/data_road/training/image_2/um_000000.png')
    height, width = image.shape[:2]
    plt.figure()
    plt.imshow(image)

    region_of_interest_vertices = [
        [0, height],
        [width / 3, height/2],
        [width*(2/3), height/2],
        [width, height],
    ]
    #
    # cropped_image = region_of_interest(
    #     image,
    #     [np.array(region_of_interest_vertices, np.int32)],
    # )
    # plt.figure()
    # plt.imshow(cropped_image)
    # plt.show()

    # Convert to grayscale here.
    #gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    ORANGE_MIN = np.array([0, 0, 180],np.uint8)
    ORANGE_MAX = np.array([360, 255, 255],np.uint8)
    blur_img = cv2.GaussianBlur(image,(5,5),0)
    plt.figure()
    plt.imshow(blur_img)
    hsv_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2HSV)

    frame_threshed = cv2.inRange(hsv_img, ORANGE_MIN, ORANGE_MAX)
    plt.figure()
    plt.imshow(frame_threshed)

    # Call Canny Edge Detection here.
    cannyed_image = cv2.Canny(frame_threshed, 50, 200)

    # Moved the cropping operation to the end of the pipeline.
    cropped_image = region_of_interest_gray(
        cannyed_image,
        [np.array(region_of_interest_vertices, np.int32)]
    )

    lines = cv2.HoughLinesP(
        cropped_image,
        rho=6,
        theta=np.pi / 60,
        threshold=160,
        lines=np.array([]),
        minLineLength=0,
        maxLineGap=50
    )

    line_image = draw_lines(image, lines)

    plt.figure()
    plt.imshow(line_image)
    plt.show()
