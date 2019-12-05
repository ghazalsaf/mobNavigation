import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib._png as png
import numpy as np
import cv2

np.seterr(divide='ignore', invalid='ignore')


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

    image = png.read_png_int('/Users/jeong-yeonji/mobNavigation/kitti/data_road/training/image_2/um_000012.png')
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

    ORANGE_MIN = np.array([0, 0, 0],np.uint8)
    ORANGE_MAX = np.array([360, 255, 80],np.uint8)

    # blur_img = cv2.GaussianBlur(image,(9,9),0)
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    frame_threshed = cv2.inRange(hsv_img, ORANGE_MIN, ORANGE_MAX)
    # cv2.imshow("hsv", frame_threshed)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    back_threshed = cv2.bitwise_not(frame_threshed)
    back_region = cv2.bitwise_and(image, image, mask=back_threshed)

    shad_region = cv2.bitwise_and(image, image, mask=frame_threshed)
    for i in range(3):
        shad_region[:,:,i] = cv2.equalizeHist(shad_region[:,:,i])
    cv2.imshow("shadow", shad_region)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    nonshad_img = cv2.bitwise_or(shad_region, back_region)
    cv2.imshow("back", nonshad_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if frame_threshed[i,j] != 0:
                hsv_img[i,j,:] *= 3

    rgb_image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

    # cv2.imshow("image", rgb_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Call Canny Edge Detection here.
    blur_img = cv2.GaussianBlur(nonshad_img,(9,9),0)
    cannyed_image = cv2.Canny(blur_img, 50, 200)

    # Moved the cropping operation to the end of the pipeline.
    cropped_image = region_of_interest_gray(
        cannyed_image,
        [np.array(region_of_interest_vertices, np.int32)]
    )

    lines = cv2.HoughLinesP(
        cropped_image,
        rho=6,
        theta=np.pi / 60,
        threshold=10,
        lines=np.array([]),
        minLineLength=0,
        maxLineGap=50
    )

    line_image = draw_lines(image, lines)

    plt.figure()
    plt.imshow(line_image)
    plt.show()
