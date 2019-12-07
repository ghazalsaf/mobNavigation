import os
import numpy as np
import matplotlib.pyplot as plt

from kitti.data import image_shape
from kitti.raw import load_stereo_frame
from kitti.velodyne import (
    load_disparity_points, lin_interp, lstsq_interp, bp_interp, bp_stereo_interp)


def test_load_disparity_points(drive=11):
    xyd = load_disparity_points(drive, color=True, filename="um_000000")
    disp = np.zeros(image_shape, dtype=np.uint8)
    for x, y, d in np.round(xyd):
        disp[int(y), int(x)] = d

    plt.figure(1)
    plt.clf()
    plt.imshow(disp)
    plt.show()


def test_interp(drive=11):
    xyd = load_disparity_points(drive, color=False, filename="um_000000")

    lin_disp = lin_interp(image_shape, xyd)
    lstsq_disp = lstsq_interp(image_shape, xyd, lamb=0.5)
    # lstsq_disp = lstsq_interp(image_shape, points, disps, maxiter=100)

    plt.figure()
    plt.clf()
    plt.subplot(211)
    plt.imshow(lin_disp)
    plt.subplot(212)
    plt.imshow(lstsq_disp)
    plt.show()


if __name__ == '__main__':
    # test_load_disparity_points()
    test_interp()