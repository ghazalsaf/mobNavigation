import os

import numpy as np

root_dir = os.path.dirname(os.path.abspath(__file__))

# modified
calib_dir = os.path.join(os.path.dirname(root_dir), 'data_road', 'training', 'calib')
cam_dir = os.path.join(os.path.dirname(root_dir), 'data_road', 'training', 'image_2')
velo_dir = os.path.join(os.path.dirname(root_dir), 'data_road_velodyne', 'training', 'velodyne')

image_shape = 375, 1242


# def get_drive_dir(drive, date='2011_09_26'):
#     return os.path.join(data_dir, date, date + '_drive_%04d_sync' % drive)

def get_velo_dir(drive, filename=None):
    return os.path.join(velo_dir, filename+".bin")

def get_cam_dir(drive, filename=None):
    return os.path.join(cam_dir, filename+".png")

def get_inds(path, ext='.png'):
    inds = [int(os.path.splitext(name)[0]) for name in os.listdir(path)
            if os.path.splitext(name)[1] == ext]
    inds.sort()
    return inds


# def get_calib_dir(date='2011_09_26'):
#     return os.path.join(data_dir, date)


def get_calib_dir(filename=None):
    return os.path.join(calib_dir, filename)


def read_calib_file(path):
    float_chars = set("0123456789.e+- ")

    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    # data[key] = np.array(map(float, value.split(' ')))
                    data[key] = np.array([float(v) for v in value.split(' ')])
                except ValueError:
                    pass  # casting error: data[key] already eq. value, so pass

    return data


def homogeneous_transform(points, transform):
    """
    Parameters
    ----------
    points : (n_points, M) array-like
        The points to transform. If `points` is shape (n_points, M-1), a unit
        homogeneous coordinate will be added to make it (n_points, M).
    transform : (M, N) array-like
        The right-multiplying transformation to apply.
    """
    points = np.asarray(points)
    transform = np.asarray(transform)
    n_points, D = points.shape
    M, N = transform.shape

    # do transformation in homogeneous coordinates
    if D == M - 1:
        points = np.hstack([points, np.ones((n_points, 1), dtype=points.dtype)])
    elif D != M:
        raise ValueError("Number of dimensions of points (%d) does not match"
                         "input dimensions of transform (%d)." % (D, M))

    new_points = np.dot(points, transform)

    # normalize homogeneous coordinates
    new_points = new_points[:, :-1] / new_points[:, [-1]]
    return new_points


def filter_disps(xyd, shape, max_disp=255, return_mask=False):
    x, y, d = xyd.T
    mask = ((x >= 0) & (x <= shape[1] - 1) &
            (y >= 0) & (y <= shape[0] - 1) &
            (d >= 0) & (d <= max_disp))
    xyd = xyd[mask]
    return (xyd, mask) if return_mask else xyd


class Calib(object):
    """Convert between coordinate frames.

    This class loads the calibration data from file, and creates the
    corresponding transformations to convert between various coordinate
    frames.

    Each `get_*` function returns a 3D transformation in homogeneous
    coordinates between two frames. All transformations are right-multiplying,
    and can be applied with `homogeneous_transform`.
    """

    # def __init__(self, date='2011_09_26', color=False):
    #     self.calib_dir = get_calib_dir(date=date)
    #     self.imu2velo = read_calib_file(
    #         os.path.join(self.calib_dir, "calib_imu_to_velo.txt"))
    #     self.velo2cam = read_calib_file(
    #         os.path.join(self.calib_dir, "calib_velo_to_cam.txt"))
    #     self.cam2cam = read_calib_file(
    #         os.path.join(self.calib_dir, "calib_cam_to_cam.txt"))
    #     self.color = color

    def __init__(self, filename=None, color=False):
        self.calib_dir = get_calib_dir(filename=filename)
        self.imu2velo = read_calib_file(self.calib_dir+".txt")
        self.velo2cam = read_calib_file(self.calib_dir+".txt")
        self.cam2cam = read_calib_file(self.calib_dir+".txt")
        self.color = color

    def get_imu2velo(self):
        RT_imu2velo = np.eye(4)
        RT_imu2velo[:3, :4] = self.imu2velo['Tr_imu_to_velo'].reshape(3, 4)

        return RT_imu2velo.T

    def get_velo2rect(self):

        RT_velo2cam = np.eye(4)
        RT_velo2cam[:3, :4] = self.velo2cam['Tr_velo_to_cam'].reshape(3, 4)

        R0_rect = np.eye(4)
        R0_rect[:3, :3] = self.cam2cam['R0_rect'].reshape(3, 3)

        RT_velo2rect = np.dot(R0_rect, RT_velo2cam)
        return RT_velo2rect.T

    def get_rect2disp(self):
        cam0, cam1 = (0, 1) if not self.color else (2, 3)
        P_rect0 = self.cam2cam['P%1d' % cam0].reshape(3, 4)
        P_rect1 = self.cam2cam['P%1d' % cam1].reshape(3, 4)

        P0, P1, P2 = P_rect0
        Q0, Q1, Q2 = P_rect1
        # assert np.array_equal(P1, Q1), "\n%s\n%s" % (P1, Q1)
        # assert np.array_equal(P2, Q2), "\n%s\n%s" % (P2, Q2)

        # create disp transform
        T = np.array([P0, P1, P0 - Q0, P2])
        return T.T

    def get_imu2rect(self):
        return np.dot(self.get_imu2velo(), self.get_velo2rect())

    def get_imu2disp(self):
        return np.dot(self.get_imu2rect(), self.get_rect2disp())

    def get_velo2disp(self):
        return np.dot(self.get_velo2rect(), self.get_rect2disp())

    def get_disp2rect(self):
        return np.linalg.inv(self.get_rect2disp())

    def get_disp2imu(self):
        return np.linalg.inv(self.get_imu2disp())

    def rect2disp(self, points):
        return homogeneous_transform(points, self.get_rect2disp())

    def disp2rect(self, xyd):
        return homogeneous_transform(xyd, self.get_disp2rect())

    def velo2rect(self, points):
        return homogeneous_transform(points, self.get_velo2rect())

    def velo2disp(self, points):
        return homogeneous_transform(points, self.get_velo2disp())

    def imu2rect(self, points):
        return homogeneous_transform(points, self.get_imu2rect())

    def rect2imu(self, points):
        return homogeneous_transform(points, self.get_rect2imu())

    def filter_disps(self, xyd, max_disp=255, return_mask=False):
        return filter_disps(
            xyd, image_shape, max_disp=max_disp, return_mask=return_mask)


# TODO: functions to automatically download data
