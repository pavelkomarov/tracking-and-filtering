"""This module contains useful functions to either transform coordinates and angles from one frame to another or return
the transformation matrices that get used outside.
"""

import numpy as np

#   UE4 (from unreal engine)         NWU (north, west, up)            In camera
#      ^ Z, up
#      |                                 Z, up ^                        ^ Z, outward from the lens
#      |                                       |  ^ X, north           /
#      |                                       | /                    /
#      .-------> X, east                       |/                    .-------> X, across image rightward
#     /                        Y, west <-------.                     |
#    /                                                               |
#   v Y, south                                                       v Y, across image downward

UE4_TO_NWU_MAT = np.array([[0, -1, 0], # X = -Y
						   [-1, 0, 0], # Y = -X
						   [0, 0, 1]]) # Z = Z
# Rotation matrix from sensor reference frame to in-camera reference frame. Reference
# https://www.youtube.com/watch?v=OZucG1DY_sY for rotation matricies.
R_cs = np.array([[0, -1, 0], # X_c = -Y_s
				 [0, 0, -1], # Y_c = -Z_s
				 [1, 0, 0]]) # Z_c = X_s


def rotate_rpy_from_ue4_to_nwu(ue4_rpy: np.ndarray) -> np.ndarray:
	"""Function that takes UE4 roll, pitch, yaw and returns phi, theta, psi Euler angles for right-handed NWU frame.

	Why this simple equation works when Euler angles aren't generally simply related is complicated but explained here
	https://math.stackexchange.com/questions/3920406/transform-roll-pitch-yaw-from-one-coordinate-system-to-another/3921872#3921872

	Note that although angles are given in RPY order to be the analog of XYZ order, they're applied in ZYX order.

	:param ue4_rpy: a shape (3,) array of left-handed UE4 RPY in radians
	:return: a shape (3,) array of angles to transform
	"""
	return np.array([ ue4_rpy[0], -ue4_rpy[1], -np.pi/2 - ue4_rpy[2] ])


def xform_kinematics_from_ue4_to_nwu(xyz_kinematics: np.ndarray) -> np.ndarray:
	"""Transform kinematic vectors (location, velocity, acceleration) from the left-handed UE4 frame to the
	right-handed NWU frame.

	:param xyz_kinematics: either a (3,) vector of kinematics (location, velocity, or acceleration) or a (3,N) matrix
		of such vectors stacked together
	:return: the transformed right-handed NWU xyz kinematic vector/array
	"""
	return np.dot(UE4_TO_NWU_MAT, xyz_kinematics)


def get_t4_matrix(rpy_rad: np.ndarray, xyz: np.ndarray) -> np.ndarray:
	"""This function creates a 4x4 transformation matrix which does both rotation and location shift. This will produce
	the transformation matrix that will convert a location from one frame to another.

	Here is an excellent, short explanation of what's going on here: https://www.youtube.com/watch?v=vlb3P7arbkU

	:param rpy_rad: Roll Pitch Yaw of frame B in frame A
	:param xyz: location of the origin of frame B in frame A
	:return: 4x4 transform matrix T_AB that converts homogeneous points in frame B to homogeneous points in frame A
	"""
	# Building up the 3D rotation from the 2D ones. The idea is that a point at some location in the original
	# frame shifts to a new relative location in the new frame based upon how that frame is angled around
	roll, pitch, yaw = rpy_rad[:]
	R_x = R_about_axis(roll, 'x')
	R_y = R_about_axis(pitch, 'y')
	R_z = R_about_axis(yaw, 'z')

	R = R_z.dot(R_y).dot(R_x) # yaw is applied first, then pitch, then roll

	T_AB = np.eye(4)
	T_AB[0:3, 0:3] = R
	T_AB[0:3, 3] = xyz

	return T_AB


def R_about_axis(angle: float, axis: str) -> np.ndarray:
	"""Direct Cosine Matrix (DCM) to rotate single axis

	:param angle: angle being rotated (rad)
	:param axis: 'x', 'y', or 'z' axis for rotation
	:return: DCM for rotation
	"""
	c = np.cos(angle)
	s = np.sin(angle)

	if axis == 'x':
		return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
	elif axis == 'y':
		return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
	elif axis == 'z':
		return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
	else:
		raise Exception('axis out of bounds')
