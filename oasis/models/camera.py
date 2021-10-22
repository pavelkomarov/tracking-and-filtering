"""This module helps us create camera objects from the metadata that gets produced by simulations (or from real cameras
down the road). We have decided that all downstream algorithms should operate on data that is in the right-handed NWU
frame. Data that is stored in this class must be provided in the right-handed NWU frame. We provide a classmethod to
handle the conversion of some legacy files that were stored in UE4. However, from here on, we expect that the data will
have already been converted into NWU when a simulation is created.
"""

import numpy as np

from dataclasses import dataclass
from typing import Dict, Tuple, Union, List

from transform import util as x_util
from tracker.Sensor import Sensor
from models.bounding_box import BoundingBox

from configuration.parameters import WorldProjectionParameters


@dataclass
class NWUCamera(Sensor):
	"""A camera object that has all locations and angles in the right-handed NWU coordinate frame

	:param res_x: the number of pixels in the camera frame width
	:param res_y: the number of pixels in the camera frame height
	:param xyz: a shape (3,) array in meters representing the camera's 3D location
	:param rpy: a shape (3,) array in radians representing the camera's 3D orientation
	:param fov: the camera's field of view in radians
	:param r_hat_pct: (R hat percentage) the percentage of the diagonal length of the bounding box that will be used to
		create our esimate of what the measurement noise covariance R is in the camera frame
	:param fps: the camera's frames per second. Optional
	:param iden: an identifying string. Optional
	:param depth_map: a (res_x, res_y) shape array holding the boresight-depth of each pixel in meters. Optional
	"""
	res_x: int # pixels
	res_y: int # pixels
	xyz: np.ndarray # a (3,) array in meters
	rpy: np.ndarray # a (3,) array in radians
	fov: float # radians
	r_hat_pct: float
	fps: float=None
	iden: str=None
	depth_map: np.ndarray=None # a (res_x, res_y) shape array in meters.

	TOP_CORNER_DEPTH_FUDGE = WorldProjectionParameters.TOP_CORNER_DEPTH_FUDGE

	def __post_init__(self):
		"""This post init method constructs some useful transformation matrixes that may be used downstream like when
		transforming points and angles to and from the camera frame to another frame. This function is called right
		after the __init__ function that is hidden with the @dataclass decorator
		"""
		# Boresight in pixels
		self.boresight = np.array([self.res_x / 2, self.res_y / 2])

		# Transformation matrix from sensor (s) to world (w)
		self.T_ws = x_util.get_t4_matrix(self.rpy, self.xyz) # from "sensor" to "world"
		T_sc = x_util.get_t4_matrix(np.array([-np.pi/2, 0, -np.pi/2]), np.array([0,0,0])) # "camera" frame is flipped
		self.T_wc = self.T_ws.dot(T_sc) # from "camera" to "world"

		# Transformation matrices going the other way
		self.T_sw = np.linalg.inv(self.T_ws) # from "world" to "sensor" (unflipped camera)
		self.T_cw = np.linalg.inv(self.T_wc) # from "world" to "camera"

		# The focal length in pixels is computed with some basic trig. This assumes that we are given the horizontal
		# field of view. Let alpha = 0.5 * fov. Let px be half of the horizontal resolution of the image. Then
		# tan(alpha) = px / focal_length. focal_length which will have units of pixels.
		self.focal_length = (self.res_x / 2) / np.tan(self.fov / 2)

		# Create the camera intrinsic matrix. Lots of good resources here.
		# https://en.wikipedia.org/wiki/Camera_resectioning#Intrinsic_parameters
		# https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/building-basic-perspective-projection-matrix
		# Page 30 of https://courses.cs.washington.edu/courses/cse490r/18wi/lecture_slides/02_07/lecture_02_07.pdf
		# We are assuming that the pixels are square. If ratio of the CCD sensor width to the width of the image is
		# different than the ratio of the CCD sensor height to the image height, then we cannot use the same focal
		# length value in the first two elements on the diagonal.
		self.K = np.array([[self.focal_length, 0, self.res_x / 2], [0, self.focal_length, self.res_y / 2], [0, 0, 1]])


	@classmethod
	def from_NWU_dict(cls, camera_dict: Dict[str, str], rhp: float, depth_map: List[int]):
		"""This class constructor method is used to create a camera object from cam.csv files that were produced using
		this convention https://safexai.atlassian.net/l/c/SzVmsyig. It is assumed that all points and velocities and
		body rotations have already been converted into the NWU frame. We want to convert all mm, microseconds, and
		microradians as they exist in the file to meters, seconds, and radians while they live in this class.

		:param camera_dict: a dictionary where the keys are the header names of the cam.csv files that was
			produced in CARLA post cookoff and marathon and the values are values for one row of that file
		:param rhp: (R hat percentage) the percentage of the diagonal length of the bounding box that will be used to
			create our esimate of what the measurement noise covariance R is in the camera frame
		:param depth_map: a list of boresight-depths in mm of each pixel.
		:return: a NWUCamera instance
		"""
		mm_to_m = 1 / 1e3
		urads_to_rads = 1 / 1e6

		# We expect this to be a 3x1 array for downstream transformations. Convert from mm to meters
		nwu_xyz = mm_to_m * np.array([ int(camera_dict["cam_loc_x"]), int(camera_dict["cam_loc_y"]),
			int(camera_dict["cam_loc_z"]) ])

		# We expect this to be a 3x1 array for downstream transformations. Convert from microradians to radians
		nwu_rpy = urads_to_rads * np.array([ int(camera_dict["cam_rot_roll"]), int(camera_dict["cam_rot_pitch"]),
			int(camera_dict["cam_rot_yaw"]) ])

		# Convert fov from microradians into radians
		fov = int(camera_dict['fov']) * urads_to_rads

		# reshape the depth map, and then convert from mm to m. Needs to be reshaped with Fortran-like index order
		# instead of the default C-like ordering. We load the data in to match the dimensions of the camera image.
		# The simulation group unravels the pixel depths starting from the top left pixel and ending at the top
		# right pixel and then moving down the image.
		depth_map = np.reshape(depth_map,
			(int(camera_dict['cam_resolution_x']), int(camera_dict['cam_resolution_y'])), 'F') * mm_to_m

		return cls(
			iden=camera_dict['camera_id'],
			res_x=int(camera_dict['cam_resolution_x']),
			res_y=int(camera_dict['cam_resolution_y']),
			xyz=nwu_xyz,
			rpy=nwu_rpy,
			fov=fov,
			fps=int(camera_dict['fps']),
			r_hat_pct=rhp,
			depth_map=depth_map
		)


	def h(self, x: np.ndarray, meas_space=True, yhat_only: bool=False) -> \
		Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
		"""A sensor has a nonlinear evaluation function: y = h(x). For the camera problem h: [x; y; z] -> [u; v] is
		given by [u; v] = KR([x; y; z] + T)/z. h returns both the predicted measurement for a point, yhat, and the
		Jacobian of h, H, evaluated at the same point. For a fuller explanation, see:
		https://stats.stackexchange.com/questions/497283/how-to-derive-camera-jacobian

		:param x: the world point at which to evaluate the Jacobian to find H, equivalently mathematically to the
			point around which we're Taylor expanding
		:param meas_space: whether the Tracker is trying to update in Measurement Space or State Space
		:returns: yhat = h(x), the result of sending the point through the camera equations, and H, the system
			Jacobian evaluated at x
		"""
		if meas_space: # we're using y and doing updates in Measurement Space
			xyz_w = x[:3] # get position alone, because yhat and H don't depend on velocity, because cameras don't measure it
			xyz_c = self.T_cw.dot(np.append(xyz_w, 1))[:3] # get the point in camera frame of reference
			K_over_z_c = self.K[:2] / xyz_c[2] # 2x3
			yhat = K_over_z_c.dot(xyz_c) # length 2
			if yhat_only: return yhat

			H = np.zeros((2,6))
			outer_product_over_z_c2 = np.outer(yhat, self.T_cw[2,:3]) / xyz_c[2] # 2x3
			H[:,:3] = K_over_z_c.dot(self.T_cw[:3,:3]) - outer_product_over_z_c2

		else: # we're doing updates in State Space
			H = np.eye(3,6) # Create a 3x6 matrix with 1s on the diagonal elements
			yhat = x[:3] # same thing as H.dot(x) here

		return yhat, H


	def R(self, raw: BoundingBox, meas_space=True) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[None, None]]:
		"""We've decided to make R depend on a param we're naming "rhat percentage". Basically R is filled with values
		from the bounding box and then scaled by some camera-specific percentage.

		:param raw: the BoundingBox that holds the values we want to transform
		:param meas_space: whether the Tracker is trying to update in Measurement Space (True) or State Space (False)
		:return: the measurement or state space centroid and noise covariance matrix estimate or (None, None)
		if projection to world puts measurement behind the camera for our current assumptions
		of where bounding box corners should land in the world
		"""
		R_xx = R_yy = raw.diag_len*self.r_hat_pct # For now, same noise in vertical and horizontal directions
		if meas_space: return raw.centroid, np.array([[R_xx**2, 0], [0, R_yy**2]])

		# If State Space, then we need to project R out in to the 3D world.
		_, bbox_3D = self.sens_to_world(raw)
		if bbox_3D is None: return None, None # possible if bbox intercepts ground behind camera

		# Create a new frame of reference, "o", oriented at the object. .cross() gives new vector orthogonal to inputs.
		toward = bbox_3D.centroid - self.xyz # vec from camera to object centroid in terms of world coordinates
		left = np.cross(np.array([0,0,1]), toward) # leftward, if your yaw is such that you're facing the object
		up = np.cross(toward, left)
		# We get R_wo by stacking vecs that describe o axes *in terms of* world axes together vertically: https://www.youtube.com/watch?v=OZucG1DY_sY
		R_wo = np.column_stack((toward, left, up))
		R_wo = R_wo / np.linalg.norm(R_wo, axis=0) # normalize, because rotation matrix columns have to be unit vectors
		R_oc = np.dot(self.T_cw[:3,:3], R_wo).T # R_co = R_cw*R_wo, and then transpose to get the inverse transform

		# Get the standard deviation in meters for this distance. We have it in pixels, so find conversion factor.
		z_c = np.dot(self.T_cw[:3,:3], toward)[2] # distance along boresight to perpendicular plane on which object lies
		A = z_c / self.K[0, 0] # meters per pixel at that distance (inverse focal length)
		# Stack together directional standard deviation contributions in to a vector.
		r_c = np.array([A*R_xx, A*R_yy, 0]) # R_xx has units pixels; r_c has units meters
		r_o = np.abs(np.dot(R_oc, r_c)) # Express that vector of stddevs in terms of the o frame. Take magnitudes only.
		r_o[0] += WorldProjectionParameters.DEPTH_STANDARD_DEVIATION # Add in the depth contribution along "toward" axis

		# Load those independent, per-dimension stddev contributions in to a diagonal matrix, and rotate to world frame
		sqrtR = R_wo * r_o # equivalent to np.dot(R_wo, np.diag(r_o))
		return bbox_3D.centroid, np.dot(sqrtR, sqrtR.T) # get us back to variance instead of stddev


	def sens_to_world(self, bbox: BoundingBox) -> Union[Tuple[np.ndarray, BoundingBox], Tuple[None, None]]:
		"""Convert a 2D bounding box from the camera frame into a 3d bounding volume in the LVLH NWU frame.

		:param bbox: the BoundingBox that holds the values we want to transform
		:return: The corners of the bbox corners tipped into the world and the BoundingBox that encases those corners
		or (None, None) if box is projected behind camera to satisfy h=0
		"""
		# Guess that there is an extra TOP_CORNER_DEPTH_FUDGE depth for the top corners of a bbox vs the bottom when the camera has a
		# pitch of 0deg. A camera overhead will have a fudge of 0 because the pitch is 90deg, which makes sense, because
		# all corners of the bbox will lie on the ground, which is equidistant from the camera along all four corner rays.
		# TODO: even in the 0 degree case, the corners arent actually equidistant unless the box is in the middle.
		#  We could take position into account?
		fudge = self.TOP_CORNER_DEPTH_FUDGE*np.cos(self.rpy[1])

		# Create the coordinates of the bounding box in the camera frame.
		corners = np.array([ [bbox.xmin, bbox.xmin, bbox.xmax, bbox.xmax],
							 [bbox.ymin, bbox.ymax, bbox.ymax, bbox.ymin] ])

		# We're going to assume that the bottom of the bounding box as seen in the camera projects to points on the
		# ground in the world. We'll also assume that when we project this bounding box, the top of the box projects to
		# points that fall behind the vehicle wrt the camera. So we we add the fudge factor to the top corners of the box.
		ground_depth = self.get_boresight_depth(np.array(corners[:,1]), 0) # second corner is lower left, on ground

		# point is projected behind camera to satisfy h=0
		if not ground_depth:
			print(f"World projection to height 0 is behind camera for the bounding_box {corners} for sensor {self.iden}")
			return None, None

		depths = np.array([ground_depth+fudge, ground_depth, ground_depth, ground_depth+fudge])

		# Project the bbox corners from the camera image to the world, and store them.
		corners_w = np.zeros((3, corners.shape[1]))
		for i in range(corners.shape[1]):
			corners_w[:,i] = self.get_3d(corners[:,i], depths[i])

		xmin, ymin, zmin = np.min(corners_w, axis=1)
		xmax, ymax, zmax = np.max(corners_w, axis=1)

		return corners_w, BoundingBox(xmin, xmax, ymin, ymax, zmin, zmax) # corners of the tipped picture frame


	def __hash__(self):
		"""Most everything in Python is hashable, notable exception being numpy arrays because they're mutable. By a
		similar rationale, dataclasses aren't hashable by default, but we need our Sensors to be.
		https://stackoverflow.com/questions/52390576/how-can-i-make-a-python-dataclass-hashable
		"""
		return hash(id(self))


	def get_3d(self, x_img: np.ndarray, depth: float) -> np.ndarray:
		"""Get the xyz coordinate out in the world for a pixel in the image. The transformation is documented here
		https://safexai.atlassian.net/l/c/XJq4s5da. The images in the link should help to visualize the geometry we are
		using.

		:param x_img: a (2,) array that holds the pixel coordinates [u_img, v_img] to transform into the 3d world
		:param depth: the value along the camera's reference frame z-axis (the depth) to project the point
		:return: a (3,) array that holds the xyz coordinates in the NWU reference frame
		"""
		# Form image point in its homogeneous form
		x_img = np.append(x_img, 1)

		x_img_scaled = x_img * depth # Get the scaled version
		xyz_c = np.linalg.inv(self.K).dot(x_img_scaled) # from image to point in camera reference frame
		xyz_w = self.T_wc.dot(np.append(xyz_c, 1)) # point in camera reference frame to point in NWU reference frame

		return xyz_w[:3]


	def get_boresight_depth(self, x_img: np.ndarray, h_obj: float = 0) -> Union[float, None]:
		"""To project an in-camera point [u_img, v_img] out into the world, we need to know how far along the z-axis of
		the camera reference frame the point needs to move. If we know the height out in the world we want the point to
		exist, we can compute this z value in the camera reference frame such that it satisfies the height requirement.
		This function does the trig necessary to get that z value. We're computing the 'w' value mentioned in this
		document https://safexai.atlassian.net/l/c/XJq4s5da. The images in the link above should help to visualize the
		geometry we are using.

		:param x_img: a (2,) array that holds the pixel coordinates [u_img, v_img] to transform into the 3d world
		:param h_obj: the height in the world that you want the point in the camera to correspond to when transformed.
		:return: the projected point's z component in the camera reference frame or "none" if x_img projects into world
		behind camera to satisfy h = h_obj
		"""
		# Get out the pitch and roll of the camera
		pitch = self.rpy[1]
		roll = self.rpy[0]

		# Raise the world ground plane to be at the height specified by h_obj
		h = self.xyz[2] - h_obj

		# Compute the angle, beta, above or below the boresight at which the v_img point projects out into the world.
		# In the general case, we first need to rotate the point [u_img, v_img] about the boresight for camera's with
		# non-zero roll. See https://safexai.atlassian.net/l/c/emfuaUCx for a visual description.
		R_z = x_util.R_about_axis(roll, 'z')
		x_img_o = R_z.dot(np.append(x_img - self.boresight, 1))
		x_img_b = x_img_o[:-1] + self.boresight

		# We then find the pixel difference between this rotated point and the line that bisects the image
		# horizontally. This is computed as v_img' - 0.5 * res_y where v_img' is the rotated version of v_img. Using
		# the focal length f, compute beta from the trig
		# tan(beta) = d_p / f
		# Again, see https://safexai.atlassian.net/l/c/emfuaUCx for visual description.
		beta = np.arctan((x_img_b[1] - self.boresight[1]) / self.focal_length)

		# The angle from the camera's vertical at which v_img projects out into the world
		gamma = (np.pi / 2) - (pitch + beta)

		# The range from the camera to a point out in space corresponding to height h_obj
		range = h / np.cos(gamma)

		# The Z Component of the range in the camera reference frame. This is 'w'!
		w = range * np.cos(beta)

		# If w is negative, it means the line at angle gamma intersects h behind the camera
		return None if w < 0 else w
