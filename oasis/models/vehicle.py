"""This module contains code to handle the ground truth vehicle locations that are generated from simulations. Data
that is passed into this class on construction must be provided in the right-handed NWU frame. We provide a classmethod
to handle the conversion of some legacy files that were stored in UE4. However, from here on, we expect that the data
will have already been converted into NWU when a simulation is created.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import numpy
import numpy as np

from tracker.Target import Target


@dataclass
class NWUVehicle(Target):
	"""A vehicle object that has all locations, speeds, and angles in the right-handed NWU coordinate frame

	:param qhat: amount of noise to add to the velocity component in the x and y direction
	:param car_id:
	:param frame_count: the frame in which this observation was made
	:param xyz: a 3x1 array in meters representing the vehicle's 3D location
	:param xyz_dot: a 3x1 array in meters/second representing the vehicle's 3D speed
	:param rpy: a 3x1 array in radians representing the vehicle's 3D orientation
	:param extent: the extent of the vehicle's 3D volume box in meters (measured from the center to edge)
	:param object_confidence: the confidence of this vehicle's detection (in [0,1] from YOLO)
	:param timestamp: the timestamp at which this observation was made in seconds
	:param vehicle_type: the vehicle type assigned by the object detector (or from ground truth)
	"""
	qhat: float
	car_id: int=None
	frame_count: int=None
	xyz: np.ndarray=None # a 3x1 array in meters
	xyz_dot: np.ndarray=None # a 3x1 array in m/s
	rpy: np.ndarray=None # a 3x1 array in radians
	extent: np.ndarray=None # a 3x1 array in meters
	object_confidence: float=None
	timestamp: int=None # in seconds
	vehicle_type: str=None


	def __post_init__(self):
		"""This is where to do any complicated initializations in a dataclass
		"""
		self.Q_ = np.zeros((6,6))
		self.Q_[3:,3:] = np.diag([1,1,0.01])*self.qhat**2 # This says "There is noise in the velocity components, 100x
			# more in x and y than z" because unfortunately we don't have flying cars


	@classmethod
	def from_NWU_dict(cls, vehicle_dict: Dict[str, str], timestamp: int, vehicle_type: str, qhat: float):
		"""This class constructor method is used to create a vehicle object from data in vehicle_frame.csv files that
		were produced using this convention https://safexai.atlassian.net/l/c/SzVmsyig. It is assumed that all points,
		velocities, and body rotations have already been converted into the NWU frame. We want to convert all mm,
		microseconds, and microradians as they exist in the file to meters, seconds, and radians while they live in
		this class.

		:param nwu_camera_dict: a dictionary where the keys are the header names of the vehicle_frame.csv files that was
			produced in CARLA post cookoff and marathon and the values are values for one row of that file
		:return: a NWUVehicle instance
		"""
		mm_to_m = 1 / 1e3
		urads_to_rads = 1 / 1e6

		nwu_xyz = mm_to_m * np.array( # convert from mm to meters
			[float(vehicle_dict[x]) for x in ['car_location_x', 'car_location_y', 'car_location_z']])
		nwu_xyz_dot = mm_to_m * np.array( # convert from mm/second to meters/second
			[float(vehicle_dict[x]) for x in ['car_velocity_x', 'car_velocity_y', 'car_velocity_z']])
		nwu_rpy = urads_to_rads * np.array( # convert from microradians to radians
			[float(vehicle_dict[x]) for x in ['car_rotation_roll', 'car_rotation_pitch', 'car_rotation_yaw']])
		extent = mm_to_m * np.array( # convert from mm to meters
			[float(vehicle_dict[x]) for x in ['car_extent_x', 'car_extent_y', 'car_extent_z']])

		# Matching a value that comes out of DeepStream which is the detection confidence from YOLO. We hardcode a
		# confidence of 1.0 because we perfectly know the object type.
		object_confidence = 1.0

		# Convert from milliseconds to seconds
		timestamp = float(timestamp / 1e6)

		return cls(
			car_id=int(vehicle_dict['car_id']),
			frame_count=int(vehicle_dict['frame_count']),
			xyz=nwu_xyz,
			xyz_dot=nwu_xyz_dot,
			rpy=nwu_rpy,
			extent=extent,
			object_confidence=object_confidence,
			timestamp=timestamp,
			vehicle_type=vehicle_type, # derived from the car_type in vehicle_global.csv file
			qhat=qhat
		)


	def f(self, *args) -> Tuple[np.ndarray, np.ndarray]:
		"""A target has some kind of evolution dynamics. For vehicles we're assuming very simple linear dynamics in the
		continuous time: xdot = F x. This function does NOT take x as a param, because it's not necessary to propagate
		anything nor evaluate the Jacobian at a point.

		:returns: state evolution matrix encoding (linear) dynamics of the vehicle
		"""
		# F is from the ODE, dx(t) = F x(t)dt, so no 1s on diagonal; things will keep their value naturally by
		F = np.zeros((6,6))	# applying no diffeq perturbation
		F[:3,3:] = np.diag([1,1,1]) # This is saying "velocities are derivatives of positions"
		return None, F # return None for xhat_prop, because this is a linear scenario


	def Q(self) -> np.ndarray:
		"""We're choosing a Q that allows for noise in the velocity predictions, because the model dynamics from f()
		only explicitly specify Newtonian inertial motion and don't account for accelerations.
		"""
		return self.Q_

	def plot_repr(self) -> numpy.ndarray:
		"""Build the rotated ground truth vehicle bounding box in an overhead view

		:return: coordinate array of element
		"""
		yaw = self.rpy[2] # the angle about the z or up axis
		R = numpy.array([[numpy.cos(yaw), -numpy.sin(yaw)],
						 [numpy.sin(yaw), numpy.cos(yaw)]]) # rotation matrix

		# the bounding box corners centered at the origin. In CARLA, extents are /2, so full car length, width, height
		# are twice extent
		bbox = numpy.array([[-self.extent[0], self.extent[0], self.extent[0], -self.extent[0], -self.extent[0]],
							[-self.extent[1], -self.extent[1], self.extent[1], self.extent[1], -self.extent[1]]])

		return numpy.dot(R, bbox) + self.xyz[:2, numpy.newaxis] # rotate and translate the bbox
