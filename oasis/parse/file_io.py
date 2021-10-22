import csv
from typing import Dict, List, Union, Tuple

from models.camera import NWUCamera
from parse.data_parse_constants import IncomingDataKeys, TargetTypes
from parse.scenario_files import ScenarioFiles


def load_depth_map(fp: str) -> List[int]:
	"""Load in the camera depth values from the depth.csv files produced in CARLA.

	:param fp: the path to the file containing the depths for a camera
	:return: the depths as ints. For a 1920x1080 camera, the first 1920 elements correspond to the top 1920 pixels of
	an image
	"""
	# load in all the values, but skip the first line, because it's the header
	with open(fp, 'r') as f: return list(map(int, f.readlines()[1:]))


def load_NWUCameras_from_NWU_metadata(scenario_files: ScenarioFiles, rhp: float) -> Dict[str, NWUCamera]:
	"""Load the camera metadata in camera.csv where the values were stored in NWU frame values as specified here
	https://safexai.atlassian.net/l/c/SzVmsyig.

	:param scenario_files: the object that contains all of the files for the scenario
	:param rhp: (R hat percentage) the percentage of the diagonal length of the bounding box that will be used to
		create our esimate of what the measurement noise covariance R is in the camera frame
	:return: A dictionary of NWUCameras with the camera id as the key
	"""
	print('Loading camera metadata and depth maps')
	NWUCamera_dict = dict()
	# Loop through all camera metadata stored in the file
	for _, raw_cam_metadata_dict in csv_iterator(scenario_files.camera_metadata_fp):
		cam_id = raw_cam_metadata_dict[IncomingDataKeys.CAMERA_ID]
		if cam_id in scenario_files.camera_detections_fps:
			depth_map = load_depth_map(scenario_files.camera_depths_fps[cam_id])
			NWUCamera_dict[cam_id] = NWUCamera.from_NWU_dict(raw_cam_metadata_dict, rhp, depth_map)
	
	return NWUCamera_dict


def vehicle_frames_from_NWU_metadata_iter(veh_frame_filename: str, veh_global_filename: str, time_filename: str) -> Dict[str, str]:
	"""Iterate through the rows of the vehicle_frame.csv file and yield that data as a dictionary. The time.csv and the
	vehicle_global.csv files also need to be passed in so that we know the vehicle type and the timestamp associated to
	the frame. Using this function assumes that the data stored in vehicle_frame.csv was stored
	in the NWU coordinate frame.

	:param veh_frame_filename: the file path of the vehicle_frame.csv (assumes data is in NWU frame)
	:param veh_global_filename: the file path of the vehicle_global.csv
	:param time_tp: the file path of the time.csv
	:return: a dictionary with all of the data for the vehicle across the vehicle_frame.csv, vehicle_global.csv, and
		time.csv files
	"""
	# Get the static characteristics/properties for each file seen during the simulation
	glbl_vehicle_dict = get_vehicle_types(veh_global_filename)

	# Get the timestamps for each frame count during the simulation
	time_dict = get_timestamps(time_filename)

	for _, vehicle_frame in csv_iterator(veh_frame_filename):
		# Get out the frame and car id so that we can query the dictionary for the timestamp and the vehicle
		# characteristics/properties for this row of data
		frame_count = int(vehicle_frame[IncomingDataKeys.FRAME_COUNT])
		car_id = int(vehicle_frame[IncomingDataKeys.CAR_ID])
		vehicle_type = glbl_vehicle_dict[car_id]
		timestamp = time_dict[frame_count]

		# Add the timestamp (as an str for consistent value type) and the vehicle type to the vehicle data dictionary
		# and yield it
		vehicle_frame[IncomingDataKeys.TIMESTAMP] = str(timestamp)
		vehicle_frame[IncomingDataKeys.VEHICLE_TYPE] = vehicle_type
		yield vehicle_frame


def write_list_of_dicts_to_csv(fp: str, xformed_data: List[Dict[Union[int, float, str], Union[int, float, str]]]) -> None:
	"""Function to write a list of dictionaries to csv.

	:param fp: the file path to where to save the data
	:param xformed_data: a list of the data dictionaries to be written to file. The keys of each dictionary in the list
		must have the same keys.
	"""
	# Can't save a file if there is there is no data to write
	if len(xformed_data) == 0:
		return

	with open(fp, 'w', newline='') as file:
		fieldnames = list(xformed_data[0].keys())
		writer = csv.DictWriter(file, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(xformed_data)


def csv_iterator(filename: str) -> Tuple[int, Dict[str, str]]:
	"""An iterator to yield one row of a csv file as a dictionary

	:param filename: the file path
	:return: a dictionary where the keys are the header names of the csv file and the values are the data in the row of
		the file
	"""
	with open(filename, 'r', newline='') as csvfile:
		reader = csv.DictReader(csvfile, skipinitialspace=True)
		for row_idx, data_dict in enumerate(reader):
			# Do not include column headers that were not named. This happens in some of the vehicle_global.csv files
			# for the vehicle color. These column headers are read in as None, so that is the key to skip them.
			out_dict = {k: v for k, v in data_dict.items() if k is not None}
			yield (row_idx, out_dict)


def get_carla_vehicle_type(car_type) -> str:
	"""This does the lookup from the `car_type` column of vehicle_global.csv to a simple vehicle type of car, bicycle,
	or motorcycle.

	:param car_type: the car type as recorded in the csv file
	:return: the vehicle type from the lookup
	"""
	bike_types = {"bh", "gazelle", "diamondback"}
	motorcycle_types = {"yamaha", "harley-davidson", "kawasaki"}

	# The `car_type` fields look like vehicle.gazelle.omafiets. We split the string
	# to get the component pieces
	type_parts = car_type.split(".")

	# The differentiator is the second descriptor in the `car_type` column. This is why we hardcode the index 1
	vehicle_type = type_parts[1]
	if vehicle_type in bike_types:
		return TargetTypes.BICYCLE
	elif vehicle_type in motorcycle_types:
		return TargetTypes.MOTORCYCLE
	else:
		return TargetTypes.CAR


def get_vehicle_types(vehicle_global_filename: str) -> Dict[int, str]:
	"""For each car id, get the vehicle type (car, bicycle, motorcycle)

	:param vehicle_global_filename: the file path of the vehicle_global.csv
	:return: a dictionary where the keys are the car id and the values are the vehicle type
	"""
	glbl_vehicle_dict = dict()
	for _, vehicle_details in csv_iterator(vehicle_global_filename):
		car_id = int(vehicle_details[IncomingDataKeys.CAR_ID])
		glbl_vehicle_dict[car_id] = get_carla_vehicle_type(vehicle_details[IncomingDataKeys.CAR_TYPE])

	return glbl_vehicle_dict


def get_timestamps(time_filename) -> Dict[int, int]:
	"""For each frame count, get the associated timestamp

	:param time_tp: the file path of the time.csv
	:return: a dictionary where the keys are the frame count and the values are the timestamps
	"""
	time_dict = dict()
	for _, time_data in csv_iterator(time_filename):
		frame_count = int(time_data[IncomingDataKeys.FRAME_COUNT])

		# The timestamps are stored as microseconds
		time_dict[frame_count] = int(time_data[IncomingDataKeys.TIMESTAMP])

	return time_dict
