# Run with: $ python -m pytest -s

import os
import itertools

from ..data_parse_constants import TrackTypes, ParsedDataKeys, IncomingDataKeys
from ..vehicle_data_iterator import VehicleDataIterator
from ..camera_data_iterator import MultiCameraDataIterator
from ..scenario_files import ScenarioFiles


TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), '../../test_data/Town10HD_Int0_228_deepstream_detections')


def test_multicam_iterator_camera_ids_in_data_dict():
	"""Make sure that the data that gets yielded by the data iterator is from exactly the cameras the user specified.
	Camera data shouldn't be added or missing.
	"""
	# All camera ids in the file
	all_camera_ids = ['0','1','2','3','4']

	# Loop through every combination of cameras that could be used from one to all six cameras
	for r in range(len(all_camera_ids) + 1):

		# Returns a list, but with only one tuple. Get that one tuple with [0]
		set_of_cams_to_use = list(itertools.combinations(all_camera_ids, r))[0]
		if len(set_of_cams_to_use) == 0:
			continue

		# Get all of the file path names loaded into memory
		scenario_files = ScenarioFiles(TEST_DATA_DIR, set_of_cams_to_use)

		# Create the data iterator
		mcdi = MultiCameraDataIterator(scenario_files)

		# Ensure that the keys of the returned data dictionary are exactly the cameras specified to use when the class was
		# constructed.
		assert all([set(set_of_cams_to_use) == set(data[ParsedDataKeys.CAMERA_DATA].keys()) for data in mcdi])


def test_multicam_iterator_same_frame_id_in_data_dict():
	"""This test checks if the track data that come out of the camera data iterator all have the same frame id."""
	# Only the data from these cameras will be iterated over
	cameras_to_use = ['0','1','2','3','4']

	# Get all of the file path names loaded into memory
	scenario_files = ScenarioFiles(TEST_DATA_DIR, cameras_to_use)

	# Create the data iterator
	mcdi = MultiCameraDataIterator(scenario_files)

	# Ensure that the frame id is the same for every new data dictionary that is yielded from the multicamera data
	# iterator. It should be the same for the DeepStream tracks and the ideal tracks
	for data in mcdi:
		# a place to store the frame ids that come in with the data dictionary
		frame_id_set = set()

		# Loop through the data that comes in for each camera
		for cam_id in cameras_to_use:
			# Check both the deepstream and ideal tracks
			for track_type in TrackTypes:
				# Loop through all data dictionaries (a dictionary corresponds to a row in the _ideal.csv or _trk.csv
				# file)
				for trk_row_dict in data[ParsedDataKeys.CAMERA_DATA][cam_id][track_type]:
					frame_id_set.add(trk_row_dict[ParsedDataKeys.FRAME_ID])

		# There should only be one frame id in the set if the iterator is working correctly. There could also be 0 if
		# none of the cameras had objects in view for the frame
		assert len(frame_id_set) <= 1


def test_vehicle_iterator_same_frame_id_in_data_dict():
	"""This test checks that the list of NWUVehicles that come out of the vehicle data iterator all have the same frame id."""

	# Get all of the file path names loaded into memory
	scenario_files = ScenarioFiles(TEST_DATA_DIR, ['0'])

	# Create the data iterator
	veh_iter = VehicleDataIterator(scenario_files)

	# Ensure that the frame id is the same for every new data dictionary that is yielded from the vehicle data
	# iterator.
	for vehicles_data in veh_iter:
		# a place to store the frame ids that come in with the data dictionary
		frame_id_set = set()

		for veh_data in vehicles_data:
			frame_id_set.add(veh_data[IncomingDataKeys.FRAME_COUNT])

		# It's possible that there may not be vehicles in a given frame, so account for the case where it's 0
		assert len(frame_id_set) <= 1
