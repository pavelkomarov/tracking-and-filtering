# To run: $ python -m pytest -s

import os
import zipfile

from .. import file_io as fio

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), '../../test_data/Town10HD_Int0_228_deepstream_detections')


def unzip():
	# unzip so that the extracted directory lives as a sibling directory of the scenario data
	with zipfile.ZipFile(os.path.join(TEST_DATA_DIR + '.zip'), 'r') as zip_f: zip_f.extractall(TEST_DATA_DIR)


def test_csv_iterator_number_of_yields():
	if not os.path.isdir(TEST_DATA_DIR): unzip()
	test_csv_file_path = os.path.join(TEST_DATA_DIR, 'Town10HD_Int0_228_camera_metadata.csv')

	num_yields = 0
	for _, _ in fio.csv_iterator(test_csv_file_path):
		num_yields += 1

	# Ensure that the iterator only yields the same number of rows as there is data. This excludes the header row. For
	# this test, we know that the _cam.csv file has 5 data rows.
	assert num_yields == 5


def test_csv_iterator_dict_keys():
	"""This test checks that the headers of the data files are made into the keys of the dictionaries that come out of
	the csv_iterator. We could have tested this with any of the data files, but chose the camera one since it is the
	shortest file."""
	if not os.path.isdir(TEST_DATA_DIR): unzip()
	test_csv_file_path = os.path.join(TEST_DATA_DIR, 'Town10HD_Int0_228_camera_metadata.csv')

	# These are the keys that should be in each dictionary that is returned for this specific file
	true_keys = set([
		'camera_id',
		'cam_loc_x',
		'cam_loc_y',
		'cam_loc_z',
		'cam_rot_roll',
		'cam_rot_pitch',
		'cam_rot_yaw',
		'cam_resolution_x',
		'cam_resolution_y',
		'fov',
		'fps'
	])

	# Ensure the keys of the data dictionary that are yielded are correct
	assert all([true_keys == set(row_dict.keys()) for _, row_dict in fio.csv_iterator(test_csv_file_path)])
