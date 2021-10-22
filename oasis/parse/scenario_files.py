import os
import glob
import zipfile

from typing import List

class ScenarioFiles:
	"""For convenience, we wanted a clean way of loading in the file names that get produced by in the deepstream
	crockpot. This class makes these accessible as fields.

	This class only handles data produced for the TF4 marathon.
	"""
	DEEPSTREAM_OBJECTS = 'dpst_objects'
	DEEPSTREAM_TRACKS = 'dpst_tracks'
	IDEAL_TRACKS = 'ideal_tracks'

	def __init__(self, scenario_path: str, cameras_to_use: List[str]) -> None:
		"""Load in the file names for the files that get generated from the simulation and deepstream pipelines

		:param scenario_path: the path to the directory that holds all of the files
			(e.g. /Users/pete/Desktop/tf4_marathon_lite/Town10HD_Int0_228_deepstream_detections). If the directory does
			not exist, this will attempt to find a .zip file with the same name as the directory and unzip the
			contents. If the zip doesn't exist, execution stops.
		:param cameras_to_use: For single scenario, select the combination of cameras desired. Strings passed in should
			match the camera ids that get produced (i.e. {'0', '1', '2', '3', '4', '5'}). Use any subset of that set.
			Alternatively, if no camera ids are provided, all of the camera files will be produced.
		"""
		self._unzip_if_not(scenario_path)
		
		self.scenario_name = os.path.basename(scenario_path).replace('_deepstream_detections', '') # e.g. 'Town10HD_Int0_228'
		self.intersection_name = '_'.join(self.scenario_name.split('_')[:2]) # e.g. 'Town10HD_Int0'

		# Get the scenario-wide files
		self.ground_truth_fp = os.path.join(scenario_path, f'{self.scenario_name}_ground_truth.csv')
		self.camera_metadata_fp = os.path.join(scenario_path, f'{self.scenario_name}_camera_metadata.csv')
		self.time_metadata_fp = os.path.join(scenario_path, f'{self.scenario_name}_time_metadata.csv')
		self.vehicle_frame_metadata_fp = os.path.join(scenario_path, f'{self.scenario_name}_vehicle_frame_metadata.csv')
		self.vehicle_global_metadata_fp = os.path.join(scenario_path, f'{self.scenario_name}_vehicle_global_metadata.csv')
		# Depth map files need to be handled a little differently because we store a set of depth maps per intersection
		# instead of in each scenario. The depth map files are 15MB each, so we're keeping things as light as possible.
		self.depths_dir = os.path.join(os.path.dirname(scenario_path), f'{self.intersection_name}_depths')
		self._unzip_if_not(self.depths_dir)

		# get the files for each camera as {cam id -> {detection type -> detections}} and {cam id -> depth map}
		self.camera_detections_fps = dict()
		self.camera_depths_fps = dict()
		for x in os.listdir(scenario_path):
			cam_path = os.path.join(scenario_path, x)
			if os.path.isdir(cam_path): # found a camera folder
				cam_id = cam_path[-1]
				if cameras_to_use is None or cam_id in cameras_to_use:
					base = os.path.join(cam_path, x)
					self.camera_detections_fps[cam_id] = { self.DEEPSTREAM_OBJECTS: f'{base}_dpst_objects.csv',
														   self.DEEPSTREAM_TRACKS: f'{base}_dpst_trks.csv',
														   self.IDEAL_TRACKS: f'{base}_ground_truth_trks.csv' }
					self.camera_depths_fps[cam_id] = os.path.join(self.depths_dir,
						f'{self.intersection_name}_camera{cam_id}_depth.csv')

		self._check_if_files_exist()

	def _unzip_if_not(self, path: str):
		"""A little helper to unzip things if they haven't been. Deletes the zip.

		:param path: the name of the thing to unzip
		"""
		if not os.path.isdir(path):
			assert os.path.isfile(path + '.zip'), f'{path}.zip does not exist. Download it from s3://banjo-sandbox-intel/pete/'
			with zipfile.ZipFile(path + '.zip', 'r') as zip_f: zip_f.extractall(path)


	def _check_if_files_exist(self):
		"""Verify that all of the files for the scenario exist"""
		for k, fp in self.__dict__.items():
			# Ignore the fields that aren't file paths
			if 'fp' not in k: continue

			# Deal with the camera detections file paths
			if 'fps' in k:
				for v in fp.values(): # for all cameras' mapped-to strings
					if isinstance(v, dict): # each has a dict of file pointers
						for cam_data_fp in v.values(): # make sure all three exist
							assert os.path.isfile(cam_data_fp), f"Can't find the camera data files. Are you using the right camera ids in the config?"
					else: # the single string is for the camera depths
						assert os.path.isfile(v)
			else:
				assert os.path.isfile(fp)
