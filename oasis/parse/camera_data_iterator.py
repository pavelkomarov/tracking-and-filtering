"""Much of this code was lifted from the intel-analytical-engine repo while it was still being developed. We modified
it to ignore iterating through the frames of the simulation for speed. The main purpose of this iterator is to imitate
what might happen in a real system where we're presented with the bounding boxes and their ids from DeepStream over
multiple cameras for a single frame. That architecture hasn't been stamped but that is what we are imagining it doing
at the moment.
"""

from collections import defaultdict
from typing import List, Dict

from . import file_io as fio
from .data_parse_constants import ParsedDataKeys, TrackTypes, IncomingDataKeys
from .scenario_files import ScenarioFiles


class SingleCameraDataManager:

	def __init__(self, objects_fp: str, dpst_trks_fp: str, ground_truth_trks_fp: str):
		"""Load in the deepstream objects and tracks and the ideal tracks from file

		:param objects_fp: the file path where the deepstream objects are stored
		:param dpst_trks_fp: the file path where the deepstream tracks are stored
		:param ground_truth_trks_fp: the file path where the ideal tracks are stored
		"""
		# Load the DeepStream objects into dictionaries indexed by the frame id
		self.dpst_objects_by_frame: Dict[str, List[Dict[str, str]]] = defaultdict(list)
		# there are some legacy data where the deepstream objects were not saved. Ignore those
		if objects_fp:
			for _, dpst_obj in fio.csv_iterator(objects_fp):
				frame_id = dpst_obj[IncomingDataKeys.FRAME_ID]
				self.dpst_objects_by_frame[frame_id].append(dpst_obj)

		# Load the DeepStream tracks into dictionaries indexed by the frame id
		self.dpst_tracks_by_frame: Dict[str, List[Dict[str, str]]] = defaultdict(list)
		for _, track in fio.csv_iterator(dpst_trks_fp):
			frame_id = track[IncomingDataKeys.FRAME_ID]
			self.dpst_tracks_by_frame[frame_id].append(track)

		# Load the ideal tracks into dictionaries indexed by the frame id
		self.ideal_tracks_by_frame: Dict[str, List[Dict[str, str]]] = defaultdict(list)
		for _, track in fio.csv_iterator(ground_truth_trks_fp):
			frame_id = track[IncomingDataKeys.FRAME_ID]
			self.ideal_tracks_by_frame[frame_id].append(track)

	def get_data_from_frame(self, frame_id: str) -> Dict[str, List[Dict[str, str]]]:
		"""Get the camera data for a given frame

		:param frame_id: the desired frame
		:return: the dpst objects, dpst tracks, and ideal tracks for the given frame
		"""
		return {
			TrackTypes.DPST_OBJECTS: self.dpst_objects_by_frame[frame_id],
			TrackTypes.DPST_TRACKS: self.dpst_tracks_by_frame[frame_id],
			TrackTypes.IDEAL_TRACKS: self.ideal_tracks_by_frame[frame_id],
		}


class MultiCameraDataIterator:

	def __init__(self, scenario_files: ScenarioFiles):
		"""The Multi Camera Data Iterator is used to iterate through the track data of each selected camera frame by
		frame.

		:param scenario_files: the object that contains all of the files for the scenario
		"""
		# Get the number of frames that were simulated in the scenario. This can be found in the _time.csv file.
		# We use the data in the file to also count the number of frames from the scenario.
		self.time_data = fio.get_timestamps(scenario_files.time_metadata_fp)
		self.num_frames = len(self.time_data)
		self.frame_id = 0

		# Create the camera data managers for each camera
		self.camera_data_mngr_dict: Dict[str, SingleCameraDataManager] = dict()

		# Loop through all camera id and camera data file path pairs
		for cam_id, cam_data_fps in scenario_files.camera_detections_fps.items():
			# create the camera data manager for this camera and store it
			cam_data_mngr = SingleCameraDataManager(
				cam_data_fps[ScenarioFiles.DEEPSTREAM_OBJECTS],
				cam_data_fps[ScenarioFiles.DEEPSTREAM_TRACKS],
				cam_data_fps[ScenarioFiles.IDEAL_TRACKS]
			)
			self.camera_data_mngr_dict[cam_id] = cam_data_mngr

	def __iter__(self):
		return self

	def __next__(self) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
		"""Iterates through data until the last frame is reached

		:return: this returns something similar to a json object that might get sent to OASIS for processing. We
			include a timestamp for the group of measurements that come in. We also include a frame id, though we won't
			be getting such a value in deployment. We add it for testing purposes only. Then we add all of the camera
			data. The dictionary keys one level in are the camera ids and the values are the dictionaries of tracks.
			The keys of the tracks dictionaries are the DeepStream and ideal track tags. The values for these two keys
			follow the same data structure. They are a list of dictionaries where the dictionary is a row from the
			track file where the keys are the header names of the track csv files. One of these dicts of dicts might
			look like
			{
				'timestamp': 12.345678, # timestamp in seconds
				'frame_id': 189 # frame id
				'camera_data': {
					'00': {
						'dpst_trks': [{row_from_file}, {row_from_file}, ...],
						'ground_truth_trks': [{row_from_file}, {row_from_file}, ...]
					},
					'01': {
						'dpst_trks': [],
						'ground_truth_trks': [{row_from_file}, {row_from_file}, ...]
					},
					...
					'05': {
						'dpst_trks': [{row_from_file}],
						'ground_truth_trks': [{row_from_file}, {row_from_file}, ...]
					}
				}
			}

			And the {row_from_file} might look like:
			{'frame_id': '613',
				'object_confidence': '100',
				'object_id': '252',
				'object_name': 'car',
				'timestamp': '29061193',
				'x_centroid': '365633',
				'xmax': '386966',
				'xmin': '344301',
				'y_centroid': '575136',
				'ymax': '591652',
				'ymin': '558620'}
		"""
		# We stop when they are equal because the frame ids start at index 0.
		if self.frame_id != self.num_frames:
			# Populate the big dictionary to get sent out
			all_camera_data = dict()
			# Grab the timestamp that exists in the _time.csv file generated from CARLA. Convert it into seconds
			all_camera_data[ParsedDataKeys.TIMESTAMP] = float(self.time_data[self.frame_id] / 1e6)
			all_camera_data[ParsedDataKeys.FRAME_ID] = self.frame_id
			# Create the dictionary that will be populated with any camera data
			all_camera_data[ParsedDataKeys.CAMERA_DATA] = dict()

			for camera_id, cam_data_mngr in self.camera_data_mngr_dict.items(): # For each camera, get the data manager
				# Get the data for this camera for this frame id
				data = cam_data_mngr.get_data_from_frame(str(self.frame_id))
				all_camera_data[ParsedDataKeys.CAMERA_DATA][camera_id] = data

			self.frame_id += 1

			return all_camera_data
		else:
			raise StopIteration
