from typing import Dict, List
from collections import defaultdict

import parse.file_io as fio
from parse.scenario_files import ScenarioFiles

from parse.data_parse_constants import IncomingDataKeys


class VehicleDataIterator:

	def __init__(self, scenario_files: ScenarioFiles) -> None:
		"""The vehicle data iterator reads in all of the data files needed to produce a NWUVehicle per line of the
		_vehicle_frame.csv file from a simulation and then pack all vehicles together that appeared in the same frame.


		:param scenario_files: the object that contains all of the files for the scenario
		"""
		# Create a place to store the vehicle data from file that are in the same frame
		self.vehicle_dynamics_by_frame: Dict[int, List[Dict[str, str]]] = defaultdict(list)

		# Iterator that returns a dictionary with the vehicle data per line of _vehicle_frame.csv. The timestamp and
		# the vehicle type also gets passed back in the dictionary from _vehicle_global.csv and _time.csv
		veh_iter = fio.vehicle_frames_from_NWU_metadata_iter(
			scenario_files.vehicle_frame_metadata_fp,
			scenario_files.vehicle_global_metadata_fp,
			scenario_files.time_metadata_fp
		)

		# keep track of the last frame a vehicle appeared
		self.last_frame_id = 0

		# Load the DeepStream and ideal track data into dictionaries with the frame id as the key
		for nwu_vehs_data in veh_iter:
			frame_id = int(nwu_vehs_data[IncomingDataKeys.FRAME_COUNT])
			self.vehicle_dynamics_by_frame[frame_id].append(nwu_vehs_data)
			self.last_frame_id = max(self.last_frame_id, frame_id)

		self._frame_number = 0

	def __iter__(self):
		return self

	def __next__(self) -> List[Dict[str, str]]:
		"""Iterates one time-step

		:return: All of the vehicle data for vehicles that appeared in the same frame
		"""
		if self._frame_number <= self.last_frame_id:
			data = self.vehicle_dynamics_by_frame[self._frame_number]

			self._frame_number += 1
			return data
		else:
			raise StopIteration
