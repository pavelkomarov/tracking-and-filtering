"""This file contains metrics objects, which, given records, compute, save, and display their own analysis info
"""

import numpy

from typing import Dict, Union, List
from .History import RecordDatum

class TruthMetrics:
	"""A class to package up metrics associated to each true record"""
	
	def __init__(self, iden: Union[int, str], track_mapping: List[Union[int, str]],
		true_record: Dict[int, RecordDatum], track_records: Dict[Union[int, str], Dict[int, RecordDatum]],
		obj_id_records: Dict[int, List]):
		"""count (1) how many frames the true target was measured (detected). This is only findable when using IDEAL
		mode, where the association between measurement and truth is known (~max amount we should track), and (2) the
		number of frames during which the true history and the track history both existed (amount we tracked) 

		:param iden: The identifier of the true record we want metrics for
		:param track_mapping: All identifiers of track records mapped to this true record
		:param true_record: The actual record we want metrics for
		:param track_records: The records of all tracks, indexed by identifier
		:param obj_id_records: A record of which object ids were logged at what times through the scenario
		"""
		self.iden = iden # just for printing
		self.n_mapped = len(track_mapping)

		self.detected = self.tracked = 0
		for frame in true_record:
			# Increment `detected` if the true history id is in the list of object ids for this frame.
			if iden in obj_id_records[frame]: self.detected += 1 # only works for ideal measurements
			# Increment `tracked` if an associated track history existed in this frame
			if any([frame in track_records[iden] for iden in track_mapping]): self.tracked += 1

	def __repr__(self) -> str:
		"""string representation of this object

		:returns: true id, number of tracks mapped to this truth, detection coverage count, track coverage count
		"""
		fields = [self.iden, self.n_mapped, self.detected, self.tracked]
		return '\t'.join(map(str, fields))


class TrackMetrics:
	"""A class to package up the tracking performance metrics of just one track record"""

	def __init__(self, iden: Union[int, str], truth_mapping: Union[int, str], track_record: Dict[int, RecordDatum],
		true_record: Dict[int, RecordDatum]):
		"""Compute metrics for a track record with respect to its associated true record

		:param iden: identifier of the Track that created the record
		:param truth_mapping: identifier of the true record the track record was mapped to
		:param track_record: {frame -> RecordDatum}
		:param true_record: {frame -> RecordDatum}
		"""
		self.iden = iden # These are just for printing
		self.truth_mapping = truth_mapping
		self.track_len = len(track_record)
		# Through time, keep:
		self.unique_obj_ids = set() # The unique object ids that were correlated with this track
		self.interframe_first_meas_obj_id_not_equal_count = 0 # The switch metric from MATLAB
		self.frames = [] # the frames during which the track and truth co-existed
		self.first_meas_obj_ids = [] # The first object id from the list of measurements correlated with the track
		self.state_errors = [] # the (xhat - x) for each frame
		self.state_cov_diagonals = [] # The diagonal elements of P for each frame
		self.mean_position_error = 0 # the mean position error between track and truth
		self.mean_velocity_error = 0 # the mean velocity error between track and truth

		# Build up the stats for this track over all frames it existed
		prev_object_ids = [] # to keep track of switches
		for frame,datum in track_record.items():
			# add to the set of measurement object ids correlated with this track
			self.unique_obj_ids.update(datum.object_ids)

			# Update the "switch count" if the object id from this frame's first-listed measurement doesn't match the
			# id from the last-frame-with-measurements' first-listed measurement.
			if len(prev_object_ids) and len(datum.object_ids) and prev_object_ids[0] != datum.object_ids[0]:
				self.interframe_first_meas_obj_id_not_equal_count += 1

			# nothing more to do if the track and truth don't co-exist
			if frame not in true_record:
				# a track can appear before the truth exists, so make sure you save off the object ids
				if len(datum.object_ids): prev_object_ids = datum.object_ids
				continue

			# append some stats/values for this frame
			self.frames.append(frame)
			self.first_meas_obj_ids.append(datum.object_ids[0] if len(datum.object_ids) else prev_object_ids[0])
			self.state_errors.append(datum.x - true_record[frame].x)
			self.state_cov_diagonals.append(datum.P.diagonal())
			self.mean_position_error += numpy.linalg.norm(datum.x[:3] - true_record[frame].x[:3])
			self.mean_velocity_error += numpy.linalg.norm(datum.x[3:] - true_record[frame].x[3:])

			# update the list of previous measurements if there were any measurements this frame, otherwise remember
			if len(datum.object_ids): prev_object_ids = datum.object_ids

		# Divide by the number of summed elements to get average error
		self.mean_position_error /= len(self.frames)
		self.mean_velocity_error /= len(self.frames)

	def __repr__(self) -> str:
		"""string representation of this object

		:returns: track id, truth mapping, object ids, position error, velocity error, switches, length
		"""
		fields = [self.iden, self.truth_mapping, self.unique_obj_ids, f'{self.mean_position_error:0.3f}',
			f'{self.mean_velocity_error:0.3f}', self.interframe_first_meas_obj_id_not_equal_count, self.track_len]
		return '\t'.join(map(str, fields))
		

class ScenarioMetrics:
	"""A class to package up the tracking performance metrics of just one scenario (but many histories)"""

	def __init__(self, iden: Union[int, str], tracks_metrics: Dict[Union[str, int], TrackMetrics]):
		"""Compute metrics for a scenario 

		:param iden: an identifier
		:param tracks_metrics: all the tracks' metrics
		"""
		self.iden = iden

		self.total_frames = self.total_unique_obj_ids_by_track = self.total_switches = 0
		self.mean_position_error = self.mean_velocity_error = 0
		for tm in tracks_metrics: # for each track's metrics
			n_frames = len(tm.frames)
			self.total_frames += n_frames
			self.total_unique_obj_ids_by_track += len(tm.unique_obj_ids)
			self.total_switches += tm.interframe_first_meas_obj_id_not_equal_count
			self.mean_position_error += tm.mean_position_error*n_frames
			self.mean_velocity_error += tm.mean_velocity_error*n_frames
		self.mean_position_error /= (self.total_frames or 1) # so if there are no tracks and therefore total_frames = 0
		self.mean_velocity_error /= (self.total_frames or 1) # this will still divide

	def __repr__(self) -> str:
		"""string representation of this object

		:returns: scenario_id, position error, velocity error, # objects, # switches, track lengths sum
		"""
		fields = [self.iden, f"{self.mean_position_error:0.3f}", f"{self.mean_velocity_error:0.3f}",
			self.total_unique_obj_ids_by_track, self.total_switches, self.total_frames]
		return '\t'.join(map(str, fields))

	@staticmethod
	def print_total(scenarios_metrics: List) -> str: # Should be List[ScenarioMetrics], but have to wait for Python 3.10
		"""A method to compute and print average and total metrics over many scenarios

		:param scenarios_metrics: Metrics for the scenarios we want to factor in
		:returns: position error, velocity error, # objects, # switches, track lengths sum
		"""
		N_frames = N_objs = N_switches = mMPE = mMVE = 0
		for sm in scenarios_metrics: # for each scenario's cumulative metrics
			N_frames += sm.total_frames
			N_objs += sm.total_unique_obj_ids_by_track
			N_switches += sm.total_switches
			mMPE += sm.mean_position_error * sm.total_frames # weight mean of means by number of frames in scenario
			mMVE += sm.mean_velocity_error * sm.total_frames
		mMPE /= (N_frames or 1) # get a weighted average from the weighted sums
		mMVE /= (N_frames or 1) # same trick to make sure this always divides

		return '\t'.join(map(str, ["---", f"{mMPE:0.3f}", f"{mMVE:0.3f}", N_objs, N_switches, N_frames]))
