import numpy

from typing import List, Dict, Union, NamedTuple
from collections import defaultdict

from models.vehicle import NWUVehicle
from tracker.Track import Track
from tracker.Measurement import Measurement


# Any data you want to store as part of the track history to then be used for analysis
RecordDatum = NamedTuple('RecordDatum', [('x', numpy.ndarray), ('P', numpy.ndarray), ('object_ids', List[Union[int, str]])])


class History:
	"""Each track and truth will need need their own history. We also need to keep track of the object ids that came in
	from the measurements. This object keeps all the records needed for a single scenario.
	"""

	def __init__(self, iden: Union[int, str]):
		"""Create a new history

		:param iden: useful identifier for reading off which scenario this history belongs to
		"""
		self.iden = iden
		# {history id -> {frame -> RecordDatum}}
		self.track_records: Dict[Union[int, str], Dict[int, RecordDatum]] = defaultdict(dict)
		self.true_records: Dict[Union[int, str], Dict[int, RecordDatum]] = defaultdict(dict)
		# {frame -> [all object ids detected in that batch]}
		self.obj_id_records = defaultdict(list)

	def update(self, nwu_vehicles: List[NWUVehicle], tracks: List[Track], measurements: List[Measurement],
		mapping: Dict[Track, List[Measurement]], frame: int):
		"""Add in the latest data into the records.

		:param nwu_vehicles: All truth vehicles present
		:param tracks: All existing tracks
		:param measurements: All measurements that came in from the sensors
		:param mapping: The mapping of track -> [Measurements] produced from the Correlator
		:param frame: the current frame for these data
		"""
		# Add the data for the current frame. Creates a new history if the true vehicle(s) first appeared in this frame
		for nwuv in nwu_vehicles:
			x = numpy.block([nwuv.xyz, nwuv.xyz_dot])
			# Currently, the z-component for a vehicle's centroid as recorded in the ground truth files tends to sit
			# right at 0m in the world. We're accounting for this when we create ideal tracks by lifting the whole
			# bounding box by the vehicle's z-extent, which in CARLA is half the vehicle's height. We'd like to fix this
			# in CARLA so the centroid of the vehicle is recorded in the right place. But until then, we need to adjust
			# here and in the code that creates the ideal tracks.
			x[2] += nwuv.extent[2]
			self.true_records[nwuv.car_id][frame] = RecordDatum(x=x,	P=None, object_ids=None)

		# Save off data for each of the tracks
		for track in tracks:
			self.track_records[track.iden][frame] = RecordDatum(x=track.xhat, P=track.P,
				object_ids=[m.raw.object_id for m in mapping.get(track, [])])

		# Save off object ids observed in all cameras
		self.obj_id_records[frame] = [m.raw.object_id for m in measurements]
