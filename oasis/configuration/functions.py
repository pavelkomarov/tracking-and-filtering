"""This file is intended to be the interface where the user defines variables and functions needed to describe a new
Tracking problem.
"""
from collections import Counter
import numpy
from typing import Dict, List, Literal

from parse.data_parse_constants import TargetTypes
from models.vehicle import NWUVehicle
from models.bounding_box import BoundingBox
from tracker.Measurement import Measurement
from tracker.Track import Track
from tracker.Correlator import Correlator
from configuration.parameters import *

from models.shoebox import SHOEBOX_XYZ_EXTENTS


def state_machine(self, num_meas: int) -> None:
	"""Get the track's state in the state machine for the current frame given its current state and other variables

	:param num_meas: the number of measurements correlated with the track in the current frame
	"""
	self.same_state_count += 1

	if self.status == Track.SEEDLING:
		# Mark track as dead if there wasn't a measurement correlated with this track
		self.status = Track.DEAD if num_meas == 0 else Track.INITIALIZING
		self.same_state_count = 1

	elif self.status == Track.INITIALIZING:
		# Kill the track if it has been initializing too long (doesn't fill bucket by time limit)
		if self.same_state_count > TrackParameters.INIT_FRAME_LIMIT:
			self.status = Track.DEAD
		#elif 

	elif self.status == Track.PARTIAL:
		# Move to multi track or coasting based on the measurement count
		if num_meas >= 2:
			self.status = Track.FULL
			self.same_state_count = 1
		elif num_meas == 0:
			self.status = Track.COAST
			self.same_state_count = 1

	elif self.status == Track.FULL:
		# move to coasting when there are no measurements, to partial if one measurement
		if num_meas <= 1:
			self.status = Track.COAST if num_meas == 0 else Track.PARTIAL
			self.same_state_count = 1

	elif self.status == Track.COAST:
		# Go back to single or multi, or kill the track
		if num_meas >= 1:
			self.status = Track.PARTIAL if num_meas == 1 else Track.FULL
			self.same_state_count = 1
		elif self.same_state_count > TrackParameters.COAST_FRAME_LIMIT:
			self.status = Track.DEAD

Track.state_machine = state_machine


def out_of_range(self, trk_meas: List[Measurement]) -> bool:
	"""Determine whether a bunch of Measurements all show the Target near the edges of their ranges. If so, it's a good
	indication that the Target is entering or exiting the cumulative view of the Sensors. This has the effect of
	killing Tracks early, rather than letting them time out.
	"""
	return False
	# bmargin = BoundingBoxParameters.OUT_OF_RANGE_BMARGIN
	#
	# # check if all of the measurements show evidence of the bbox leaving the camera frame
	# return numpy.all([m.raw.xmin < bmargin or
	# 				  m.raw.ymin < bmargin or
	# 				  m.raw.xmax > m.sensor.res_x - bmargin or
	# 				  m.raw.ymax > m.sensor.res_y - bmargin
	# 				for m in trk_meas])

Track.out_of_range = out_of_range

# We need this because the Tracker currently both propagates and updates "extras", additional information stored in
# a Track that isn't part of the state. E.g., bounding boxes. For now we don't need anything to happen on the propagate
# side of this.
def propagate_extras(self, dx: numpy.ndarray) -> None:
	pass

Track.propagate_extras = propagate_extras

def update_extras(self, trk_meas: List[Measurement]) -> None:
	"""This function does some additional processing steps to update the track's bounding box, footprint corners,
	and extent.

	:param trk_meas: the track measurements correlated with the track for this frame
	"""
	# Determine what the most common object name was among all measurements correlated with this track.
	# TODO What about ties? Maybe look back at the previous time the object type was set for this track and use that?
	most_common_object_name = Counter([m.raw.object_name for m in trk_meas]).most_common()[0][0]
	# Set this track's extent based on the most common object type in the list of correlated measurements
	self.extent = SHOEBOX_XYZ_EXTENTS.get(most_common_object_name)

	yaw = numpy.arctan2(self.xhat[4], self.xhat[3])
	# If you only have one camera on a target, we'll want the target moving faster before we update its yaw. If we have
	# multiple cameras on a target, we'll be more sure of its position, and therefore velocity, so we can use a lower
	# threshold.
	speed_threshold = TrackParameters.SINGLE_MEASUREMENT_SPEED_THRESHOLD if len(trk_meas) == 1 \
		else TrackParameters.MULTI_MEASUREMENT_SPEED_THRESHOLD

	# Set this track's rpy. Note that if on the next time through the tracker loop, this track only has the propagate
	# step applied, the rpy won't change because linear state evolution dynamics means velocity can't change by propagation
	# alone; only updates change the course
	if numpy.linalg.norm(self.xhat[3:]) > speed_threshold: self.rpy = numpy.array([0, 0, yaw])

Track.update_extras = update_extras


def gating_func_pavel(pairing_costs: numpy.ndarray, pairing_ndx: int) -> Literal[Correlator.OK,
	Correlator.UNEXPLAINED, Correlator.AMBIGUOUS]:
	"""My rationale here is that if the pairing is too high cost, then the Measurement is too far from any Track to
	reasonably be explained by them; if a pairing is too close in cost to the next-best and next-worse choices, then
	it's ambiguous; and otherwise it's likely right.

	:param pairing_costs: a vector of costs for all (Track, Measurement) pairs for a particular Measurement
	:param pairing_ndx: If this function returns OK, the Measurement will be correlated with the active Track with this
		index.
	"""
	if pairing_costs[pairing_ndx] > CorrelationParameters.UNEXPLAINED_DIST_THRESHOLD:
		return Correlator.UNEXPLAINED
	# I'd prefer the pairing to be the best choice and for the next-best choice to not be close in cost
	elif len(pairing_costs) > 1:
		a, b = pairing_costs.argsort()[:2]
		if a != pairing_ndx or pairing_costs[b] - pairing_costs[a] < CorrelationParameters.AMBIGUOUS_DIST_THRESHOLD:
			return Correlator.AMBIGUOUS

	return Correlator.OK


def spawn_tracks_franz(measurements: List[Measurement], time: float) -> List[Track]:
	"""Franz approach: Initialize tracks from only one camera, the one with the most unexplained Measurements. This
	averts having to carefully combine and averts possible duplications.

	:param measurements: unexplained Measurements that we want to appropriately turn into Tracks
	:param time: when the Measurements are from
	:returns: the Tracks we need
	"""
	# Counter.most_common(1) returns [(item, count)], so index out that item
	plurality_sensor = Counter([unexplained.sensor for unexplained in measurements]).most_common(1)[0][0]

	tracks = []
	spawn_mapping = {}
	for unexplained in measurements:
		if unexplained.sensor.iden == plurality_sensor.iden: # == on a dataclass compares fields, breaks with numpy arrays
			# the state variables we'll be tracking, position and velocity
			x_0 = numpy.block([unexplained.y_w, numpy.zeros(3)])
			# the state covariance that matches the 6 state variables, position and velocity
			P_0 = numpy.zeros((6,6)); P_0[:3,:3] = unexplained.R_w

			# Include a few more variables that will get used in performance analysis
			trk_params = populate_seedling_track(unexplained)

			# Create the track. At this point, it is considered a seedling
			new_track = Track(str(time) + "_" + str(len(tracks)), time, x_0, P_0, NWUVehicle(KalmanFilterParameters.QHAT),
				**trk_params)
			tracks.append(new_track)
			spawn_mapping[new_track] = [unexplained]

	return tracks, spawn_mapping


def populate_seedling_track(m: Measurement):
	"""Extra parameters each track updates. Might expand this function or move it somewhere else. Lots of commented out
	values until I know where we'll need them.

	:param m: a Measurement that creates the seedling track
	"""
	return dict(
		rpy=numpy.zeros(3),
		extent=SHOEBOX_XYZ_EXTENTS.get(m.raw.object_name),
	)


def _bhattacharyya_distance(u_1, u_2, S_1, S_2):
	"""A distance function between points from two different Gaussian distributions
	"""
	S = (S_1 + S_2)/2
	d = (u_1 - u_2)[:, numpy.newaxis] # now nx1 instead of shape (n,)

	return d.T.dot(numpy.linalg.inv(S)).dot(d)/8 + \
		numpy.log( numpy.linalg.det(S) / numpy.sqrt(numpy.linalg.det(S_1)*numpy.linalg.det(S_2)) )/2


def dist_func_7(track: Track, measurement: Measurement) -> float:
	"""implementation of Solution 7: https://safexai.atlassian.net/wiki/spaces/PROG/pages/1689453018/Correlation
	"""
	xhat = track.xhat # get the position of the track (state space)
	P = track.P # get the covariance of track position error (state space)

	yhat, H = measurement.sensor.h(xhat) # finds x projected in camera frame and the Jacobian and evaluated at x

	# transform the state estimate to measurement space through H, and take Bhattacharyya distance there
	return _bhattacharyya_distance(measurement.y, yhat, measurement.R, H.dot(P).dot(H.T))


def dist_func_2(track: Track, measurement: Measurement) -> float:
	"""implementation of Solution 2: https://safexai.atlassian.net/wiki/spaces/PROG/pages/1689453018/Correlation

	"""
	track_pos = track.xhat[:3] # get the position of the track (state space)
	track_cov = track.P[:3, :3] # get the covariance of track position error (state space)

	meas_pos = measurement.y_w # get the numerical measurement vector (measurement space)
	meas_cov = measurement.R_w # get the covariance of measurement error (measurement space)

	# transform the state estimate to measurement space through H, and take Bhattacharyya distance there
	return _bhattacharyya_distance(meas_pos, track_pos, meas_cov, track_cov)


def make_measurements(multicam_data, cameras):
	"""Turn the multicam data into proper Measurement objects with BoundingBoxes and references to the right NWUCameras.
	Get the time (average) all this data comes from so Measurements can be compared against properly-propagated Tracks.

	:param multicam_data: A json-like dictionary yielded (by a MultiCameraDataIterator in dev) each frame
	:param cameras: dictionary of camera ids -> NWUCameras for the scenario
	"""
	measurements = []
	times = []

	# Get the timestamp that comes out of the iterator. We'll use this time because it comes straight from the
	# simulation data. As of 2021-01-06, there isn't a standard timestamp across files thats come out of the simulation
	# and deepstream pipeline. Still yet to be determined is how timestamps will actually come in from the messaging
	# piece. So we'll use this for now.
	timestamp = multicam_data['timestamp']
	frame = multicam_data['frame_id']
	# It is possible that no camera shows a detection for this frame. If this is the case, the `times` list will be
	# empty and we won't return a timestamp from this function. Adding it so that we have at least one timestamp. We
	# will want it so that the tracks propagate even when there are no new measurements.
	times.append(timestamp)
	for cam_id, cam_data in multicam_data['camera_data'].items(): # for each camera
		cam = cameras[cam_id] # The JSON only has Camera ID, but we'd like the full Camera
		# Below might be a place we append another timestamp as a camera is likely to send in a timestamp for a list of
		# detections from deepstream for a given frame.

		for det_idx, detection in enumerate(cam_data[OasisRunParameters.TRACK_TYPE]): # for each object detected in that camera
			# send in the timestamp that comes in with the whole object detections "packet"
			bbox = BoundingBox.from_track_file_dictionary(detection, cam.res_x, cam.res_y, timestamp)

			# Ignore bboxes that aren't around vehicles or have centroids in the margins
			bmargin = MeasurementDiscardParameters.BMARGIN
			if not any(bbox.object_name==x.value for x in TargetTypes) or \
				bbox.centroid[0] <= bmargin or bbox.centroid[0] >= cam.res_x - bmargin or \
				bbox.centroid[1] <= bmargin or bbox.centroid[1] >= cam.res_y - bmargin:
				continue

			y, R = cam.R(bbox, meas_space=True) # Get the centroid and R (measurement space)

			y_w, R_w = cam.R(bbox, meas_space=False) # Get the (guessed) centroid and R (state space)

			# drop measurement if no centroid is returned (world projection falls behind camera)
			if y_w is None:
				continue

			m_iden = f'{timestamp}-{cam.iden}-{det_idx}'

			new_meas = Measurement(y, cam, raw=bbox, R=R, y_w=y_w, R_w=R_w, frame_id=bbox.frame_id, iden=m_iden)

			measurements.append(new_meas)

	# print(numpy.std(times)) # zero for the ideal case
	return measurements, numpy.mean(times), frame


def create_NWUVehicles(vehicles_data: List[Dict[str, str]]) -> List[NWUVehicle]:
	"""Create NWUVehicles from data dictionaries produced by VehicleDataIterator.

	:param vehicles_data: The list of json-like dictionaries yielded by a VehicleDataIterator each frame
	:return: The newly created NWUVehicles
	"""
	return [
		NWUVehicle.from_NWU_dict(veh_data, int(veh_data['timestamp']), veh_data['vehicle_type'], KalmanFilterParameters.QHAT)
		for veh_data in vehicles_data]
