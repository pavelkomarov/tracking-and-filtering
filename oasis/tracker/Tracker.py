
import numpy
from typing import List, Callable

from .Measurement import Measurement
from .Track import Track
from .Correlator import Correlator
from .KalmanFilter import KalmanFilter
from .Initializer import Initializer
from configuration.parameters import InitializerParameters

class Tracker:
	"""This class keeps a list of Tracks, each representing an object, and a Correlator to properly associate new
	Measurements. The Tracker handles spawning and killing Tracks under the right conditions.
	"""
	def __init__(self, correlator: Correlator, spawn_tracks: Callable[[List[Measurement]], List[Track]]):
		"""Start the Tracker off with everything it needs to do its job

		:param correlator: an object that decides how to assign Measurements to Tracks
		:param spawn_tracks: a function that creates new Tracks from unassigned Measurements
		"""
		self.correlator = correlator
		self.spawn_tracks = spawn_tracks

		self.tracks = []
		self.time = 0


	def step(self, measurements: List[Measurement], time: float) -> List[Track]:
		"""Perform one tracking step based on a set of new observations.

		:param measurements: new input information to consider
		:param time: when the update is done, all tracks should reflect propagation and measurements up to this moment
		:returns:
		"""
		# We need to know how much time has passed so the KalmanFilter can make proper Phi_ks and Q_ks (see _make_phi).
		dt = time - self.time # so the first time dt is going to be yuge, but it won't get used because tracks is empty
		self.time = time # We're about to be up-to-the-moment, so reflect this

		# Propagate tracks to get current state estimates. This needs to be done before correlation and before
		# measurement Jacobians are found because those depend on xhats being at up-to-the-moment locations.
		for track in self.tracks:
			# Don't propagate seedling and initializing tracks, which don't yet have stable state
			if track.status in [Track.SEEDLING, Track.INITIALIZING, Track.DEAD]: continue

			xprime = track.xhat # save this value so that we can use it to pass dx to extras update function
			xhat_prop, F = track.target.f(xprime, dt)
			track.xhat, track.P = KalmanFilter.state_prop(track.xhat, track.P, F, track.target.Q(), dt, xhat_prop)

			# Propagate all the stuff that isn't the core state
			track.propagate_extras(track.xhat - xprime)

		# Use the Correlator to assign those measurements to tracks. {Track: [correlated Measurements]}
		mapping = self.correlator.correlate(self.tracks, measurements)

		# Update the status of all existing Tracks based on how many Measurements they got this bundle
		for track in self.tracks: # This needs to come before spawn_tracks
			track.state_machine(len(mapping.get(track,[])))

		# Drop tracks that are dead (or, rather, keep only tracks that aren't). Remove them from the mapping as well
		[[track, mapping.pop(track, [])] for track in self.tracks if track.status == Track.DEAD] # keep list?
		self.tracks = [track for track in self.tracks if track.status != Track.DEAD]

		# Initialize new Tracks for Measurements that don't seem to belong to any Track. Be careful to combine
		# Measurements that likely arise from the same thing or otherwise avoid duplications.
		spawn_mapping = {}
		if None in mapping: # And of course skip if there are no unexplained Measurements
			new_tracks, spawn_mapping = self.spawn_tracks(mapping[None], time)
			self.tracks += new_tracks
			del mapping[None] # so I don't have to check for this special case below

		# Calculate updates for each Track that has Measurements.
		for track, trk_meas in mapping.items():
			# Baby Tracks don't have known velocity so can't get Kalman update. Accumulate and average to aid correlation
			if track.status == Track.INITIALIZING:
				track.bucket[time] = trk_meas

				# Check if the total number of measurements added to the bucket exceed a value
				if sum(len(x) for x in track.bucket.values()) > InitializerParameters.MIN_BATCH_SIZE:
					_, F = track.target.f(track.xhat, dt)
					track.xhat, track.P = Initializer.batch_init(track.bucket, F)
					# Move this into the state machine update function when
					track.status = Track.PARTIAL if len(trk_meas) == 1 else Track.FULL # Camera problem only
					track.same_state_count = 1
				else:
					# TODO: some conditional here to decide whether to accumulate this info in measurement or state space
					R = numpy.sum([m.R_w for m in trk_meas], axis=0) # Add up all state-space Rs element-wise
					track.P[:R.shape[0],:R.shape[1]] = R # rest is zeros; would we always want to fill upper left?
					track.xhat = numpy.mean([numpy.block([m.y_w, numpy.zeros(3)]) for m in trk_meas], axis=0) # get the mean of all state-space estimates

			# Full-grown Tracks get a real Kalman update, which fuses new info with predictions for each object.
			else:
				xprime = track.xhat
				meas_space = track.status == Track.FULL
				assert track.status == Track.FULL or track.status == Track.PARTIAL # this line can be removed after we're confident it always passes

				for m in trk_meas:
					yhat, H = m.sensor.h(xprime, meas_space=meas_space) # yhat = where we expect to be; H = transform (or I)
					# where we measured (or guessed from measured), measurement noise (or guess of state noise)
					y, R = [m.y, m.R] if meas_space else [m.y_w, m.R_w]

					track.xhat, track.P = KalmanFilter.state_update(track.xhat, track.P, H, R, y, yhat)

				# I think you'll only want to do these things for full-grown Tracks
				# Kill tracks that are out of range. BAD: this funciton doesn't make a distinction between coming or going
				if track.out_of_range(trk_meas): track.status = Track.DEAD # Is this too aggressive?

				# Update all the stuff that isn't the core state
				track.update_extras(trk_meas)

			track.time_last_updated = time

		# Make sure the measurements used to spawn tracks are added to the mapping
		mapping.update(spawn_mapping)
		# Return all active tracks. Some of these were propagated and updated with information from Measurements. Also return mapping for history
		return self.tracks, mapping
