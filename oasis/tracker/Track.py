import numpy

from .Target import Target


class Track:
	"""A data storage class to make correlation and tracking more convenient."""
	SEEDLING = -1 # created when new tracks are spawned. Die next frame if no correlated measurements, or get promoted to INITIALIZING
	DEAD = 0 # This track should no longer be propagated or updated
	INITIALIZING = 1 # This track is collecting a batch of measurements to prime the Kalman filter
	PARTIAL = 2 # Partially Observable: The track is being updated with sets of measurements that don't cover its full state + extra info
	FULL = 3 # Fully Observable: The track is being updated with measurements that natively cover its full state
	COAST = 4 # The track has been in the FULL or PARTIAL state previously, but has not received any new measurements

	def __init__(self, iden: str, time_last_updated: float, xhat: numpy.ndarray, P: numpy.ndarray, target: Target,
		**kwargs):
		"""Populate the Track with initial information, which will thereafter be updateable via public attributes

		:param iden: an identifier to label this track uniquely
		:param time_last_updated: the time this track was last updated, equal to creation time for initialization
		:param xhat: a state estimate; we never actually know the real x
		:param P: a state error covariance, specifically of (x - xhat), where x is the unknown real state
		:param target: the thing we're tryig to track, knows its own state evolution dynamics
		:param kwargs: any extra attributes the user wishes to specify, allows storage of arbitrary features.
			Possibilities include: time, frame, bounding box, trackID, last active time, object id
		"""
		# Set params to be attributes of the object, without having to do each one individually
		for k, v in locals().items():
			if k != 'self' and k != 'kwargs':
				setattr(self, k, v)

		self.__dict__.update(kwargs) # sets the object to have fields with values given in the keyword arguments

		self.status = Track.SEEDLING # All Tracks start as fragile seedlings
		self.same_state_count = 1 # for counting the number of frames the track has had its current status
		self.bucket = {} # Where correlated Measurements are stored while a Track is initializing
