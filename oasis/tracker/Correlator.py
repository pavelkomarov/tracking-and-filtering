
import numpy
from typing import List, Callable
from collections import defaultdict
from scipy.optimize import linear_sum_assignment as hungarian_algorithm

from .Track import Track
from .Measurement import Measurement


class Correlator:
	"""Used to properly associate new measurements with tracks
	"""
	UNEXPLAINED = -1 # I had NaN up here, but that breaks ==
	AMBIGUOUS = -2
	OK = True

	def __init__(self, dist_func: Callable[[Measurement, Track], float],
		gating_func: Callable[[numpy.ndarray, int], float]):
		"""Constructor

		:param dist_func: a user-supplied distance function to determine how similar a Measurement is to a Track. This
			is defined differently from problem to problem based on state variables, measured variables, and metadata,
			so for maximum flexibility the Correlator takes it as a parameter. Note: distance(a, b) >= 0
		:param gating_func: a user-supplied function to determine when a (track, measurement) pair assigned by the
			Hungarian algorithm should be thrown away. Takes in the costs/distances of a Measurement's pairings with
			all active Tracks and the index of the best pairing in context of others (as determined by Hungarian
			algorithm). Returns Correlator.UNEXPLAINED, .AMBIGUOUS, or .OK judgement based on user's logic.
		"""
		self.dist_func = dist_func
		self.gating_func = gating_func


	def correlate(self, active_tracks: List[Track], measurements: List[Measurement]) -> numpy.ndarray:
		"""Figure out which tracks the measurements belong to.

		:param active_tracks: all the tracks currently under consideration to match against
		:param measurements: all the measurements that either need to be matched to tracks or considered new
		:returns: a mapping of Track -> correlated Measurements
		"""
		# each measurement gets assigned to a track or retains -1 -> no correlation
		assignment = numpy.ones(len(measurements), dtype='<i4')*Correlator.UNEXPLAINED # initially all are unexplained

		subset_dict = defaultdict(list) # make dict from Sensor -> indices of Measurements with that Sensor
		for l,m in enumerate(measurements):
			subset_dict[m.sensor].append(l)

		for _, subset in subset_dict.items(): # for each group of measurements from the same sensor (subset of all)
			
			# Calculate the distance metric between all (track, measurement) pairs
			D = numpy.zeros((len(active_tracks), len(subset)))
			for i,track in enumerate(active_tracks):
				for j,l in enumerate(subset):
					D[i,j] = self.dist_func(track, measurements[l])

			# This implementation of the Hungarian algorithm returns row_ind, col_ind, where D[row_ind, col_ind].sum()
			# is minimized. len(row_ind) == len(col_ind) == min(D.shape) == min(|tracks|, |measurements|). Here rows are
			# tracks and columns are measurements. If |tracks| >= |measurements|, then all measurements get assigned
			# to a track, and some tracks can be left over. If |tracks| < |measurements|, then |tracks| measurements get
			# assigned to a track, and some measurements are left over.
			r, c = hungarian_algorithm(D) # r = matched track indices; c = matched measurement subset indices
			for k in range(len(r)): # have to loop because assignment[subset][c] is a copy, not a view
				assignment[subset[c[k]]] = r[k] # any leftover measurements don't get assigned here and just stay -1

			# The above is the easiest way to use the Hungarian algorithm (works no matter whether |measurements| or
			# |tracks| is greater), but it forces matches. A forced solution might be suspect if a measurement doesn't
			# actually arise from any of the tracked objects (large distances), or if a measurement is ambiguously close
			# to several tracks (similar, near-minimal distances).
			for i,j in zip(r,c): # i is track index, j is measurements subset index
				judgement = self.gating_func(D[:,j], i) # pass the distances to all tracks and the index of the match
				if judgement != Correlator.OK:
					if judgement != Correlator.UNEXPLAINED and judgement != Correlator.AMBIGUOUS:
						raise ValueError("gating_func should return Correlator.UNEXPLAINED, .AMBIGUOUS, or .OK")
					assignment[subset[j]] = judgement # numeric so it can be stored in the assignments array

		# At this point assignment[i] = j means the ith measurement in the `measurements` list correlates with the jth
		# track in the `active_tracks` list. This isn't super interpretable, and the caller is going to have to deal
		# with dropping tracks, which will cause indices to change. So a more useful return value is a dictionary
		# {Track: [Measurements that correlate]}, including a None key for Measurements with no track, and excluding
		# Measurements that are ambiguous.
		mapping = defaultdict(list)
		for i,j in enumerate(assignment):
			if j != Correlator.AMBIGUOUS: # skip/drop the ambiguous Measurements
				k = None if j == Correlator.UNEXPLAINED else active_tracks[j]
				mapping[k].append(measurements[i])

		return dict(mapping) # references the same object (doesn't make a copy)
