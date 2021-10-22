# run with python3 -m pytest -s

import pytest
import numpy

from ..Correlator import Correlator
from ..Track import Track
from ..Measurement import Measurement
from ...models.vehicle import NWUVehicle
from ...models.camera import NWUCamera


# Correlators take a distance function for initialization, so we'll need to come up with one.
def _dist_func(track: Track, measurement: Measurement) -> float:
	# Let's assume y completely covers x, so they represent the same state variables. H is now an identity matrix. For
	# simplicity let's also not consider notions of covariance and only care about the means.
	return numpy.linalg.norm(track.xhat - measurement.y)

# Correlators take a gating function at initialization, so we'll need one of these too.
def _gating_func_thresholds(pairing_costs: numpy.ndarray, pairing_ndx: int):
	if pairing_costs[pairing_ndx] > 5:
		return Correlator.UNEXPLAINED
	elif len(pairing_costs) > 1:
		a, b = pairing_costs.argsort()[:2]
		if a != pairing_ndx or pairing_costs[b] - pairing_costs[a] < 1:
			return Correlator.AMBIGUOUS

	return Correlator.OK

# We'll need some Tracks and Measurements to try to correlate together.
#
# Tracks have an id, a time, a state estimate and covariance, and a Target. I'm making the Target None here because its
# real use is to provide F and Q (and for nonlinear dynamics, xhat_prop) to the Tracker for the propagation step. It
# goes unused by the Correlator, because the Correlator only cares about how "far" Tracks and Measurements are from each
# other, which for the _dist_func used here depends only on xhat, which is stored directly.
#
# Measurements have a numerical value and a Sensor. I'm making the Sensor a string here because its use to the
# Correlator is merely to partition the matching process. In general the Sensor's use is to provide the Tracker with
# H and R (and, for nonlinear measurement, yhat), but the Correlator doesn't necessarily need that information. Here
# I'm considering state space and measurement space to be exactly the same, so I don't have to find H from a sensor,
# and my _dist_func gets to only care about the Track's state estimate and the Measurement's numerical value.
active_tracks = [Track('A', 0, numpy.array([0,0]), numpy.eye(2), None),
				 Track('B', 0, numpy.array([5,5]), numpy.eye(2), None),
				 Track('C', 0, numpy.array([10,0]), numpy.eye(2), None)]
measurements = [Measurement(numpy.array([0,0]), 'camA'),
				Measurement(numpy.array([5,5]), 'camB'),
				Measurement(numpy.array([10,0]), 'camC')]

correlator = Correlator(_dist_func, _gating_func_thresholds)


def test_correlate_one_to_one():
	mapping = correlator.correlate(active_tracks, measurements)

	assert len(mapping) == 3
	for t,mm in mapping.items(): # Track -> Measurements
		assert len(mm) == 1 # only one Measurement per Track here
		assert ('cam' + t.iden) == mm[0].sensor


def test_correlate_ambiguous():
	# Now let's add a Track really close to another. This should interfere with Track A and cause the Measurement
	# to be ambiguous and get tossed and not come out in the mapping
	active_tracks.append( Track('D', 0, numpy.array([0.1,0.1]), numpy.eye(2), None) )

	mapping = correlator.correlate(active_tracks, measurements)

	assert len(mapping) == 2
	for t,mm in mapping.items():
		assert t.iden != 'A' # A should be the one that got interfered with
		assert len(mm) == 1 # still just one per Track
		assert ('cam' + t.iden) == mm[0].sensor


def test_correlate_unexplained():
	# Now let's add a Measurement that's far enough from everyone else to not be accounted for.
	measurements.append( Measurement(numpy.array([10,10]), 'camB') )

	mapping = correlator.correlate(active_tracks, measurements)

	assert len(mapping) == 3
	for t,mm in mapping.items():
		assert len(mm) == 1
		if not t: # if the None
			assert numpy.allclose(mm[0].y, [10,10]) and mm[0].sensor == 'camB'
		else:
			('cam' + t.iden) == mm[0].sensor


def test_correlate_multi_measurements():
	# Now let's add a few more Measurements from other cameras of the same stuff
	measurements.append( Measurement(numpy.array([5.1, 5.1]), 'camA') ) # camA now sees camB's thing
	measurements.append( Measurement(numpy.array([10.1, 0.1]), 'camB') ) # camB now sees camC's thing

	mapping = correlator.correlate(active_tracks, measurements)

	assert len(mapping) == 3 # A, B, and C, -A due to ambiguity with D, and there's an unnamed new thing on scene
	for t,mm in mapping.items():
		if not t: # if the None
			assert numpy.allclose(mm[0].y, [10,10]) and mm[0].sensor == 'camB'
		else:
			assert len(mm) == 2
			can_see = set([mm[0].sensor, mm[1].sensor])
			if t.iden == 'B': # The B target is seen by camA and camB
				assert can_see == set(['camA', 'camB'])
			elif t.iden == 'C': # The C target is seen by camB and camC
				assert can_see == set(['camB', 'camC'])


def test_correlate_with_extra_tracks():
	# Now let's add in a couple extra Tracks that don't get paired with anything, because Measurements are too far away.
	# Remember "too far" is defined as distance 5 by our _dist_func.
	active_tracks.append( Track('E', 0, numpy.array([10,-10]), numpy.eye(2), None) )
	active_tracks.append( Track('F', 0, numpy.array([15,5]), numpy.eye(2), None) )

	mapping = correlator.correlate(active_tracks, measurements)

	assert len(mapping) == 3 # Still length 3 because E and F shouldn't be included in the mapping
	for t,mm in mapping.items(): # everything else is the same as last test case
		if not t: # if the None
			assert numpy.allclose(mm[0].y, [10,10]) and mm[0].sensor == 'camB'
		else:
			assert len(mm) == 2
			can_see = set([mm[0].sensor, mm[1].sensor])
			if t.iden == 'B': # The B target is seen by camA and camB
				assert can_see == set(['camA', 'camB'])
			elif t.iden == 'C': # The C target is seen by camB and camC
				assert can_see == set(['camB', 'camC'])


def test_no_tracks():
	# This one is subtle. The Correlator needs its assignment vector to be initialized to UNEXPLAINED so that if there
	# aren't enough Tracks for the Hungarian algorithm to match against Measurements, and therefore the "judgement"
	# loop isn't triggered for every Measurement, they still fall through with the right assignemnt value. I used to
	# initialize my assignment vector as numpy.empty*NaN, which worked to fill with NaNs, but then when I changed
	# UNEXPLAINED's value to -1 to simplify == checking, numpy.empty*-1 ends up being full of zeros and large negative
	# numbers. This doesn't cause a failure for any of the test cases above, because in those cases all entries of the
	# assignment vector get overwritten, because |Tracks| >= |Measurements| from any given Sensor, so extra Measurements
	# are force-paired with not-very-sensible leftover Tracks, and the judgement loop sorts it out.
	mapping = correlator.correlate([], measurements)

	assert len(mapping) == 1 # all have the None key
	assert len(mapping[None]) == len(measurements)
