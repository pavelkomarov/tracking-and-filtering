# run with python3 -m pytest -s

import pytest
import numpy

from ..KalmanFilter import KalmanFilter
from ...analysis.viz import plot_2D_track


def test_on_2d_particle():
	# let state = [x, y, x', y']
	# x_{k+1} = x_k + dt*x'_k
	# y_{k+1} = y_k + dt*y'_k
	# x'_{k+1} = x'_k
	# y'_{k+1} = y'_k
	dt = 1
	F = numpy.array([[1, 0, dt, 0],
					 [0, 1, 0, dt],
					 [0, 0, 1, 0],
					 [0, 0, 0, 1]])
	# observation = [x y]
	H = numpy.array([[1, 0, 0, 0],
					 [0, 1, 0, 0]])

	x_0 = numpy.array([0, 0, 0, 0]) # we believe the system starts out at the origin at rest
	P_0 = numpy.eye(F.shape[0])
	Q = numpy.eye(F.shape[0])
	R = numpy.eye(H.shape[0])

	real_xs = []
	filtered_xs = []

	x = numpy.array([100, 300, 10, 5]) # real state of the system starts off the origin and moving
	xhat = x_0
	P = P_0
	real_xs.append(x)
	filtered_xs.append(xhat)
	for k in range(100):
		# get random noise contributions
		w_k = numpy.random.multivariate_normal(numpy.zeros(Q.shape[0]), Q)
		v_k = numpy.random.multivariate_normal(numpy.zeros(R.shape[0]), R)

		# evolve the true system
		x = F.dot(x) + w_k
		y = H.dot(x) + v_k

		real_xs.append(x)

		# combine current guess for x with a measurement to get a better guess
		xhat, P = KalmanFilter.state_update(xhat, P, H, R, y)

		filtered_xs.append(xhat)

		# guess what x will be next iteration by propagating forward a step
		xhat, P = KalmanFilter.state_prop(xhat, P, F, Q)

	plot_2D_track(real_xs, filtered_xs, 1)

	L2error = [numpy.linalg.norm(real_xs[k] - filtered_xs[k]) for k in range(100)]

	print("mean L2 error for last 90 steps:", numpy.mean(L2error[10:]))
	assert numpy.mean(L2error[10:]) < 4 # I generally see error slightly over 2
	assert L2error[-1] < L2error[0]
