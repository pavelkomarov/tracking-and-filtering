import numpy
from scipy.linalg import block_diag

from typing import Dict, List, Tuple
from .Measurement import Measurement
from .KalmanFilter import KalmanFilter

from configuration.parameters import InitializerParameters


class Initializer:
	"""A family of solvers that refines the estimate of x_0 from a collection of recent measurements. In a Kalman
	filter, it is assumed that the distribution of x_0 is known, and the filtering procedure generally starts from the
	mean and variance of x_0. In practice, we may not know know the distribution of x_0. We could guess the mean value
	and increase the variance artificially to accommodate for the uncertainties. Or instead, you can use a solver to
	arrive at a better estimate of the mean and variance by using a history of measurements.
	"""

	dRMS_THRESHOLD = InitializerParameters.BATCH_INIT_dRMS_THRESHOLD
	RMS_THRESHOLD = InitializerParameters.BATCH_INIT_RMS_THRESHOLD
	MAX_LOOP_COUNT = InitializerParameters.MAX_LOOP_COUNT

	@staticmethod
	def batch_init(batch: Dict[float, List[Measurement]], F: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
		"""Take a batch of Measurements collected over time and use them to refine the xhat for the for the last
		bundle of measurements in the batch. Iteratively run a propagate and update step using the measurements in the
		batch and perform a least squares fit of the resulting track. Use the least squares fit to update the x_0 and
		repeat the steps. When a set RMSE value is reached or a number of iterations has occured, stop and return the
		xhat corresponding to the last group of measurements in the batch to be up to the moment.

		TODO: This explanation is the best interpretation of the undocumented and unsourced MATLAB code. We need some
		reference to understand what each line below is doing and to verify the interpretation above.

		TODO: In batch_init we're assuming F is from the ODE and finding dt and getting Phi, but it's conceivable you'd
		have a case where you're already given Phi (F from the discrete update equation). Keep this in mind in case we
		want to reuse this code.

		:param bucket: a {time -> [Measurements]} dictionary that holds the list of measurements for each of the past
			time bundles
		:param F: the evolution dynamics for the target
		:return: the xhat and P for the last measurement bundle in the batch
		"""
		# Set initial values
		dRMS = 1
		RMSold = 1
		RMSnew = 100
		loop_count = 0

		# Pull out the first bundle in the batch, and then the first measurement in that bundle
		time_of_first_bundle, first_meas_bundle_in_batch = list(batch.items())[0]
		m = first_meas_bundle_in_batch[0] # get first measurement in bundle

		# Prime XHAT with an initial guess from the first measurement in the oldest bundle in the batch
		if len(first_meas_bundle_in_batch) >= 2 and (coord_w := m.sensor.get_3d(m.y, m.sensor.get_boresight_depth(m.y, 1))) is not None:
			# coord_w is the projection of the point into the world at a height of 1m
			XHAT = numpy.block([coord_w, numpy.zeros(3)])
		else:
			XHAT = numpy.block([m.y_w, numpy.zeros(3)])

		while (dRMS > Initializer.dRMS_THRESHOLD and RMSnew > Initializer.RMS_THRESHOLD and loop_count < Initializer.MAX_LOOP_COUNT):
			loop_count += 1

			# Reinitialize the variables. --- What do these represent?
			time = time_of_first_bundle
			AtWA = AtWb = btWb = 0
			xh = XHAT # Use the latest estimate of xhat corresponding to the first bundle of measurements in the batch
			PHI = numpy.eye(6)

			# Loop through the {time -> [Measurement]} dictionary.
			for meas_time, meas_bundle in batch.items():
				dt = meas_time - time # this will be 0 for the first measurement bundle
				time = meas_time # update the time so that we can compute dt next time through this loop

				# We don't need Q, so just pass in an I matrix with the same shape as F. Note that when dt=0, we still
				# get back the correct F. And passing in a dummy Q does not change F in any way in this case.
				Phi, _ = KalmanFilter.make_phi(F, numpy.eye(*F.shape), dt)
				xh = Phi.dot(xh)

				if len(meas_bundle) >= 2:
					ys_and_Rs = [[m.y, m.R] for m in meas_bundle]
					# Put the in-camera R's along the diagonal of a overall R matrix, resulting in a
					R = block_diag(*[i[1] for i in ys_and_Rs]) # (2*|meas|, 2*|meas|) shape array
					y = numpy.block([i[0] for i in ys_and_Rs]) # you end up with a (2*|meas|,) shape array
					yhs_and_hs = [m.sensor.h(xh) for m in meas_bundle]
					yhat = numpy.block([i[0] for i in yhs_and_hs]) # you end up with a (2*|meas|,) shape array
					# you end up with a (2*|meas|, 6) shape arrays. It's h stacked on itself |meas| times
					H = numpy.block([[i[1]] for i in yhs_and_hs])
				else:
					H = numpy.block([numpy.eye(3), numpy.zeros((3,  3))]) # a (3, 6) shape array
					R = m.R_w # Get the R in the world. A (3, 3) shape array
					y = m.y_w # get the bounding volume's centroid in the world. A (3,) shape array
					yhat = H.dot(xh) # A (3,) shape array

				# Matrix Summation
				PHI = Phi.dot(PHI) # results in 6x6
				A = H.dot(PHI) # results in a ({2*|meas|, 3}, 6) shape array depending on |meas|
				b = y - yhat # a (6,) shape array

				Rinv = numpy.linalg.inv(R)
				AtWA = AtWA + A.T.dot(Rinv.dot(A)) # results in a (6, 6) shape array
				AtWb = AtWb + A.T.dot(Rinv.dot(b)) # results in a (6,) shape array
				btWb = btWb + b.T.dot(Rinv.dot(b)) # results in a scalar

			# Correction term
			P = numpy.linalg.inv(AtWA)
			dx = P.dot(AtWb)

			# State estimate update
			XHAT = XHAT + dx # this should be the best guess of XHAT for the first bundle of measurements in the batch

			# RMS of measurements
			RMSnew = numpy.sqrt(btWb / len(batch)) # dividing by the number of frames that produced measurements
			dRMS = numpy.abs((RMSnew - RMSold) / RMSold)
			RMSold = RMSnew

		# Propagate XHAT forward until it is at the same time as the last bundle in the batch. The scale by 10 factor
		# is unannotated. Just increase how uncertain you are about the estimate?
		return PHI.dot(XHAT), P * 10
