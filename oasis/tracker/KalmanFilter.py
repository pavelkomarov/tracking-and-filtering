
import numpy
from numpy.linalg import inv
from scipy.linalg import expm
from typing import Tuple

class KalmanFilter:
	"""The filtering algorithm can broadly be thought of as alternating between a prediction/state estimate propagation
	step and an incorporate measurements/state estimate update step. This class has two corresponding public methods.

	The functions in this class are static and take all information from outside. This is to remain generic and
	flexible:

	Need to filter many things? Just pass a different x,P for each.
	Need to describe the dynamics of one thing differently from the dynamics of another? Just pass a different F,Q.
	Need to describe different kinds of measurement? Just pass different H,R,y.
	Need to use an EKF? Just pass in f(xhat) or h(xhat). https://en.wikipedia.org/wiki/Extended_Kalman_filter#Discrete-time_predict_and_update_equations
	Need to propagate a system described with a linear ODE? Just pass in F from the ODE and dt.
	Need to change one of your matrices mid-tracking? Just pass in a different one.

	If you don't have a guess for your noise covariances, pass in identity matrices for Q and R.
	"""
	@staticmethod
	def initial_state_from_measurement(H: numpy.ndarray, R: numpy.ndarray, y: numpy.ndarray) -> \
		Tuple[numpy.ndarray, numpy.ndarray]:
		"""To get started, a filtering cycle needs initial estimates of state and state error covariance. If initial
		state is unknown and you're instead beginning with a measurement y, then that single time step really just
		comprises a classic linear inverse problem, so optimal guesses have known form: the solution to the Best Linear
		Unbiased Estimator (BLUE).

		BIG CAVEAT: this only works if the measurement covers the whole state; if it does not, like a camera viewing a
		3D object, then the system is underdetermined, and (H.T inv(R) H) will be singular. In that case, you need a
		different way to produce initial estimates.

		:param H: Measurement matrix (linear case) or measurement Jacobian (EKF case), dimension MxN
		:param R: Measurement noise covariance, dimension MxM
		:param y: A new measurement, dimension M
		:return: initial state estimate, dimension N, and initial error covariance, dimension NxN
		"""
		#P_0 = inv(H.T inv(R) H)
		P_0 = inv( H.T.dot( inv(R) ).dot(H) )
		#x_0 = inv(H.T inv(R) H) H.T inv(R) y
		x_0 = P_0.dot(H.T).dot( inv(R) ).dot(y)

		return x_0, P_0


	@staticmethod
	def state_prop(xhat: numpy.ndarray, P: numpy.ndarray, F: numpy.ndarray, Q: numpy.ndarray, dt: float=None,
		xhat_prop: numpy.ndarray=None) -> Tuple[numpy.ndarray, numpy.ndarray]:
		"""Guess what state will be next iteration (or some time later) by propagating forward.

		This is half the filtering cycle. Usually you'd enter the cycle here after estimating initial state from a
		measurement.

		A linear continuous-time system evolves as d/dt xhat = F_t xhat + (w_t ~ N(0,Q_t))
		A linear discrete-time system evolves as xhat_next = F_k xhat + (w_k ~ N(0,Q_k)), where F_k is often called Phi.

		:param xhat: An estimated state, dimension N
		:param P: State error covariance, dimension NxN
		:param F: State evolution matrix (linear case) or state evolution Jacobian (EKF case), dimension NxN
		:param Q: State evolution noise covariance, dimension NxN.
		:param dt: A time differential. If given, F and Q are taken to be continuous-time versions, and the actual
			discrete system update needs to be calculated with _make_phi. If not given, F and Q are taken to be their
			discrete-time versions and are used in calculations directly.
		:param xhat_prop: (EKF case) next state, already propagated by nonlinear evolution function f(xhat), to avert
			having to specify, take, or propagate custom dynamics here
		:return: new state estimate, dimension N, and new error covariance, dimension NxN
		"""
		if dt: F, Q = KalmanFilter.make_phi(F, Q, dt) # in the event we're going from continuous to discrete time

		# \hat{x}_{k+1|k} = F_k \hat{x}_{k|k} : state update extrapolation
		xhat = F.dot(xhat) if xhat_prop is None else xhat_prop
		# P_{k+1|k} = F_k P_{k|k} F_k^T + Q : state error covariance matrix extrapolation
		P = F.dot(P).dot(F.T) + Q

		return xhat, P


	@staticmethod
	def state_update(xhat: numpy.ndarray, P: numpy.ndarray, H: numpy.ndarray, R: numpy.ndarray, y: numpy.ndarray,
		yhat: numpy.ndarray=None) -> Tuple[numpy.ndarray, numpy.ndarray]:
		"""Combine current state guess with measurement(s) to get a better guess.

		This is half the filtering cycle. It is possible to start here by choosing initialization arbitrarily and
		accounting for first measurement via update.

		Measurement (linear case) obeys y = H x + (v ~ N(0,R))

		Unlike on the state propagation side, measurement happens at a particular moment, so there is no integration
		over evolving dynamics, and discrete-time H and R are the same as their continuous-time cousins. Hence t or k
		subscripts are dropped.

		:param xhat: An estimated state, dimension N
		:param P: State error covariance, dimension NxN
		:param H: Measurement matrix (linear case) or measurement Jacobian (EKF case), dimension MxN
		:param R: Measurement noise covariance, dimension MxM
		:param y: A new measurement, dimension M
		:param yhat: (EKF case) measurement calculated by nonlinear function h(xhat), to avert having to specify, take,
			or calculate custom measurement function here
		:return: new state estimate, dimension N, and new error covariance, dimension NxN
		"""
		G = KalmanFilter._calculate_kalman_gain(P, H, R) # get the Kalman gain for this update

		if yhat is None: yhat = H.dot(xhat) # linear case
		# \hat{x}_{k+1|k+1} = \hat{x}_{k+1|k} + G_{k+1} (y_{k+1} - H_{k+1} \hat{x}_{k+1|k}) : state update
		xhat = xhat + G.dot(y - yhat)
		# P_{k+1|k+1} = (I - G_{k+1} H_{k+1}) P_{k+1|k} : state error covariance matrix update
		P = (numpy.eye(P.shape[0]) - G.dot(H)).dot(P)

		return xhat, P


	@staticmethod
	def _calculate_kalman_gain(P: numpy.ndarray, H: numpy.ndarray, R: numpy.ndarray) -> numpy.ndarray:
		"""Use the latest measurement matrix and measurement noise covariance, along with the filter's extrapolation
		of the state error covariance P_{k+1|k}, to find the new gain, which balances how much to weigh the filter's
		previous guess of the state vs an incoming measurement.

		:param P: State error covariance, dimension NxN
		:param H: measurement matrix, dimension MxN
		:param R: measurement noise covariance, dimension MxM
		:return: a Kalman Gain matrix, dimension NxM
		"""
		S = H.dot(P).dot(H.T) + R # H_{k+1} P_{k+1|k} H_{k+1}^T + R_k of dimension MxM
		return P.dot(H.T).dot(inv(S)) # P_{k+1|k} H_{k+1}^T S^-1 of dimension NxM


	@staticmethod
	def make_phi(F: numpy.ndarray, Q: numpy.ndarray, dt: float) -> Tuple[numpy.ndarray, numpy.ndarray]:
		"""Turn continuous-time F and Q into their discrete-time versions for an update period of dt. This formula is
		explained here: http://www.diva-portal.org/smash/get/diva2:699061/FULLTEXT01.pdf "Discrete-time Solutions to
		the Continuous-time Differential Lyapunov Equation With Applications to Kalman Filtering". Essentially, you're
		integrating the dynamical system's differential equation to produce a discrete-time update, as they do with
		Laplace Transforms here: http://www.robots.ox.ac.uk/~ian/Teaching/Estimation/LectureNotes2.pdf.

		:param F: continuous-time state evolution matrix, dimension NxN
		:parma Q: continuous-time state noise covariance, dimension NxN
		:param dt: the time difference between updates, which will effect how much the system evolves between the kth
			and k+1th steps, as reflected in the values of discretized matrices. Neat: if dt=0, then Phi comes out as
			the identity matrix, and Q_k comes out as zeros, because there's no system evolution and no extra noise.
		:return: Phi, Q_k discrete-time state evolution matrix and noise covariance, both dimension NxN
		"""
		N = F.shape[0]

		C = numpy.block([[F, Q],[numpy.zeros((N,N)), -F.T]])
		C = expm(C*dt)

		Phi = C[:N,:N] # this is the same as expm(F*dt)
		Q_k = C[:N,N:] @ Phi.T

		return Phi, Q_k
