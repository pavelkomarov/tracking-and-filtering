
from abc import ABC, abstractmethod
import numpy
from typing import Tuple, Union, Any

class Sensor(ABC):
	"""Every tracking problem involves some kind of sensor, and every sensor has a different way of transforming
	information from the outside world into a measurement. Inheriting from this abstract base class ensures the user
	specifies what that relationship is in a commonly-accessible place.
	"""

	@abstractmethod
	def h(self, x: numpy.ndarray=None) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
		"""h() is from Extended Kalman Filter notation: y = h(x). If nonlinear, then you need yhat = h(xhat) and
		H = Jacobian_h(xhat). If linear, then H = Jacobian_h, and where you evaluate that doesn't make a difference
		because the variables have all disappeared, and yhat = H xhat. So it makes sense to hard-code in the linear
		case.

		:param x: where to evaluate the h function. Usually you're passing in a propagated estimate here, in which
			case you might call x "xhat". If linear case, it isn't necessary to pass this in.
		:returns: the function and the Jacobian evaluated at the input (nonlinear case) or just the static Jacobian
			(linear case)
		"""
		pass

	@abstractmethod
	def R(self, raw: Any=None) -> numpy.ndarray:
		"""R is the covariance of the measurement noise: innovation covariance S = E[(y-yhat)(y-yhat).T] = H P H.T + R,
		R = E[v v.T], yhat = h(xhat) + v. The noise level will be unique to each Sensor type, so it's convenient to be
		able to access it from the Sensor. The function also takes `raw` to allow R to be contingent on information from
		a particular Measurement.

		:param raw: salient information that belongs to a Measurement. Can be the value y, can be something like a bbox
		:returns: R, the covariance of the measurement noise
		"""
		pass

	@abstractmethod
	def __hash__(self):
		"""The Correlator partitions incoming Measurements into a dictionary keyed by originating Sensor, so Sensors
		need to be hashable.

		:returns: a uniformly-at-random but consistent whole number that can be used for hashing
		"""
		pass
