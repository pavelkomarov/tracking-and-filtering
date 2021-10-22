
from abc import ABC, abstractmethod
import numpy
from typing import Tuple, Union

class Target(ABC):
	"""Every tracking problem involves some kind of thing-being-tracked, a target. And different kinds of targets can
	have different dynamics to govern how state evolves through time. Inheriting from this abstract base class ensures
	the user specifies those dynamics in a commonly-accessible place.
	"""

	@abstractmethod
	def f(self, x: numpy.ndarray=None, dt: float=None) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
		"""f() is from Extended Kalman Filter notation: x[k] = f(x[k-1]) or xdot(t) = f(x(t)). If nonlinear and
		operating in the time domain (dt given), then you'll need an ODE solver like Runge-Kutta to propagate those
		dynamics forward. Otherwise, if your f is known for a given step, you can just apply it directly. For both
		cases: F = Jacobian_f(x), where for continuous x = x(t), and for discrete x = x[k-1]. For the linear case
		F = Jacobian_f, and where you evaluate that doesn't make a difference because the variables have all
		disappeared, and x[k] = F x[k-1] or xdot(t) = F x(t). So it makes sense to hard-code in the linear case.

		:param x: where to evaluate the f function. If linear case, it isn't necessary to pass this in.
		:param dt: If nonlinear case and time domain, you'll need to know how far to propagate the dynamics.
		:returns: the function and the Jacobian evaluated at the input (nonlinear case) or just the static Jacobian
			(linear case)
		"""
		pass

	@abstractmethod
	def Q(self) -> numpy.ndarray:
		"""Q is the covariance of the state evolution noise: state error covariance P = E[(x-xhat)(x-xhat).T] =
		F P_{k-1} F.T + Q, Q = E[w w.T], xhat = f(xhat_{k-1}) + w. The noise level will be unique to each Target type,
		so it's convenient to be able to access it from the Target.
		"""
		pass
