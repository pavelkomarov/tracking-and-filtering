
import numpy
from typing import Any

from .Sensor import Sensor


class Measurement:
	"""Purely a data storage class to make correlation and tracking more convenient.
	"""
	def __init__(self, y: numpy.ndarray, sensor: Sensor, raw: Any=None, **kwargs):
		"""Populate the Measurement with information, which will become publicly accessible attributes

		:param y: measurement values for all measured variables
		:param sensor: which sensor gave rise to this measurement
		:param raw: optional raw, extra information which will get passed to the Sensor's R() by the Tracker.
		:param kwargs: any extra attributes the user wishes to specify, allows storage of arbitrary features.
			Possibilities include: measurement ID, H matrix associated with obtaining y = Hx, a bounding box, object ID
		"""
		# Set all those params to be attributes of the object, without having to do each one individually
		for k, v in locals().items():
			if k != 'self' and k != 'kwargs':
				setattr(self, k, v)

		self.__dict__.update(kwargs) # sets the object to have fields with values given in the keyword arguments
