import numpy
from typing import Dict

from parse.data_parse_constants import TargetTypes

# The extents are listed in xyz order
SHOEBOX_XYZ_EXTENTS: Dict[str, numpy.ndarray] = {
	TargetTypes.BICYCLE.value: numpy.array([1.2, 0.3, 0.9]),
	TargetTypes.MOTORCYCLE.value: numpy.array([1.2, 0.3, 0.9]),
	TargetTypes.MOTORBIKE.value: numpy.array([1.2, 0.3, 0.9]),
	TargetTypes.CAR.value: numpy.array([2.4, 1.3, 0.8]),
	TargetTypes.BUS.value: numpy.array([5.0, 2.2, 1.6]),
	TargetTypes.TRUCK.value: numpy.array([5.0, 2.2, 1.6]),
}
