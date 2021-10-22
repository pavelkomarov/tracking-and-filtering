import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class AnalysisParameters:
	# directory where you want to save all of the animations, figures, and metrics
	OUTPUT_DUMP_DIR: str = os.path.join(os.path.expanduser('~'), 'Desktop/oasis_dump')

	# Minimum number of points a track needs to have for us to run a tracking performance analysis on it.
	MIN_TRACK_LENGTH = 5


# VISUALIZATION CLASSES

class VisualizationView(Enum):
	CAMERA: str = 'camera'
	OVERHEAD: str = 'overhead'

@dataclass
class AxesFormatting:
	# if no extremas are provided, camera dimensions will be used
	xunit: str = '' # will be loaded with "meters", "pixels", whatever when this is instantiated
	yunit: str = ''
	xmin: int = None
	xmax: int = None
	ymin: int = None
	ymax: int = None


# ELEMENT PLOTTING CLASSES

class StationaryVisualizationElementType(Enum):
	CAMERA = 'camera'
	CAMERA_FIELD_OF_VIEW = 'camera_fov'


class VisualizationElementType(Enum):
	BOUNDING_BOX = 'bounding box'
	STANDARD_DEVIATION_ELLIPSE = 'sd ellipse'
	COVARIANCE_ELLIPSE = 'covariance ellipse'
	CENTROID = 'centroid'


class ElementConcatenationType(Enum):
	# determines whether element coordinates are concatenated normally or with an empty vector in between
	# the empty vector prevent line elements from being connected to each other by lines when plotted together
	SQUISHED = 'squished'
	EMPTY_VECTOR_SEPARATED = 'empty_vector_separated'

class PlottingType(Enum):
	# some elements are camera specific and should appear only on the plot for that camera, such as sensor measurements
	# others, such as tracks and truth measurements should be on every axes
	CAMERA_BASED: str = 'camera_based'
	EVERY_AXES: str = 'every_axes'

@dataclass
class ElementFormatting:
	# the formatting string is for the pyplot fmt parameter that allows formatting shortcuts like 'r+' for a red plus
	format_str: str = ''
	# kwargs are for more explicit formatting kwargs, such as linestyle='dashed'
	kwargs: Dict = field(default_factory=dict)
	# see description in ElementConcatenationType
	concat_type: ElementConcatenationType = None
