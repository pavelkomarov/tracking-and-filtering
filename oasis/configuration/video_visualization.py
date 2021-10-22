from typing import List

from configuration.analysis_parameters import *


@dataclass
class VideoVisualizationConfig:
	"""This is the configuration class for video visualizations. It specifies the types required
	for a video visualization configuration and contains some defaults.
	"""
	visualization_output_file: str
	visualization_view: VisualizationView
	camera_ids: List[str]
	# constants that adjust look of animation
	axes_formatting: AxesFormatting
	max_camera_count_per_axis: int
	figure_name: str = 'Animation Name'
	figure_size: int = 10
	animation_interval: int = 50
	# elements and histories to plot
	stationary_elements: Dict[StationaryVisualizationElementType, ElementFormatting] = field(default_factory=dict)
	measurement_elements: Dict[VisualizationElementType, ElementFormatting] = field(default_factory=dict)
	measurement_element_histories: Dict[VisualizationElementType, ElementFormatting] = field(default_factory=dict)
	truth_vehicle_elements: Dict[VisualizationElementType, ElementFormatting] = field(default_factory=dict)
	truth_vehicle_element_histories: Dict[VisualizationElementType, ElementFormatting] = field(default_factory=dict)
	track_elements: Dict[VisualizationElementType, ElementFormatting] = field(default_factory=dict)
	track_element_histories: Dict[VisualizationElementType, ElementFormatting] = field(default_factory=dict)


# Parameters for from-camera view
CAM_VIEW = VideoVisualizationConfig(
	# general configuration
	visualization_output_file='cam_view.mp4',
	visualization_view=VisualizationView.CAMERA,
	camera_ids=[str(x) for x in range(6)],

	# constants that adjust look of animation
	axes_formatting=AxesFormatting(xunit='pixels', yunit='pixels'),
	max_camera_count_per_axis=10,

	figure_name='Camera View',
	figure_size=10,
	animation_interval=50,

	# elements and histories to plot
	measurement_elements={
		VisualizationElementType.CENTROID: ElementFormatting(
			format_str='r+', kwargs={'markersize': 3}, concat_type=ElementConcatenationType.SQUISHED),
		VisualizationElementType.BOUNDING_BOX: ElementFormatting(
			format_str='-r', kwargs={'linewidth': 0.5}, concat_type=ElementConcatenationType.EMPTY_VECTOR_SEPARATED),
		VisualizationElementType.COVARIANCE_ELLIPSE: ElementFormatting(
			format_str='-r', kwargs={'linewidth': 1}, concat_type=ElementConcatenationType.EMPTY_VECTOR_SEPARATED),
		VisualizationElementType.STANDARD_DEVIATION_ELLIPSE: ElementFormatting(
			format_str='-b', kwargs={'linewidth': 0.5}, concat_type=ElementConcatenationType.EMPTY_VECTOR_SEPARATED)
	},

	measurement_element_histories={
		VisualizationElementType.CENTROID: ElementFormatting(
			format_str='r+', kwargs={'markersize': 1}, concat_type=ElementConcatenationType.SQUISHED),
		VisualizationElementType.BOUNDING_BOX: ElementFormatting(
			format_str='-b', kwargs={'linewidth': 0.2}, concat_type=ElementConcatenationType.EMPTY_VECTOR_SEPARATED),
		VisualizationElementType.COVARIANCE_ELLIPSE: ElementFormatting(
			format_str='-r', kwargs={'linewidth': 0.2}, concat_type=ElementConcatenationType.EMPTY_VECTOR_SEPARATED),
		VisualizationElementType.STANDARD_DEVIATION_ELLIPSE: ElementFormatting(
			format_str='-b', kwargs={'linewidth': 0.5}, concat_type=ElementConcatenationType.EMPTY_VECTOR_SEPARATED)
	}
)

# Parameters for bird's eye view
OVERHEAD_VIEW = VideoVisualizationConfig(
	# general configuration
	visualization_output_file='overhead_view.mp4',
	visualization_view=VisualizationView.OVERHEAD,
	camera_ids=[str(x) for x in range(6)],

	# constants that adjust look of animation
	axes_formatting=AxesFormatting(xunit='meters', yunit='meters', xmin=-45, xmax=20, ymin=0, ymax=75),
	max_camera_count_per_axis=10,

	figure_name='Overhead View',
	figure_size=10,
	animation_interval=50,

	# elements and histories to plot
	stationary_elements={
		StationaryVisualizationElementType.CAMERA: ElementFormatting(),
		StationaryVisualizationElementType.CAMERA_FIELD_OF_VIEW: ElementFormatting(kwargs={'alpha': 0.1}),
	},

	measurement_elements={
		VisualizationElementType.CENTROID: ElementFormatting(
			format_str='r+', kwargs={'markersize': 3}, concat_type=ElementConcatenationType.SQUISHED),
		#VisualizationElementType.BOUNDING_BOX: ElementFormatting(
		#	format_str='-r', kwargs={'linewidth': 0.5}, concat_type=ElementConcatenationType.EMPTY_VECTOR_SEPARATED),
		#VisualizationElementType.COVARIANCE_ELLIPSE: ElementFormatting(
		#	format_str=':r', kwargs={'linewidth': 0.5}, concat_type=ElementConcatenationType.EMPTY_VECTOR_SEPARATED),
		VisualizationElementType.STANDARD_DEVIATION_ELLIPSE: ElementFormatting(
			format_str='--r', kwargs={'linewidth': 0.5}, concat_type=ElementConcatenationType.EMPTY_VECTOR_SEPARATED)
	},

	measurement_element_histories={
		#VisualizationElementType.CENTROID: ElementFormatting(
		#	format_str='r+', kwargs={'markersize': 1}, concat_type=ElementConcatenationType.SQUISHED),
		#VisualizationElementType.BOUNDING_BOX: ElementFormatting(
		#	format_str='-r', kwargs={'linewidth': 0.2}, concat_type=ElementConcatenationType.EMPTY_VECTOR_SEPARATED),
		#VisualizationElementType.COVARIANCE_ELLIPSE: ElementFormatting(
		#	format_str=':r', kwargs={'linewidth': 0.2}, concat_type=ElementConcatenationType.EMPTY_VECTOR_SEPARATED),
		#VisualizationElementType.STANDARD_DEVIATION_ELLIPSE: ElementFormatting(
		#	format_str='-r', kwargs={'linewidth': 0.2}, concat_type=ElementConcatenationType.EMPTY_VECTOR_SEPARATED)
	},

	truth_vehicle_elements={
		VisualizationElementType.CENTROID: ElementFormatting(
			format_str='g.', kwargs={'markersize': 2}, concat_type=ElementConcatenationType.SQUISHED),
		VisualizationElementType.BOUNDING_BOX: ElementFormatting(
			format_str='-g', kwargs={'linewidth': 0.5}, concat_type=ElementConcatenationType.EMPTY_VECTOR_SEPARATED)
	},

	truth_vehicle_element_histories={
		VisualizationElementType.CENTROID: ElementFormatting(
			format_str='g.', kwargs={'markersize': 1}, concat_type=ElementConcatenationType.SQUISHED),
		#VisualizationElementType.BOUNDING_BOX: ElementFormatting(
		#	format_str='-g', kwargs={'linewidth': 0.5}, concat_type=ElementConcatenationType.EMPTY_VECTOR_SEPARATED)
	},

	track_elements={
		VisualizationElementType.CENTROID: ElementFormatting(
			format_str='k.', kwargs={'markersize': 2}, concat_type=ElementConcatenationType.SQUISHED),
		VisualizationElementType.BOUNDING_BOX: ElementFormatting(
			format_str='-k', kwargs={'linewidth': 2}, concat_type=ElementConcatenationType.EMPTY_VECTOR_SEPARATED),
		#VisualizationElementType.COVARIANCE_ELLIPSE: ElementFormatting(
		#	format_str=':k', kwargs={'linewidth': 0.5}, concat_type=ElementConcatenationType.EMPTY_VECTOR_SEPARATED),
		VisualizationElementType.STANDARD_DEVIATION_ELLIPSE: ElementFormatting(
			format_str='--k', kwargs={'linewidth': 0.5}, concat_type=ElementConcatenationType.EMPTY_VECTOR_SEPARATED)
	},

	track_element_histories={
		VisualizationElementType.CENTROID: ElementFormatting(
			format_str='k.', kwargs={'markersize': 1}, concat_type=ElementConcatenationType.SQUISHED),
		#VisualizationElementType.BOUNDING_BOX: ElementFormatting(
		#	format_str='-k', kwargs={'linewidth': 2}, concat_type=ElementConcatenationType.EMPTY_VECTOR_SEPARATED),
		#VisualizationElementType.STANDARD_DEVIATION_ELLIPSE: ElementFormatting(
		#	format_str='-k', kwargs={'linewidth': 0.5}, concat_type=ElementConcatenationType.EMPTY_VECTOR_SEPARATED),
		#VisualizationElementType.COVARIANCE_ELLIPSE: ElementFormatting(
		#	format_str=':k', kwargs={'linewidth': 0.5}, concat_type=ElementConcatenationType.EMPTY_VECTOR_SEPARATED)
	}
)
