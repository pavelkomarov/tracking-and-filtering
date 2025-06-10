from abc import ABC
from enum import Enum

import numpy

from typing import Dict, List, Mapping, Union

from matplotlib.axes import Subplot

from configuration.analysis_parameters import (
	ElementConcatenationType,
	ElementFormatting,
	StationaryVisualizationElementType,
	PlottingType,
	VisualizationElementType,
)
from models.camera import NWUCamera
from models.vehicle import NWUVehicle
from tracker.Measurement import Measurement
from tracker.Track import Track


class BaseVisualizationElementPlotter(ABC):
	"""The base class for view specific visualization element plotters"""

	# vertical 2 x 1 empty vector, useful for starting object coordinate arrays and separating sets of object coordinates
	EMPTY_VECTOR = numpy.ones((2, 1)) * numpy.nan

	class ElementDatumType(Enum):
		MEASUREMENT = 'measurement'
		TRUTH = 'truth'
		TRACK = 'track'


	def __init__(self, cameras: NWUCamera, axes: Union[Subplot, Dict[NWUCamera,Subplot]], store_plotted_elements: bool = False):
		"""Prepare plotter with cameras, axes, and method switch dictionaries to allow for a higher level of abstraction
		when calling this class

		:param cameras: cameras for which elements should be plotted
		:param axes: axes on which elements should be plotted- can be a single set of axes or a dictionary specifying
		axes for each camera (values are not unique if multiple cameras are on each axes)
		:param store_plotted_elements: whether plotted elements should be stored in the class, for example for use by animations
		"""
		self._cameras = cameras

		# handle the two type of axes so the rest of the code can expect a dictionary
		if isinstance(axes, Mapping):
			self._axes_dictionary = axes
		else:
			self._axes_dictionary = {camera: axes for camera in cameras}

		# keeps track of plotted elements so video animations can add them at the same time
		self._store_plotted_elements = store_plotted_elements
		self.plotted_elements = []

		# element coordinates are concatenated differently based on whether they are lines or points
		# lines require an empty vector separating elements to prevent cross-element lines
		self.concatenation_methods = {
			ElementConcatenationType.SQUISHED: self.__squish_concatenate_elements,
			ElementConcatenationType.EMPTY_VECTOR_SEPARATED: self.__empty_vector_separated_concatenate_elements,
		}

		# elements sets can be sorted into a dictionary by camera or plotted on each axes
		self.element_list_plotting_methods = {
			PlottingType.CAMERA_BASED: self.plot_camera_based_elements,
			PlottingType.EVERY_AXES: self.plot_every_axes_elements,
		}

		self._stationary_element_plotting_methods = {
			StationaryVisualizationElementType.CAMERA: self._plot_cameras,
			StationaryVisualizationElementType.CAMERA_FIELD_OF_VIEW: self._plot_camera_fields_of_view,
		}

		# To add a new element, add a method creating the coordinates to the appropriate views and an empty method
		# to the base class.
		# If the element is for a new datum type, then add the type to the ElementDatumType enum and a
		# plotting method to the creator class.
		# If the element is of a new type, add it to the VisualizationElementType enum.
		# To keep the sample configurations complete, add an example of the element to all appropriate configurations.
		self._element_builder_methods = {
			(self.ElementDatumType.MEASUREMENT, VisualizationElementType.COVARIANCE_ELLIPSE): self._build_measurement_covariance_ellipse,
			(self.ElementDatumType.MEASUREMENT, VisualizationElementType.STANDARD_DEVIATION_ELLIPSE): self._build_measurement_standard_deviation_ellipse,
			(self.ElementDatumType.MEASUREMENT, VisualizationElementType.CENTROID): self._build_measurement_centroid,
			(self.ElementDatumType.MEASUREMENT, VisualizationElementType.BOUNDING_BOX): self._build_measurement_bounding_box,
			(self.ElementDatumType.TRUTH, VisualizationElementType.CENTROID): self._build_truth_vehicle_centroid,
			(self.ElementDatumType.TRUTH, VisualizationElementType.BOUNDING_BOX): self._build_truth_vehicle_bounding_box,
			(self.ElementDatumType.TRACK, VisualizationElementType.COVARIANCE_ELLIPSE): self._build_track_covariance_ellipse,
			(self.ElementDatumType.TRACK, VisualizationElementType.STANDARD_DEVIATION_ELLIPSE): self._build_track_standard_deviation_ellipse,
			(self.ElementDatumType.TRACK, VisualizationElementType.CENTROID): self._build_track_centroid,
			(self.ElementDatumType.TRACK, VisualizationElementType.BOUNDING_BOX): self._build_track_bounding_box,
		}

	# ELEMENT PLOTTING METHODS

	def plot_stationary_elements(self, element_type: StationaryVisualizationElementType, element_formatting: ElementFormatting):
		"""Plot elements that do not change during the scenario

		:param element_type: element type to plot- member of the stationary visualization element type enum
		:param element_formatting: formatting that will be applied when plotting element
		"""
		self._stationary_element_plotting_methods[element_type](element_formatting)

	def plot_camera_based_elements(
		self, element_dictionary: Dict[NWUCamera, List[numpy.ndarray]], element_formatting: ElementFormatting,
	):
		"""Plot elements that are sorted by camera on the axes value for that camera

		:param element_dictionary: lists of elements as values for cameras for which they are plotted
		:param element_formatting: formatting for this set of elements
		"""
		for camera in self._cameras:
			element_list = element_dictionary[camera]

			self._plot_element_list(self._axes_dictionary[camera], element_list, element_formatting)

	def plot_every_axes_elements(
		self, element_list: List[numpy.ndarray], element_formatting: ElementFormatting,
	):
		"""Plot list of elements on every axes

		:param element_list: list of elements in coordinate form
		:param element_formatting: formatting for this set of elements
		"""
		for axes in self._axes_dictionary.values():
			self._plot_element_list(axes, element_list, element_formatting)

	def _plot_element_list(self, axes: Subplot, element_list: List[numpy.ndarray], element_formatting):
		"""Plot list of elements on given axes

		:param axes: axes on which elements are plotted
		:param element_list: list of elements in coordinate form
		:param element_formatting: formatting for this set of elements
		"""
		element_array = self.concatenation_methods[element_formatting.concat_type](element_list)

		plotted_elements = axes.plot(
			element_array[0], element_array[1], element_formatting.format_str, **element_formatting.kwargs
		)

		# for non-animated visualization, the elements are plotted
		# for animated, they are added to plotted_elements to be added to the animation at the same time
		if self._store_plotted_elements:
			self.plotted_elements.extend(plotted_elements)

	# ELEMENT BUILDING SWITCH
	# can be expanded to include plotting if needed

	def build_element(self, element_datum_type: ElementDatumType, element_type: VisualizationElementType,
					  datum: Union[Measurement, Track, NWUVehicle]) -> numpy.ndarray:
		"""Build requested element type from measurement

		:param element_datum_type: element datum used to build element
		:param element_type: element type to plot- member of the visualization element type enum
		:param datum: datum for which element should be built
		:return: coordinate array of element
		"""
		# select and call appropriate method for element type
		return self._element_builder_methods[(element_datum_type, element_type)](datum)

	# PLOTTER METHODS IMPLEMENTED IN CHILD CLASSES

	def _plot_cameras(self, element_formatting: ElementFormatting):
		"""Plot camera location, ray, and name
		depending on view, plots cameras or does nothing, since cameras will not appear in camera view

		:param element_formatting: currently unused, but can be used camera formatting is moved to config
		"""
		pass

	def _plot_camera_fields_of_view(self, element_formatting: ElementFormatting):
		"""Plot camera fields of view
		depending on view, plots fov or does nothing, since camera field of view does not make sense in camera view

		:param element_formatting: formatting for camera field of view
		"""
		pass

	# BUILDER METHODS IMPLEMENTED IN CHILD CLASSES

	def _build_measurement_bounding_box(self, measurement: Measurement) -> numpy.ndarray:
		"""Build measurement bounding box in appropriate view

		:param measurement: measurement for which bounding box is built
		:return: coordinate array of element
		"""
		pass

	def _build_measurement_standard_deviation_ellipse(self, measurement: Measurement) -> numpy.ndarray:
		"""Build measurement covariance ellipse in appropriate view

		:param measurement: measurement for which ellipse is built
		:return: coordinate array of element
		"""
		pass

	def _build_measurement_covariance_ellipse(self, measurement: Measurement) -> numpy.ndarray:
		"""Build measurement covariance ellipse in appropriate view

		:param measurement: measurement for which ellipse is built
		:return: coordinate array of element
		"""
		pass

	def _build_measurement_centroid(self, measurement: Measurement) -> numpy.ndarray:
		"""Build measurement centroid in appropriate view

		:param measurement: measurement for which centroid is built
		:return: coordinate array of element
		"""
		pass

	def _build_track_covariance_ellipse(self, track: Track) -> numpy.ndarray:
		"""Build track covariance ellipse in appropriate view

		:param track: track for which covariance ellipse is built
		:return: coordinate array of element or empty vector if in camera view
		"""
		pass

	def _build_track_standard_deviation_ellipse(self, track: Track) -> numpy.ndarray:
		"""Build track deviation ellipse in appropriate view

		:param track: track for which standard deviation ellipse is built
		:return: coordinate array of element or empty vector if in camera view
		"""
		pass

	def _build_track_centroid(self, track: Track) -> numpy.ndarray:
		"""Build track centroid in appropriate view

		:param track: track for which centroid is built
		:return: coordinate array of element or empty vector if in camera view
		"""
		pass

	def _build_track_bounding_box(self, track: Track) -> numpy.ndarray:
		"""Build track bounding box in appropriate view

		:param track: track for which bounding box is built
		:return: coordinate array of element or empty vector if in camera view
		"""
		pass

	def _build_truth_vehicle_centroid(self, vehicle: NWUVehicle) -> numpy.ndarray:
		"""Build coordinate array for vehicle centroid
		depending on view returns coordinate array or empty vector (truth vehicles do not exist yet in camera view)

		:param vehicle: vehicle for which centroid is built
		:return: coordinate array of element or empty vector if in camera view
		"""
		pass

	def _build_truth_vehicle_bounding_box(self, vehicle: NWUVehicle):
		"""Build bounding box for truth vehicle
		depending on view returns coordinate array or empty vector (truth vehicles do not exist yet in camera view)

		:param vehicle: vehicle for which bounding box is built
		:return: coordinate array of element or empty vector if in camera view
		"""
		pass

	# STATIC METHODS

	@staticmethod
	def get_ellipse_outline(xc: float, yc: float, P: numpy.ndarray) -> numpy.ndarray:
		"""Get a list of xy coordinates that form the outline of an ellipse centered at (xc, yc) and with a
		covariance matrix P. For why this works, see https://cookierobotics.com/007/. Every 2x2 matrix is essentially
		just stretching and rotating. So in the “Rotated Ellipse” formulation on that page, if you factor the r_x and
		r_y out of the [cos(t); sin(t)] vector and put it instead in the rotation matrix, you recover P.

		:param xc: the x-coordinate of the ellipse's center
		:param yc: the y-coordinate of the ellipse's center
		:param P: the 2D rotation matrix for the ellipse direction
		:return: the array of x-y coordinates that make up the ellipse's outline
		"""
		t = numpy.linspace(-numpy.pi, numpy.pi, 50)
		xy = numpy.vstack((numpy.cos(t), numpy.sin(t)))

		return numpy.dot(P, xy) + numpy.array([[xc], [yc]])

	@staticmethod
	def __squish_concatenate_elements(element_list: List[numpy.ndarray]) -> numpy.ndarray:
		"""Concatenate coordinate elements by combining them into a single matrix
		returns an empty vector if there are no elements to concatenate

		:param element_list: list of coordinate elements to concatenate
		:return:
		"""
		if not element_list:
			return BaseVisualizationElementPlotter.EMPTY_VECTOR

		return numpy.concatenate(element_list, axis=1)

	@staticmethod
	def __empty_vector_separated_concatenate_elements(element_list: List[numpy.ndarray]) -> numpy.ndarray:
		"""Concatenate coordinate elements by combining them into a single matrix, inserting empty vectors in between
		returns an empty vector if there are no elements to concatenate

		:param element_list: list of coordinate elements to concatenate
		:return:
		"""
		if len(element_list) == 0:
			return BaseVisualizationElementPlotter.EMPTY_VECTOR

		# intersperse empty vector between element cooridnates
		spaced_list = [BaseVisualizationElementPlotter.EMPTY_VECTOR] * (len(element_list) * 2 - 1)
		spaced_list[0::2] = element_list

		return numpy.concatenate(spaced_list, axis=1)
