import numpy

from analysis.BaseVisualizationElementPlotter import BaseVisualizationElementPlotter
from configuration.analysis_parameters import ElementFormatting
from models.vehicle import NWUVehicle
from tracker.Measurement import Measurement
from tracker.Track import Track


class CamViewElementPlotter(BaseVisualizationElementPlotter):
	"""Builds and plots visualization elements for a camera view"""

	def _build_measurement_bounding_box(self, measurement: Measurement) -> numpy.ndarray:
		"""Build a bounding box for a measurement in the camera field of view

		:param measurement: measurement used to build bounding box
		:return: coordinate array of element
		"""
		bbox_data = measurement.raw

		return bbox_data.plot_repr()

	def _build_measurement_centroid(self, measurement: Measurement) -> numpy.ndarray:
		"""Build a centroid for a measurement

		:param measurement: measurement used to build centroid
		:return: coordinate array of element
		"""
		return numpy.array([[measurement.raw.centroid[0]], [measurement.raw.centroid[1]]])

	def _build_measurement_standard_deviation_ellipse(self, measurement: Measurement) -> numpy.ndarray:
		"""Build an ellipse for a measurement in the camera field of view and returns the coordinate array

		:param measurement: measurement used to build ellipse
		:return: coordinate array of element
		"""
		covariance = measurement.R

		# construct ellipse using standard deviation matrix
		# (camera view covariance is a diagonal matrix, so a simple square root works)
		return self.get_ellipse_outline(measurement.raw.centroid[0], measurement.raw.centroid[1], numpy.sqrt(covariance))

	def _build_measurement_covariance_ellipse(self, measurement: Measurement) -> numpy.ndarray:
		"""Build an ellipse for a measurement in the camera field of view and returns the coordinate array

		:param measurement: measurement used to build ellipse
		:return: coordinate array of element
		"""
		covariance = measurement.R

		return self.get_ellipse_outline(measurement.raw.centroid[0], measurement.raw.centroid[1], covariance)
