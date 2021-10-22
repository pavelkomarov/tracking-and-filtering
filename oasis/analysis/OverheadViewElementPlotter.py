import numpy

from typing import Union

from analysis.BaseVisualizationElementPlotter import BaseVisualizationElementPlotter
from configuration.analysis_parameters import ElementFormatting
from models.camera import NWUCamera
from models.vehicle import NWUVehicle
from tracker.Measurement import Measurement
from tracker.Track import Track


class OverheadViewElementPlotter(BaseVisualizationElementPlotter):
	"""Builds and plots visualization elements for an overhead view"""

	def _build_measurement_bounding_box(self, measurement: Measurement) -> numpy.ndarray:
		"""Build a bounding box from a measurement in the overhead view

		:param measurement: measurement used to build a bounding box
		:return: coordinate array of element
		"""
		bbox = measurement.sensor.sens_to_world(measurement.raw)[1]
		return bbox.plot_repr()

	def _build_measurement_centroid(self, measurement: Measurement) -> numpy.ndarray:
		"""Build a centroid from a measurement in the overhead view

		:param measurement: measurement used to build a centroid
		:return: coordinate array of element
		"""
		return numpy.array([[measurement.y_w[0]], [measurement.y_w[1]]])

	def _build_measurement_standard_deviation_ellipse(self, measurement: Measurement) -> numpy.ndarray:
		"""Build an ellipse from a measurement in the overhead view

		:param measurement: measurement used to build an ellipse
		:return: coordinate array of element
		"""
		standard_deviation_matrix = self.__get_stands_deviation_from_covariance(measurement.R_w)

		return self.get_ellipse_outline(measurement.y_w[0], measurement.y_w[1], standard_deviation_matrix[:2, :2])

	def _build_measurement_covariance_ellipse(self, measurement: Measurement) -> numpy.ndarray:
		"""Build an ellipse from a measurement in the overhead view

		:param measurement: measurement used to build an ellipse
		:return: coordinate array of element
		"""
		return self.get_ellipse_outline(measurement.y_w[0], measurement.y_w[1], measurement.R_w[:2, :2])

	def _build_track_centroid(self, track: Track) -> numpy.ndarray:
		"""Build a centroid for a track in the overhead view

		:param track: track used to build a centroid
		:return: coordinate array of element
		"""
		return numpy.vstack(track.xhat[:2])

	def _build_track_covariance_ellipse(self, track: Track) -> numpy.ndarray:
		"""Build an ellipse for a track in the overhead view

		:param track: track used to build ellipse
		:return: coordinate array of element
		"""
		return self.get_ellipse_outline(*track.xhat[:2], track.P[:2, :2])

	def _build_track_standard_deviation_ellipse(self, track: Track) -> numpy.ndarray:
		"""Build an ellipse for a track in the overhead view

		:param track: track used to build ellipse
		:return: coordinate array of element
		"""
		standard_deviation_matrix = self.__get_stands_deviation_from_covariance(track.P)

		return self.get_ellipse_outline(*track.xhat[:2], standard_deviation_matrix[:2, :2])

	def _build_track_bounding_box(self, track: Track) -> numpy.ndarray:
		"""Build the rotated track's bounding box

		:param track: track used to build bounding box
		:return: coordinate array of element
		"""
		return NWUVehicle(qhat=1, xyz=track.xhat[:3], rpy=track.rpy, extent=track.extent).plot_repr()

	def _build_truth_vehicle_centroid(self, vehicle: NWUVehicle) -> numpy.ndarray:
		"""Build a centroid from a truth vehicle in the overhead view

		:param vehicle: truth vehicle used to build ellipse
		:return:
		"""
		return [[vehicle.xyz[0]], [vehicle.xyz[1]]]

	def _build_truth_vehicle_bounding_box(self, vehicle: NWUVehicle) -> numpy.ndarray:
		"""Build the rotated ground truth vehicle bounding box

		:param vehicle: vehicle used to build bounding box
		:return: coordinate array of element
		"""
		return vehicle.plot_repr()

	def _plot_cameras(self, element_formatting: ElementFormatting):
		"""Plot camera location, direction, and label

		:param element_formatting: currently unused, but can be used camera formatting is moved to config
		"""
		for index, camera in enumerate(self._cameras):
			cam_location = camera.xyz[:2, numpy.newaxis] # 2x1 vector

			# Get the length of the hypot for a 5m ray in the pointing direction of the camera. rpy[2] is the yaw, and
			# rpy[1] is the pitch (which affects how much of the camera vector we should see in the horizontal plane.
			cam_ray = numpy.array([[5 * numpy.cos(camera.rpy[2]) * numpy.cos(camera.rpy[1])],
								   [5 * numpy.sin(camera.rpy[2]) * numpy.cos(camera.rpy[1])]])

			cam_ray = numpy.concatenate((cam_location, cam_location + cam_ray), axis=1)

			# Add some text to the camera marker to label the id
			self._axes_dictionary[camera].text(camera.xyz[0] + 0.5, camera.xyz[1] + 0.5, camera.iden)

			# plot camera markers and ray lines on the correct axes
			self._axes_dictionary[camera].plot(cam_location[0], cam_location[1], # little black circles
				"o", markersize=5, markerfacecolor="w", markeredgewidth=0.5, markeredgecolor=(0.0, 0.0, 0.0, 1))  
			self._axes_dictionary[camera].plot(cam_ray[0], cam_ray[1], "k--", linewidth=0.5)  # black dashed lines

	def _plot_camera_fields_of_view(self, element_formatting: ElementFormatting):
		"""Plot the polygons representing the fields of view for the cameras

		:param element_formatting: formatting for field of view filled polygons
		"""
		for camera in self._cameras:
			x_max = camera.res_x
			y_max = camera.res_y

			horizontal_frame_edge = numpy.linspace(0, x_max, 100)
			vertical_frame_edge = numpy.linspace(y_max, 0, 100)

			bottom = [ camera.get_3d([x, y_max], depth)[:2, numpy.newaxis] for x in horizontal_frame_edge
					if (depth := camera.get_boresight_depth(numpy.array([x, y_max]), 0)) is not None ]
			right = [ camera.get_3d([x_max, y], depth)[:2, numpy.newaxis] for y in vertical_frame_edge
					if (depth := camera.get_boresight_depth(numpy.array([x_max, y]), 0)) is not None ]
			top = [ camera.get_3d([x, 0], depth)[:2, numpy.newaxis] for x in horizontal_frame_edge
					if (depth := camera.get_boresight_depth(numpy.array([x, 0]), 0)) is not None ]
			left = [ camera.get_3d([0, y], depth)[:2, numpy.newaxis] for y in vertical_frame_edge
					if (depth := camera.get_boresight_depth(numpy.array([0, y]), 0)) is not None ]

			# reverse orientation on top and left lines to ensure we keep moving in the same direction around the focal plane
			top.reverse()
			left.reverse()
			polygon_outline = bottom + right + top + left # not a problem if the line projections do not meet perfectly
			# if none of the lines had any projection, there is no field of view in the world x=0 plane
			if len(polygon_outline) == 0: return
			polygon_array = numpy.concatenate(polygon_outline, axis=1)

			self._axes_dictionary[camera].fill(polygon_array[0], polygon_array[1], **element_formatting.kwargs)

	@staticmethod
	def __get_stands_deviation_from_covariance(covariance: numpy.ndarray):
		""" makes standard deviation matrix from covariance matrix
		just a sketch of the approach- can be made more precise as we solidify it

		:param covariance: covariance matrix
		:return:
		"""
		cov_eigen_values, cov_eigen_vectors = numpy.linalg.eig(covariance)
		return cov_eigen_vectors * numpy.sqrt(cov_eigen_values)
