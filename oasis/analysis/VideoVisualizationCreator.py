from collections import defaultdict

import numpy
from matplotlib import animation, pyplot
from matplotlib.axes import Subplot
from matplotlib.dviread import Text
from matplotlib.lines import Line2D

import os
from typing import Dict, List, Union

from analysis.CamViewElementPlotter import CamViewElementPlotter
from analysis.OverheadViewElementPlotter import OverheadViewElementPlotter
from configuration.analysis_parameters import *
from configuration.video_visualization import VideoVisualizationConfig
from models.camera import NWUCamera
from models.vehicle import NWUVehicle
from tracker.Measurement import Measurement
from tracker.Track import Track


class VideoVisualizationCreator:
	"""Creates video visualizations. Using a configuration, it builds axes, determines which cameras go on which plots,
	and plots stationary data. It then takes in step by step data from oasis and uses the configuration to build and
	plot elements. Finally, it animates all the plotted elements and saves a video visualization.
	"""
	visualization_element_plotter_dictionary = {
		VisualizationView.CAMERA: CamViewElementPlotter,
		VisualizationView.OVERHEAD: OverheadViewElementPlotter,
	}

	def __init__(self, config: VideoVisualizationConfig, cameras: List[NWUCamera]):
		"""Initialize the list of axes and figure we're plotting on and plot stationary elements that are unchanging in
		all frames.

		:param config: which configuration to use, governs appearance of things
		:param cameras: visualizations should be created for these; currently assumes data exists exactly for these
			cameras; can add ability to allow different visualization camera list when needed;
		"""
		self._config = config
		self._cameras = [camera for camera in cameras if camera.iden in config.camera_ids]

		# the number of plots we need in order to respect the maximum camera count on each
		subplot_count = int(numpy.ceil(len(self._cameras)/config.max_camera_count_per_axis))

		self._fig, subplots = pyplot.subplots(subplot_count, figsize=(config.figure_size, config.figure_size))
		self._axes_dictionary = self._build_subplots(subplots, config.axes_formatting)
		self._format_figure()

		# visualization element plotter that is aware of configured view
		self.visualization_element_plotter = self.visualization_element_plotter_dictionary[self._config.visualization_view](
			self._cameras, self._axes_dictionary, store_plotted_elements=True
		)

		# plot stationary elements that will appear on all frames
		for stationary_element_type, element_formatting in self._config.stationary_elements.items():
			self.visualization_element_plotter.plot_stationary_elements(stationary_element_type, element_formatting)

		# plotted elements that are animated after the oasis run
		self._plotted_elements = []

		# these keep track of elements from frame to frame to allow plotting of element history
		self._measurement_element_history = {
			element_type: {camera: [] for camera in self._cameras}
			for element_type in config.measurement_elements
		}
		self._truth_vehicle_element_history = {
			element_type: [] for element_type in config.truth_vehicle_elements
		}
		self._track_element_history = {element_type: [] for element_type in config.track_elements}

	# PUBLIC INTERFACE

	def plot_step(
		self, frame_num: int, measurements: List[Measurement], vehicles: List[NWUVehicle], tracks: List[Track]
	):
		"""Plot all elements for a given time step. These get stored internally so the caller doesn't have to
		handle them.

		:param measurements: measurements for this frame
		:param vehicles: truth vehicles for this frame
		:param tracks: tracks for this frame
		:param frame_num: frame number for this frame
		"""
		# it is important to reset the list of elements in the plotter at each step
		# it stores the elements it plots so they can be added to the list of plotted elements together and appear on
		# the same animation step
		self.visualization_element_plotter.plotted_elements = []

		self._plot_measurements(measurements)
		self._plot_truth_vehicles(vehicles)
		self._plot_tracks(tracks)

		# add frame number and plotted elements together so they will appear on the same step of the animation
		self._plotted_elements.append(
			self._plot_frame_text(frame_num) + self.visualization_element_plotter.plotted_elements
		)

	def animate(self):
		"""Create the actual animation from all the accumulated plot elements and save as a movie at the output file location."""
		cam_view_animation = animation.ArtistAnimation(
			self._fig, self._plotted_elements, interval=self._config.animation_interval, blit=True
		)
		visualization_output_file = os.path.join(
			AnalysisParameters.OUTPUT_DUMP_DIR, self._config.visualization_output_file
		)
		cam_view_animation.save(visualization_output_file)

	# PRIVATE HELPERS

	def _build_subplots(self, subplots: Union[Subplot, List[Subplot]], axes_formatting: AxesFormatting) -> Dict[NWUCamera, Subplot]:
		"""Build plots based on configuration and assign them to cameras

		:param subplots: list of plain plots that need axes camera assignment or a single plot (quirk of pyplot.subplots function)
		:param axes_formatting: axes formatting configuration
		:return: dictionary that has an axes for each camera- values are not unique since multiple cameras can be on the same axes
		"""
		# for a single plot, plots is a single AxesSubplot, not a list
		if isinstance(subplots, Subplot):
			subplots = [subplots]

		subplot_dictionary = {}
		for index, camera in enumerate(self._cameras):
			plot = subplots[index % len(subplots)]

			# the first time we're seeing this subplot, we need to set all the things
			if index < len(subplots):
				plot.set_title(f"Camera {camera.iden}")
				plot.set_aspect("equal")
				xmin = axes_formatting.xmin if axes_formatting.xmin is not None else 0
				xmax = axes_formatting.xmax if axes_formatting.xmax is not None else camera.res_x
				ymin = axes_formatting.ymin if axes_formatting.ymin is not None else camera.res_y
				ymax = axes_formatting.ymax if axes_formatting.ymax is not None else 0
				plot.set_xlim(xmin, xmax)
				plot.set_ylim(ymin, ymax)
				plot.set_xlabel(axes_formatting.xunit)
				plot.set_ylabel(axes_formatting.yunit)

			# we've seen this subplot before, so we can just add a new camera to the name
			else:
				title = plot.get_title() + f", {camera.iden}"
				plot.set_title(title)

			# assign subplot to camera in the dictionary
			subplot_dictionary[camera] = plot

		return subplot_dictionary

	def _format_figure(self):
		"""Format figure as a whole outside the individual plots. Must be applied after plots are created
		for proper formatting.
		"""
		self._fig.suptitle = self._config.figure_name
		self._fig.tight_layout()

	def _plot_frame_text(self, frame_num: int) -> List[Text]:
		"""Build text element that adds frame text to plots

		:param frame_num: the number of the frame to be displayed
		:return: a list of plotted text elements ready to display
		"""
		frame_texts = []
		for plot in self._axes_dictionary.values():
			# the text placement is 1/30th the length of the axes from the axes start
			x_placement = plot.get_xlim()[0] + (plot.get_xlim()[1] - plot.get_xlim()[0]) / 30
			y_placement = plot.get_ylim()[0] + (plot.get_ylim()[1] - plot.get_ylim()[0]) / 30

			frame_text = plot.text(x_placement, y_placement, f"Frame: {frame_num}")
			frame_texts.append(frame_text)

		return frame_texts

	def _plot_measurements(self, measurements: List[Measurement]) -> List[Line2D]:
		""" plot elements from a list of measurements

		:param measurements: measurements for which elements are created
		:return: a list of plotted line elements ready to display
		"""
		# elements are gathered so they can all be plotted together to help visualization speed
		elements = {
			element_type: {camera: [] for camera in self._cameras}
			for element_type in self._config.measurement_elements.keys()
		}

		for measurement in measurements:
			camera = measurement.sensor

			# only plot measurements for the cameras that are part of the visualization
			if camera.iden not in self._config.camera_ids:
				continue

			for element_type in self._config.measurement_elements.keys():
				element = self.visualization_element_plotter.build_element(
					self.visualization_element_plotter.ElementDatumType.MEASUREMENT, element_type, measurement)
				elements[element_type][measurement.sensor].append(element)

				if element_type in self._config.measurement_element_histories.keys():
					self._measurement_element_history[element_type][camera].append(element)

		self._plot_elements(
			elements,
			self._measurement_element_history,
			self._config.measurement_elements,
			self._config.measurement_element_histories,
			PlottingType.CAMERA_BASED,
		)

		return self.visualization_element_plotter.plotted_elements

	def _plot_truth_vehicles(self, vehicles: List[NWUVehicle]) -> List[Line2D]:
		""" plot elements from a list of truth vehicles

		:param vehicles: truth vehicles for which elements are created
		:return: a list of plotted line elements ready to display
		"""
		# elements are gathered so they can all be plotted together to help visualization speed
		elements = defaultdict(list)

		for vehicle in vehicles:
			for element_type in self._config.truth_vehicle_elements.keys():
				element = self.visualization_element_plotter.build_element(
					self.visualization_element_plotter.ElementDatumType.TRUTH, element_type, vehicle)
				elements[element_type].append(element)

				if element_type in self._config.truth_vehicle_element_histories.keys():
					self._truth_vehicle_element_history[element_type].append(element)

		self._plot_elements(
			elements,
			self._truth_vehicle_element_history,
			self._config.truth_vehicle_elements,
			self._config.truth_vehicle_element_histories,
			PlottingType.EVERY_AXES,
		)

		return self.visualization_element_plotter.plotted_elements

	def _plot_tracks(self, tracks: List[Track]) -> List:
		""" plot elements from a list of tracks

		:param tracks: tracks for which elements are created
		:return: a list of plotted line elements ready to display
		"""
		# elements are gathered so they can all be plotted together to help visualization speed
		elements = defaultdict(list)

		for track in tracks:
			for element_type in self._config.track_elements.keys():
				element = self.visualization_element_plotter.build_element(
					self.visualization_element_plotter.ElementDatumType.TRACK, element_type, track)
				elements[element_type].append(element)

				if element_type in self._config.track_element_histories.keys():
					self._track_element_history[element_type].append(element)

		self._plot_elements(
			elements,
			self._track_element_history,
			self._config.track_elements,
			self._config.track_element_histories,
			PlottingType.EVERY_AXES,
		)

		return self.visualization_element_plotter.plotted_elements

	def _plot_elements(
		self,
		elements: Union[Dict[VisualizationElementType, List[numpy.ndarray]],
						Dict[VisualizationElementType, Dict[NWUCamera, List[numpy.ndarray]]]],
		element_history: Union[Dict[VisualizationElementType, List[numpy.ndarray]],
							   Dict[VisualizationElementType, Dict[NWUCamera,List[numpy.ndarray]]]],
		element_configuration: Dict[VisualizationElementType, ElementFormatting],
		history_configuration: Dict[VisualizationElementType, ElementFormatting],
		plotting_type: PlottingType
	):
		"""
		:param elements: a dictionary of coordinate elements to plot sorted by type and by camera if applicable
		:param element_history: a dictionary of history coordinate elements to plot sorted by type and by camera if applicable
		:param element_configuration: a dictionary of element formatting configuration for visualization types
		:param history_configuration: a dictionary of history element formatting configuration for visualization types
		:param plotting_type: plotting type representing whether elements should be plotted per camera or on every axes
		"""
		plotting_method = self.visualization_element_plotter.element_list_plotting_methods[plotting_type]

		for element_type, element_formatting in element_configuration.items():
			plotting_method(elements[element_type], element_formatting)

			if element_type in history_configuration:
				plotting_method(
					element_history[element_type], history_configuration[element_type],
				)
