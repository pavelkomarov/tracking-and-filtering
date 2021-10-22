import os
import numpy
from matplotlib import pyplot
from matplotlib.axes import Subplot
from typing import Dict, List, Tuple, Union, NamedTuple
from collections import defaultdict

from models.camera import NWUCamera
from configuration.analysis_parameters import AnalysisParameters

from .History import History
from .Metrics import *


class Analyzer:
	"""This class uses both true and track records to aggregate truth, track, and scenario-wide metrics and plots to
	show how closely tracker performance matched the truth.
	"""
	# The following are the names of some output files where plots and results end up
	SWITCHES_PLOT_FNAME = 'switches.png'
	MAPPED_TIMELINE_FNAME = 'best_guess_lifetime_truth_association.png'
	METRICS_FNAME = 'metrics.txt'

	def __init__(self, history: History):
		"""The constructor calls a series of functions that either aid in or end up computing the metrics we use to
		analyze the tracks' performance. Once the numbers have been computed, the constructor also calls all of the
		methods that begin with plot. So just creating the object will do all of the analysis. The metrics are then
		available after it's been constructed if the results are needed elsewhere.

		:param history: records of where true objects were, track states and uncertainties, and object ids over time
		"""
		self.history = history
		self.track_map, self.true_map = self._make_maps() # Make the track <-> truth mappings

		# Compute all necessary metrics for true records, tracks, and the overall scenario
		self.truths_metrics = [TruthMetrics(true_id, self.true_map.get(true_id, []), true_record,
			self.history.track_records, self.history.obj_id_records)
				for true_id,true_record in self.history.true_records.items()]
		
		# make a TrackMetrics for all track records that have an associated true record and are over a certain length
		self.tracks_metrics = [TrackMetrics(track_id, self.track_map[track_id], track_record,
			self.history.true_records[self.track_map[track_id]])
				for track_id,track_record in self.history.track_records.items()
				if self.track_map[track_id] is not None and len(track_record) >= AnalysisParameters.MIN_TRACK_LENGTH ]
		
		self.scenario_metrics = ScenarioMetrics(history.iden, self.tracks_metrics) # one scenario per analyzer


	def gen_all_plots(self):
		"""Get all plotting functions and run them
		"""
		plot_methods = [getattr(self, func)
			for func in dir(self) if callable(getattr(self, func)) and 'plot_' in func]

		for plot_method in plot_methods: plot_method()
		self._plot_timeline(plot_switches=False) # Call this function with non-default boolean


	def _make_maps(self) -> Dict[Union[str, int], Union[str, int]]:
		"""Decide which true target each track is closest to. Each track maps on to one truth or no truth. Then get
		the reverse mapping as well, where one truth may map to many tracks.

		:return: mapping from track records to true records, mapping from true records to track records
		"""
		track_map = {}
		for track_id,track_record in self.history.track_records.items():
			best_mean_dist = float('inf')
			track_map[track_id] = None # It's possible the track doesn't map to any true track. Keep for stats

			for true_id,true_record in self.history.true_records.items():
				# Match point-by-point between truth and track
				distances = [] # distances from ith track record to jth true record over lifetime of the track
				for frame,record_datum in track_record.items():
					if frame in true_record: # then find L2 distance between (x,y) of track and truth
						distances.append(numpy.linalg.norm(true_record[frame].x[:2] - record_datum.x[:2]))

				# we're considering the distance from i to j to be the mean of the distances when there are matches
				if len(distances):
					d = numpy.mean(distances)
					if d < best_mean_dist:
						best_mean_dist = d
						track_map[track_id] = true_id # map the ith track to the jth target

		# track_map should be done. Now true_map:
		true_map = defaultdict(list)
		for track_id, true_id in track_map.items():
			true_map[true_id].append(track_id)

		return track_map, dict(true_map) # get true_map as ordinary dict so we can get KeyErrors again


	def __repr__(self) -> str:
		"""save to file in a structured readout and return the string to be printed to screen

		:return: a string with all of the metrics for a scenario
		"""
		titles = [' Truth Metrics ', ' Track Metrics ', ' Scenario Metrics ']
		headers = ['true id\t# ->tru\t# steps detected\t# steps tracked',
			'track id\t->tru\tobject ids\tMPE\tMVE\t# sw\tlength',
			'scenario id            \tMPE\tMVE\t# objs\t# sw\tsum(trk len)']
		glom = ''

		for title, header, metrics in zip(titles, headers, [self.truths_metrics, self.tracks_metrics, [self.scenario_metrics]]):
			breaker = ('#'*(len(header) // 2-len(title) // 2)) + title + ('#' * (len(header) // 2-len(title) // 2))
			glom += breaker + '\n' + header + '\n'
			for x in metrics:
				glom += str(x) + '\n'

		with open(os.path.join(AnalysisParameters.OUTPUT_DUMP_DIR, Analyzer.METRICS_FNAME), 'w') as f:
			f.write(glom)
		return glom

	@staticmethod
	def print_all(analyzers: List): # It's actually List[Analyzer], but that statement doesn't work until Python 3.10
		scenarios_metrics = [a.scenario_metrics for a in analyzers]

		title = ' Scenario Metrics '
		header = 'scenario id            \tMPE\tMVE\t# objs\t# sw\tsum(trk len)'
		glom = ('#'*(len(header) // 2-len(title) // 2)) + title + ('#' * (len(header) // 2-len(title) // 2))
		glom += '\n' + header + '\n'

		for sm in scenarios_metrics: glom += str(sm) + '\n'

		glom += '-'*len(header) + '\n'
		glom += 'Totals: ' + ScenarioMetrics.print_total(scenarios_metrics)

		with open(os.path.join(AnalysisParameters.OUTPUT_DUMP_DIR, Analyzer.METRICS_FNAME), 'w') as f:
			f.write(glom)
		print(glom)


	def plot_track_records(self) -> None:
		"""Plot the entire true track records and overlay all the estimated track records from overhead view"""
		fig, ax = pyplot.subplots(figsize=(10, 10))
		ax.grid()
		ax.set_aspect('equal')

		#TODO Add in the cameras, their pointing directions, and their fields of view

		# Lay down each true record to be the back drop
		for true_record in self.history.true_records.values():
			xs, ys = zip(*[datum.x[:2] for datum in true_record.values()])
			ax.plot(xs, ys, '-ko', markersize=4, linewidth=4)

		# Lay down the track record
		for track_record in self.history.track_records.values():
			xs, ys = zip(*[datum.x[:2] for datum in track_record.values()])
			ax.plot(xs, ys, '-o', markersize=1, linewidth=1.0)

		figure_output_file = os.path.join(AnalysisParameters.OUTPUT_DUMP_DIR, 'overhead_view_' +
			str(self.scenario_metrics.iden)) + '.png'
		fig.savefig(figure_output_file, dpi=200, bbox_inches='tight')
		pyplot.close(fig)


	def _plot_error_timeseries(self):
		"""Plot error bounds, i.e. the state estimate error and 1-stddev state error"""
		for track_metrics in self.tracks_metrics:
			# Don't plot the track if it didn't map to a true track.
			if len(frames := track_metrics.frames) == 0:
				continue

			state_error = numpy.array(track_metrics.state_errors)
			state_cov = numpy.array(track_metrics.state_cov_diagonals)
			labels = ['x error (m)', 'y error (m)', 'z error (m)', 'xdot error (m/s)', 'ydot error (m/s)', 'zdot error (m/s)']

			fig, axes = pyplot.subplots(3, 2, figsize=(10, 10), sharex=True)
			[axes[2, i].set_xlabel('frame') for i in range(2)]

			# position error in first column, velocity error in second. Top to bottom -> x,y,z components
			for i,label in enumerate(labels):
				axes[i % 3, i // 3].plot(frames, state_error[:, i], 'k-', label=label)
				axes[i % 3, i // 3].plot(frames, state_cov[:, i]**0.5, 'r--')
				axes[i % 3, i // 3].plot(frames, -state_cov[:, i]**0.5, 'r--')
				axes[i % 3, i // 3].grid()
				axes[i % 3, i // 3].legend()
				axes[i % 3, i // 3].set_ylim(-5, 5)

			figure_output_file = os.path.join(AnalysisParameters.OUTPUT_DUMP_DIR, f'errors_{track_metrics.iden}.png')
			fig.savefig(figure_output_file, dpi=200, bbox_inches='tight')
			pyplot.close(fig)


	def _plot_timeline(self, plot_switches: bool = True):
		"""Plots one of two figures.

		1. Plot frames in which first measurement in list was incorrectly correlated with a track. This is also
		refered to as the "timeline" plot or "switch" plot. This is meant to be used as a diagonostic tool when feeding
		the	tracker with "ideal" detections. When using deepstream detections, the object ids won't have a direct
		mapping	to the true tracks' ids, so no colored lines will show up.

		2.The track map tells us the best guess truth a track belongs to based on which truth it's closest to over the
		course of its life, regardless of whether you have objids from "ideal" or not. This will not show switching,
		just best-guess association through time. Use it to if tracks vaguely line up with true records through time
		and how many a true record has over its life.

		:param plot_switches: plots 1. when true, 2. when false
		"""
		# Sort the true ids, and then map them to integers on the y-axis. Used to conveniently plot the true tracks in
		# sorted order without gaps
		true_id_to_yaxis_map = {true_id: i for i, true_id in enumerate(sorted(self.history.true_records))}

		fig, ax = pyplot.subplots(figsize=(10,10))
		ax.set_yticks(list(true_id_to_yaxis_map.values())) # force the yticks to be the mapped integers
		ax.set_yticklabels(list(true_id_to_yaxis_map.keys())) # force the ytick labels to be the true ids
		ax.set_ylabel('True Vehicle ID')
		ax.set_xlabel('Frame')

		# For each true record, lay down a horizontal line that spans the frames it existed. Make it's y value
		# correspond to the mapping we gave it above
		for true_id in true_id_to_yaxis_map.keys():
			true_record = self.history.true_records[true_id]
			frames = list(true_record.keys())
			ax.plot(frames, true_id_to_yaxis_map[true_id]*numpy.ones(len(frames)), 'k-')

		# For each track record, lay down a horizontal line that spans the frames it existed. Place it just below the
		# true record indicated by the first correlated measurement's object id for that frame.
		for track_metrics in self.tracks_metrics:
			track_id = track_metrics.iden
			if plot_switches:
				# Find what true track each object id in the list corresponds to. Then get the y-axis value for that truth.
				# These will be our y-values for this track
				mapped_timeline = [true_id_to_yaxis_map[true_id] - 0.1
					for true_id in track_metrics.first_meas_obj_ids if true_id in true_id_to_yaxis_map]
				if len(mapped_timeline) == 0: continue # some tracks don't map to truth
				frames = track_metrics.frames
			else:
				true_id = self.track_map[track_id]
				if true_id is None: continue # some tracks don't map to a truth
				mapped_tracks_list = self.true_map[true_id]
				# Find where this track appears in the list of tracks mapped to the true id. We use this to stagger
				stagger_amount = mapped_tracks_list.index(track_id) + 1 # the track's record
				frames = self.history.track_records[track_id].keys()
				mapped_timeline = true_id_to_yaxis_map[true_id] * numpy.ones(len(frames)) - 0.1 * stagger_amount
			ax.plot(frames, mapped_timeline)

		fname = Analyzer.SWITCHES_PLOT_FNAME if plot_switches else Analyzer.MAPPED_TIMELINE_FNAME
		figure_output_file = os.path.join(AnalysisParameters.OUTPUT_DUMP_DIR, fname)
		fig.savefig(figure_output_file, dpi=200, bbox_inches='tight')
		pyplot.close(fig)
