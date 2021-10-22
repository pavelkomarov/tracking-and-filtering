"""This is the main function where we run the tracker in the Object Action Space (OAS). Run with
python3 oasis_analytics.py -d ~/path/to/simulation_folder
"""
import os
import shutil
from argparse import ArgumentParser
from typing import List
from joblib import Parallel, delayed
from tqdm import tqdm

from parse.camera_data_iterator import MultiCameraDataIterator
from parse.vehicle_data_iterator import VehicleDataIterator
from parse.scenario_files import ScenarioFiles
from parse import file_io
from tracker.Tracker import Tracker
from analysis.Analyzer import Analyzer
from analysis.VideoVisualizationCreator import VideoVisualizationCreator
from analysis.History import History
from configuration.functions import *
from configuration.analysis_parameters import AnalysisParameters


def run_scenario(sf: ScenarioFiles, multi: bool):
	"""run a single scenario

	:param sf: object holding all the relevant file pointers to csvs for this scenario
	"""
	# Load in camera metadata and create camera objects.
	cameras = file_io.load_NWUCameras_from_NWU_metadata(sf, KalmanFilterParameters.RHP)
	# Create the data iterators
	mcdi = MultiCameraDataIterator(sf)
	vdi = VehicleDataIterator(sf)

	# Analysis!
	video_creators = [] if multi else [VideoVisualizationCreator(config, cameras.values())
		for config in OasisRunParameters.VIDEO_VISUALIZATIONS]
	if OasisRunParameters.KEEP_HISTORY: history = History(sf.scenario_name)

	# Core tracking
	correlator = Correlator(dist_func_7, gating_func_pavel) # swap out or modify these functions as necessary
	tracker = Tracker(correlator, spawn_tracks_franz)
	# Loop through time
	for i,(multicam_data, vehicles_data) in enumerate(zip(mcdi, vdi)):
		# Create a list of true NWUVehicles for tracking comparison. This won't be in production.
		nwu_vehicles = create_NWUVehicles(vehicles_data)

		# Create Measurements for this time step, and figure out what time it is
		measurements, time, frame = make_measurements(multicam_data, cameras)
		if not multi: print(time) # to give a sense of life

		# Pass the Measurements to the Tracker's update(). Get back stuff for the history.
		tracks, mapping = tracker.step(measurements, time)

		# store info from this time step
		for vizcreant in video_creators: vizcreant.plot_step(i, measurements, nwu_vehicles, tracks)
		if OasisRunParameters.KEEP_HISTORY: history.update(nwu_vehicles, tracks, measurements, mapping, frame)
	# I think error metrics should actually get returned
	# Visualize the history. In production this won't be a thing, but for anything less it's helpful.
	for vizcreant in video_creators: vizcreant.animate()
	if OasisRunParameters.KEEP_HISTORY: return history


if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument("-d", "--data_dir", required=False, type=str, help="scenario directory or folder of many")
	args = parser.parse_args()

	# Delete the directory if it exists and recreate so that we start fresh
	if os.path.exists(AnalysisParameters.OUTPUT_DUMP_DIR): shutil.rmtree(AnalysisParameters.OUTPUT_DUMP_DIR)
	os.makedirs(AnalysisParameters.OUTPUT_DUMP_DIR)

	# Get all of the file path names loaded into memory
	data_dir = args.data_dir or OasisRunParameters.DEFAULT_DATA_DIRECTORY
	scenarios = []
	if not os.path.isdir(data_dir) or any(['.csv' in x for x in os.listdir(data_dir)]): # if zip file or has a csv inside
		scenarios.append(ScenarioFiles(data_dir, OasisRunParameters.CAMERAS_TO_USE))
	else: # then we have many scenarios
		for d in set(x.replace('.zip','') for x in os.listdir(data_dir)): # deduplicated scenario names
			d = os.path.join(data_dir, d)
			try: #fails if not enough cameras in the scenario or if actually a depth maps directory
				scenarios.append(ScenarioFiles(d, OasisRunParameters.CAMERAS_TO_USE))
			except: pass

	histories = Parallel(n_jobs=4)(delayed(run_scenario)(sf, len(scenarios)>1) for sf in tqdm(scenarios))
	
	# Analysis
	if histories[0] is not None:
		if len(histories) == 1 and histories[0]:
			analyzer = Analyzer(histories[0])
			print(analyzer) # also prints to file
			analyzer.gen_all_plots() # puts plots in the dump

		else: # the multi case
			analyzers = [Analyzer(history) for history in histories]
			Analyzer.print_all(analyzers) # print at scenarios-level
			for a in analyzers: a.plot_track_records() # only produce "track reccords" plot for each

