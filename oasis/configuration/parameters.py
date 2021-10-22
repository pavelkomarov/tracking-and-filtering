from typing import List

from configuration import video_visualization
from parse.data_parse_constants import TrackTypes

# Note: these parameters are documented in https://safexai.atlassian.net/wiki/spaces/PROG/pages/1929970866/Python+OASIS+Configuration+Parameters
# Keep the config and the documentation in sync


# RUN PARAMETERS

class OasisRunParameters:
	# input data directory- can be overwritten by command line argument.
	DEFAULT_DATA_DIRECTORY: str = 'test_data/Town10HD_Int0_228_deepstream_detections'

	# the type of track to use- DeepStream or ideal true boxes
	TRACK_TYPE: TrackTypes = TrackTypes.IDEAL_TRACKS # DPST_OBJECTS # DPST_TRACKS = nvidia tracker; 

	# If running on a single scenario, the subset of cameras to use. Data from other camera data will be discarded
	CAMERAS_TO_USE: List[str] = ['0', '1']
	# If running on multiple scenarios, the number of cameras to use. For scenarios with more than this number, we
	# select a random subset with the right cardinality, and we skip scenarios with fewer.
	N_CAMERAS_TO_USE: int = 3 # TODO: Not used yet

	# Whether to keep track info and create error bounds and switching plots for tracks over their lifetimes
	KEEP_HISTORY: bool = True

	VIDEO_VISUALIZATIONS = [
		# video_visualization.custom, # configs to govern the look of the output
		# video_visualization.CAM_VIEW,
		video_visualization.OVERHEAD_VIEW
	]

class MeasurementDiscardParameters:
	# bounding boxes with centroids within this margin of the frame edge will be discarded
	BMARGIN: float = 0

	# bounding boxes within this margin of all the sensor’s frame edges will be discarded
	# the method implementing this is not currently in use, but included for completeness
	OUT_OF_RANGE_BMARGIN: float = 50


# TRACKER MODULE PARAMETERS

class KalmanFilterParameters:
	# used to construct Q- the estimate of noise in the velocity component
	# full value is used in the x and y direction and 1/100 of this in the z direction
	QHAT: float = 5

	# rhat percentage: used to adjust the estimate of what the measurement noise covariance R is in the camera frame:
	# the percentage of the diagonal length of the bounding box that will be used to create the estimate.
	RHP: float = 0.05


class CorrelationParameters:
	# unexplained means that the measurement is above this threshold from any track and therefore should start a new one
	UNEXPLAINED_DIST_THRESHOLD: float = 3

	# ambiguous means that the difference in how close the measurement is to two different tracks is below this threshold
	AMBIGUOUS_DIST_THRESHOLD: float = 0.5


class InitializerParameters:
	# RMS is the root mean square across the measurements from the batch used so far of b.T.dot(Rinv.dot(b)) where b=y-yhat
	# when RMS is below this threshold, we stop iteration
	BATCH_INIT_RMS_THRESHOLD: float = 0.001

	# dRMS is the standardized difference between the RMS in two subsequent batch init loops
	# if this difference falls below the given threshold, we stop iteration
	BATCH_INIT_dRMS_THRESHOLD: float = 0.01

	# when the number of times we tried to improve our estimates reaches 30, we stop, even if other thresholds were not met
	MAX_LOOP_COUNT: int = 30

	# minimum number of measurements needed to run batch initialization
	MIN_BATCH_SIZE: int = 8


class TrackParameters:
	# Number of frames track can spend in corresponding state with no new measurements before being killed.
	INIT_FRAME_LIMIT: int = 12
	COAST_FRAME_LIMIT: int = 15

	# Speed thresholds used to decide when to update a target's yaw estimate.
	SINGLE_MEASUREMENT_SPEED_THRESHOLD: float = 1.0
	MULTI_MEASUREMENT_SPEED_THRESHOLD: float = 0.2

# MODELS MODULE PARAMETERS

class WorldProjectionParameters:
	# Noise in meters we expect along Z dimension when projecting R out in to world
	DEPTH_STANDARD_DEVIATION: float = 4

	# estimate of how much farther from the camera the top corners of a box are projected to height 0 than the bottom corners
	TOP_CORNER_DEPTH_FUDGE: float = 3
