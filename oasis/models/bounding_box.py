
import numpy
from typing import Dict


class BoundingBox:

	def __init__(self, xmin: float, xmax: float, ymin: float, ymax: float, zmin: float=None, zmax: float=None,
		frame_id: int=None, timestamp: float=None, object_id: int=None, object_name: str=None,
		object_confidence: float=None) -> None:
		"""This class stores useful values related to a 2D or 3D bounding box

		:param xmin: the xmin of bounding box
		:param xmax: the xmax of bounding box
		:param ymin: the ymin of bounding box
		:param ymax: the ymax of bounding box
		:param zmin: the zmin of bounding box
		:param zmax: the zmax of bounding box
		:param frame_id: the frame id in which this bounding box appeared. Optional
		:param timestamp: the timestamp associated to the frame id. Optional
		:param object_id: a unique id for this bounding box. Optional
		:param object_name: the object was assigned during classification (or from ground truth). Optional
		:param object_confidence: a value representing the confidence of this being an actual object. Optional
		"""
		for k, v in locals().items(): # set all incoming params as attributes of the object
			if k != 'self':
				setattr(self, k, v)

		xc = (xmin + xmax)/2; yc = (ymin + ymax)/2
		self.centroid = numpy.array([xc, yc, (zmin + zmax)/2]) if zmin is not None else numpy.array([xc, yc])

		# Compute the diagonal of the (2D) bounding box
		self.diag_len: float = ((self.xmax - self.xmin)**2 + (self.ymax - self.ymin)**2)**0.5

	def plot_repr(self) -> numpy.ndarray:
		"""Works for both overhead and in-camera views, because overhead only uses x and y, and in camera u,v are the
		same thing as x,y

		:return: plotable numpy array representation of this bounding box
		"""
		return numpy.array([ [self.xmin, self.xmax, self.xmax, self.xmin, self.xmin],
							 [self.ymin, self.ymin, self.ymax, self.ymax, self.ymin] ])

	@classmethod
	def from_track_file_dictionary(cls, trk_dict: Dict[str, str], frame_w: int, frame_h: int, timestamp: float=None):
		"""Use this classmethod to load in the data from a JSON-like string representation. Follows convention here:
		https://safexai.atlassian.net/l/c/A9gcAtjX. Note values are in microseconds, and "microframes", but we convert
		to seconds and "frames" here.

		:param trk_dict: a dictionary where the keys are the header names in the _trk.csv or _ideal.csv file and the
			values are the entries in one row of the file
		:param frame_w: number of pixels along the width of the camera frame
		:param frame_h: number of pixels along the height of the camera frame
		:param timestamp: optional, the timestamp (in seconds) for the frame in which when the object was detected.
			Provide this value if you don't want to use the timestamp that comes with the dictionary.
		"""
		# turn values into ints
		for k in ['frame_id', 'object_id']:
			trk_dict[k] = int(trk_dict[k])

		# put confidence back to [0,1] range
		trk_dict['object_confidence'] = float(int(trk_dict['object_confidence']) / 100.0)

		# Timestamps are stored as seconds in this class. This is where the timestamp is computed from the value
		# in the file or the function argument timestamp. Convert from usec to sec in the case of the former. Note that
		# the / makes float results
		trk_dict['timestamp'] = timestamp if timestamp else int(trk_dict['timestamp']) / 1e6

		# mins, maxes, and centroids: convert to pixels
		for k in ['xmin', 'xmax']:
			trk_dict[k] = float(frame_w * int(trk_dict[k]) / 1e6)
		for k in ['ymin', 'ymax']:
			trk_dict[k] = float(frame_h * int(trk_dict[k]) / 1e6)

		# The 'camera_id' was added as a csv field in TF4 that wasn't in TF3 data files. Remove it if it is in this
		# dictionary. We pass in None so that we don't throw an error if 'camera_id' isn't a key in the dict. We also
		# toss the x and y centroids that exist in the file. Those values get computed during __init__ now.
		for k in ['x_centroid', 'y_centroid', 'camera_id']: trk_dict.pop(k, None)

		# just leave string values alone (object_name), and pass the dict as kwargs to the constructor
		return cls(**trk_dict)
