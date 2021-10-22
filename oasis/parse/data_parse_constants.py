from enum import Enum


class TrackTypes(Enum):
	DPST_TRACKS: str = 'dpst_trks'
	IDEAL_TRACKS: str = 'ground_truth_trks'
	DPST_OBJECTS: str = 'dpst_objects'


class ParsedDataKeys:
	FRAME_ID: str = 'frame_id'
	TIMESTAMP: str = 'timestamp'
	CAMERA_DATA: str = 'camera_data'


class TargetTypes(Enum):
	BICYCLE: str = 'bicycle'
	CAR: str = 'car'
	MOTORCYCLE: str = 'motorcycle'
	MOTORBIKE: str = 'motorbike'
	BUS: str = 'bus'
	TRUCK: str = 'truck'


class IncomingDataKeys:
	FRAME_ID: str = 'frame_id'
	CAMERA_ID: str = 'camera_id'
	FRAME_COUNT: str = 'frame_count'
	CAR_ID: str = 'car_id'
	CAR_TYPE: str = 'car_type'
	TIMESTAMP: str = 'timestamp'
	VEHICLE_TYPE: str = 'vehicle_type'


