# This is where you put everything you want publicly accessible. With statements here, it
# becomes possible to `from oasis import Tracker2D`, for example. Without, you'd have to specify
# from `oasis.tracker.Tracker2D import Tracker2D`.
# https://stackoverflow.com/questions/16981921/relative-imports-in-python-3
from .tracker.Tracker import Tracker

__version__ = 0.1
