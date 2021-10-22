# Project Structure
Most elements of this system require state and each have a few related functionalities, which lends itself to an object oriented design.

Here are the main components:

`models` contains Camera, Vehicle, and BoundingBox objects, basically convenience classes for the video-object problem.

`parse` contains tools to read `.csv` files as dictionaries and create Camera, Vehicle, and the like from them. It also contains iterator classes to collect simulation information by frame so it can be conveniently passed on a frame at a time to the tracker.

`sim_data_tools` "takes the ground truth vehicle locations and orientations and produces the ideal bounding boxes in the camera frame" "it was intended to be incorporated at some point into the crockpot stuff that Josh is doing"

`test_data` holds some small CARLA outputs so we have something to read in to test Vehicle objects and the like. It doesn't get packaged as part of the wheel.

`tracker` contains a KalmanFilter class, a Correlator class, a Track class, a Measurement class, a Target class, and a Tracker class that puts the pieces together in a tracking loop. All of this is intended to be generic.

`transform` contains tools to change coordinate systems and create transformation matrices, as well as turn bounding boxes into volumes and back again.

`analysis` contains tools to visualize things.


# Getting Started

* Set up your environment. If you like, do this in a virtual environment:
    - `python3 -m pip install venv`
    - `python3 -m venv .venv`
    - `source .venv/bin/activate`
    - `python3 -m pip install -r requirements.txt`
    - `deactivate` when done
* To run oasis:
    * cd to the `oasis` directory
    * `python oasis_main.py (optional: -d [directory with test data, for instance test_data/EVIL_CHK001])`
* To run tests:
    * cd to the `oasis` directory
    * `python -m pytest -s`


# Configuration
OASIS is configured in the `oasis_configuration` module. The file `oasis_config.py` contains
the functions used in this specific implementation of OASIS and can be adjusted without affecting the project core.
Parameters, such as thresholds, numeric constants for computations, input data field names
can be configured in `oasis_parameters.py`.
This is also where runtime configurations can be adjusted, such as which visualizations to create and which data to use as input.


# Contributing

## Comments
This code implements non-trivial calculations that would not be clear from clean code and naming alone.
Hence the proliferation of comments. There is no requirement to fully match a specific comment style or level of elaboration,
but the goal is to have enough comments for a reader to fully understand what is happening.

## Testing
Unit tests should be written to cover any new functionality and modified with any changes.
Please ensure all tests are passing before committing code.
Due to the highly computational nature of the code, tests should be checking the mathematical accuracy of the
calculations. Testing of edge cases is especially important.

When possible, maintain tight testing boundaries in unit tests. This means that a unit test should not cover more
than one method- calls to other methods should be mocked.
There is room for a small set of broader "happy path" integration style tests, but they work in conjunction, not to replace the unit tests.


