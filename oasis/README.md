# Project Structure
Due to most elements of this process requiring state, this project uses an object oriented design.
Once there is a natural grouping of functionality, especially if there is any state involved,
it should be contained within a class.
For the details of the current project structure, please see the documentation on Confluence.
Here are the main components:

`models` contains Camera, Vehicle, and BoundingBox objects, basically convenience classes for the video-object problem.

`parse` contains tools to read `.csv` files as dictionaries and create Camera, Vehicle, and the like from them. It also contains iterator classes to collect simulation information by frame so it can be conveniently passed on a frame at a time to the tracker.

`sim_data_tools` "takes the ground truth vehicle locations and orientations and produces the ideal bounding boxes in the camera frame" "it was intended to be incorporated at some point into the crockpot stuff that Josh is doing"

`test_data` holds some small CARLA outputs so we have something to read in to test Vehicle objects and the like. It doesn't get packaged as part of the wheel.

`tracker` contains a KalmanFilter class, a Correlator class, a Track class, a Measurement class, a Target class, and a Tracker class that puts the pieces together in a tracking loop. All of this is intended to be generic.

`transform` contains tools to change coordinate systems and create transformation matrices, as well as turn bounding boxes into volumes and back again.

`analysis` contains tools to visualize things.


# Getting Started

* This project uses [poetry](https://github.com/python-poetry/poetry) for virtual environment and dependency management
    * If necessary install poetry
    `curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python`
    * Run `poetry env use [/path/to/python/binary]` e.g. `poetry env use ~/.pyenv/versions/3.8.5/bin/python3`
    This creates the virtual environment and lets you explicitly tell it which python to run when the environment is active
    * Run `source $(poetry env info --path)/bin/activate` to activate the environment (`poetry shell` can also be used,
     but can cause potential issues if the shell is already activated. See
     [poetry documentation](https://python-poetry.org/docs/basic-usage/#activating-the-virtual-environment) for more info)
    * Run `poetry install` to install all dependencies into the environment
    * When finished, deactivate the environment with `deactivate`
* To run oasis:
    * Make sure poetry environment is activated
    * Make sure you're in the inner oasis directory
    * Run `python oasis_main.py (optional: -d [directory with test data, for instance test_data/EVIL_CHK001])`
* To run tests:
    * Make sure poetry environment is activated
    * Make sure you're in the inner oasis directory
    * Run `python -m pytest -s`


# Configuration
OASIS is configured in the `oasis_configuration` module. The file `oasis_config.py` contains
the functions used in this specific implementation of OASIS and can be adjusted without affecting the project core.
Parameters, such as thresholds, numeric constants for computations, input data field names
can be configured in `oasis_parameters.py`.
This is also where runtime configurations can be adjusted, such as which visualizations to create and which data to use as input.


# Contributing

## Comments
This code implements non-trivial calculations that would not be clear from clean code and naming alone.
Hence the propensity of comments. There is no requirement to fully match a specific comment style or level of elaboration,
but the goal is to have enough comments for a reader to fully understand what is happening.
It is the contributor's responsibility to ensure that comments are updated with the code
and accurately represent the current state.

## Testing
Unit tests should be written to cover any new functionality and modified with any changes.
Please ensure all tests are passing before committing code.
Due to the highly computational nature of the code, tests should be checking the mathematical accuracy of the
calculations. Testing of edge cases is especially important.

When possible, maintain tight testing boundaries in unit tests. This means that a unit test should not cover more
than one method- calls to other methods should be mocked.
There is room for a small set of broader "happy path" integration style tests, but they work in conjunction, not to replace the unit tests.

## Style/Formatting
For now, we are minimizing the number of tools used for automatically formatting the project
and are not enforcing a specific style beyond clear and readable code and standard best practices.
This might evolve as the team and the project grows, but we will not be adding extraneous tools or convention
until a need is felt.

### Tools
* To help automatically organize imports, you can use [isort](https://pypi.org/project/isort/).
* The project has [black-with-tabs](https://pypi.org/project/black-with-tabs/) available for automatic code formatting. Feel free to use it on new files.
Do not use on existing files if it causes excessive reformatting, since some contributors prefer a different coding style in their work.
