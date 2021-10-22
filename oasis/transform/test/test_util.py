# run with python3 -m pytest -s

import pytest
import numpy

from ..util import *
from ...analysis.viz import *


def test_rotate_rpy_from_ue4_to_nwu():
	# In UE4 this is a car rotated southward 45 degrees then rotated to face downward at 45 degrees with no roll
	rpy = numpy.array([0, numpy.pi/4, numpy.pi/4])
	# Relative to the NWU frame, this car now has to rotate 90 extra degress in the yaw direction to match the world
	# frame because the X axis is now further away, is tilted down 45 degrees by rotating the opposite way around an
	# opposite-facing Y axis, and still has no roll.
	in_nwu = numpy.array([0, -numpy.pi/4, -3*numpy.pi/4])
	assert numpy.allclose(in_nwu, rotate_rpy_from_ue4_to_nwu(rpy))


def test_xform_kinematics_from_ue4_to_nwu():
	# Say we have a particle at (1,1,1), going with speed 2 all the positive X,Y,Z directions, accelerating in the X
	# direction only at 5/time^2
	kinematics = numpy.array([[1,2,5],[1,2,0],[1,2,0]]) # 1st column is position, 2nd is velocity, 3rd is accel
	# In NWU coordinates the point will be now be at (-1,-1,1), because X and Y point the other way. The velocity
	# will be (-2,-2,2) for the same reason. And acceleration is going to be now in the -Y direction.
	in_nwu = numpy.array([[-1,-2,0],[-1,-2,-5],[1,2,0]])
	assert numpy.allclose(in_nwu, xform_kinematics_from_ue4_to_nwu(kinematics))
	# ensure also works on 1D, single vectors representing location, velocity, or acceleration alone
	kinematics_singleton = numpy.array([1,1,1])
	in_nwu = numpy.array([-1,-1,1])
	assert numpy.allclose(in_nwu, xform_kinematics_from_ue4_to_nwu(kinematics_singleton))


def test_get_t4_matrix():
	# Say we have a camera located at (30,20,10) in NWU that's got a yaw of 90 degrees, so it's facing west, a pitch
	# downward of 20 degrees, and no roll
	rpy = numpy.array([0, numpy.pi/9, numpy.pi/2])
	xyz = numpy.array([30, 20, 10])

	T = get_t4_matrix(rpy, xyz)
	plot_frames(T, "location of camera in world frame", 1) # to verify by eye it looks right

	c20 = numpy.cos(numpy.pi/9)
	s20 = numpy.sin(numpy.pi/9)

	assert numpy.allclose(T.dot(numpy.array([0,0,0,1])), numpy.array([30,20,10,1]))
	assert numpy.allclose(T.dot(numpy.array([1,1,1,1])), numpy.array([30-1, 20+c20+s20, 10+c20-s20, 1]))


def test_R_about_axis():
	# This one should definitely work if the t4 matrix test passes, but if that fails it'd be nice to at least know
	# whether R could be the problem
	for axis in ['x','y','z']:
		assert numpy.allclose(R_about_axis(0, axis), numpy.eye(3))

	assert numpy.allclose(R_about_axis(numpy.pi/2, 'x'), numpy.array([[1,0,0],[0,0,-1],[0,1,0]]))
	assert numpy.allclose(R_about_axis(numpy.pi/2, 'y'), numpy.array([[0,0,1],[0,1,0],[-1,0,0]]))
	assert numpy.allclose(R_about_axis(numpy.pi/2, 'z'), numpy.array([[0,-1,0],[1,0,0],[0,0,1]]))
