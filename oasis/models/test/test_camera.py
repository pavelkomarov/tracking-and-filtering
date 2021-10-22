# run with python3 -m pytest -s

import pytest
import numpy
from matplotlib import pyplot

from ..camera import NWUCamera
from ...analysis.viz import plot_frames, plot_camera_perspective
from ...transform.util import R_about_axis, get_t4_matrix


# Create a 1080p camera on a pole at the origin, looking east, pitched down 45 degrees, with 0.08 r hat percentage
cam = NWUCamera(1920, 1080, numpy.array([0,0,10]), numpy.array([0,numpy.pi/4,-numpy.pi/2]), numpy.pi/2, 0.08)


def test_hashable():
	hash(cam) # needs to be hashable for the Correlator


def test_h():
	plot_frames(cam.T_wc, "camera frame vs NWU frame", 1) # double check frames of ref by viz

	# Say we have an unmoving point at (0,-10,0)
	x_w = numpy.array([0, -10, 0, 0, 0, 0], dtype=float) # that's [x, y, z, x', y', z']

	# The projected point should be right in the middle of the field of view
	yhat, H = cam.h(x_w)
	plot_camera_perspective(cam.res_x, cam.res_y, yhat[0], yhat[1], 1)
	assert numpy.allclose(yhat, [960, 540])

	# As for the Jacobian, consider this relationship: f(x) - f(p) = J_f(p)(x - p) + o(||x - p||)
	# Our original x_w is p. So let's choose a new x along the camera's "boresight".
	x_mid = (numpy.concatenate([cam.xyz, [0, 0, 0]]) + x_w) / 2 # [0, -5, 5, 0, 0, 0]

	# Now if we send this through the camera equations, we're going to get yhat_mid = [960, 540]. So:
	# yhat - yhat_mid = 0 = H(x_w - x_mid)
	assert numpy.allclose(H.dot(x_w - x_mid), [0, 0])

	# Very cool. Here's a more visual exposition of that relationship above: f(x) - f(p) = J_f(p)(x - p) + o(||x - p||)
	# -> h(x) - h(p) - H(p)(x-p) = o(||x - p||). Let's find the left and right of this using H and corruptions of H.
	for noise_level in [0, 1, 5, 10]:
		H_ = numpy.copy(H)
		H_[:,:3] += noise_level*numpy.random.normal(size=(2,3))

		lnorms =[]
		rnorms = []
		for i in range(1000):
			x_new = numpy.copy(x_w)
			x_new[:3] += numpy.random.normal(size=3)

			yhat_new = cam.h(x_new, yhat_only=True) # don't need the Jacobian at the new point for this

			left = yhat_new - yhat - H_.dot(x_new - x_w)
			right = x_new - x_w

			lnorms.append(numpy.linalg.norm(left))
			rnorms.append(numpy.linalg.norm(right))

		pyplot.scatter(rnorms, lnorms)
		pyplot.xlabel('||x - p||')
		pyplot.ylabel('||h(x) - h(p) - H(p)(x-p)||')
		pyplot.title('noise level = ' + str(noise_level))
		pyplot.show(block=False)
		pyplot.pause(1) # only show each for a second
		pyplot.close()

	# Now let's check again with a point rotated around the camera 45 degrees left
	x_c = cam.T_cw.dot(numpy.append(x_w[:3], 1))[:3] # find position in camera
	x_c = R_about_axis(-numpy.pi/4, 'y').dot(x_c) # rotate position in camera
	x_w = cam.T_wc.dot(numpy.append(x_c, 1))[:3] # translate rotated point back to world coordinates
	x_w = numpy.concatenate([x_w, [0,0,0]]) # put back in terms of full state

	yhat, H = cam.h(x_w)

	# The point should appear right on the edge of the field of view
	assert numpy.allclose(yhat, [0, 540])
	plot_camera_perspective(cam.res_x, cam.res_y, yhat[0], yhat[1], 1)

	# And once again our Jacobian identity should hold: If we find another point along this ray from the camera,
	# yhat_mid will equal the same thing as yhat, so the diff will be zero, which should equal our Jacobian dotted with
	# the diff of the raw points.
	x_mid = (numpy.concatenate([cam.xyz, [0, 0, 0]]) + x_w) / 2
	assert numpy.allclose(H.dot(x_w - x_mid), [0, 0])


def test_camera_intrinsic_matrix():
	# Say we have a camera mounted at (0,0,10) looking east, pitched down 45 degrees, no roll
	rpy = numpy.array([0, numpy.pi/4, -numpy.pi/2])
	xyz = numpy.array([0, 0, 10])

	T_wk = get_t4_matrix(rpy, xyz)
	plot_frames(T_wk, "camera orientation in world frame", 1)

	# But we would like to be able to transform in and out of the camera's flipped coordinate system too
	T_kc = get_t4_matrix(numpy.array([-numpy.pi/2, 0, -numpy.pi/2]), numpy.array([0,0,0]))

	# The total transformation is the product of the two
	T_wc = T_wk.dot(T_kc)
	plot_frames(T_wc, "camera orientation with flipped camera axes in world frame", 1)

	# Say the camera's field of view is 90 degrees, x resolution is 1080 and y resolution is 1920. Get its intrinsic
	# matrix
	K = cam.K

	# Say we have a point out in the world at (0, -10, 0). The camera should be pointed right at it.
	xyz_w = numpy.array([0,-10,0])

	# It will get projected in to the camera according to:
	T_cw = numpy.linalg.inv(T_wc)
	xyz_c = T_cw.dot(numpy.append(xyz_w, 1))[:3]

	assert numpy.allclose(xyz_c, [0, 0, 10*numpy.sqrt(2)])

	uv1 = K.dot(xyz_c)/xyz_c[2]
	assert numpy.allclose(uv1, [960, 540, 1])
	plot_camera_perspective(1920, 1080, uv1[0], uv1[1], 1) # should be dead center

	# Now let's try moving the point off to the side. Field of view horizontally is 90 degrees, so we have 45 degrees
	# each side. So we should *just barely* be able to see the point if we rotate it 45 degrees around the camera.
	xyz_c = R_about_axis(-numpy.pi/4, 'y').dot(xyz_c) # rotate side to side around the camera's Y axis

	uv1 = K.dot(xyz_c)/xyz_c[2]
	assert numpy.allclose(uv1, [0, 540, 1])
	plot_camera_perspective(1920, 1080, uv1[0], uv1[1], 1) # should be on the left edge

	# Now let's check against a point nearer to the camera
	xyz_c = R_about_axis(numpy.pi/4, 'y').dot(xyz_c) # back to center
	xyz_c[2] = 5 # rather than 14.14...

	uv1 = K.dot(xyz_c)/xyz_c[2]
	assert numpy.allclose(uv1, [960, 540, 1])
	plot_camera_perspective(1920, 1080, uv1[0], uv1[1], 1) # dead center again

	xyz_c = R_about_axis(numpy.pi/4, 'y').dot(xyz_c)

	uv1 = K.dot(xyz_c)/xyz_c[2]
	assert numpy.allclose(uv1, [1920, 540, 1])
	plot_camera_perspective(1920, 1080, uv1[0], uv1[1], 1) # right edge


def test_get_3d_for_no_roll_camera():
	# Start with the point in the image that is in the exact middle both horizontally and vertically.
	x_img = numpy.array([960, 540], dtype=float)

	# Find the point out in the world at height 0
	xyz_w = cam.get_3d(x_img, cam.get_boresight_depth(x_img, 0))

	# The point should end up 10 meters away along the negative y axis. This is because the camera is up 10 z, yawed to
	# gaze along the -y axis, looking 45 degrees down, and we're asking for a point along the boresight at 0 height.
	# The boresight crosses the ground at [0, -10, 0]
	assert(numpy.allclose(xyz_w, numpy.array([0, -10, 0])))

	# Now let's look at the same point in the image but have it land 5 meters above the ground.
	xyz_w = cam.get_3d(x_img, cam.get_boresight_depth(x_img, 5))

	# The point should end up 5 meters away along the negative y axis and 5 meters up from ground
	assert(numpy.allclose(xyz_w, numpy.array([0, -5, 5])))

	# Now let's look at the same point in the image but have it land 2 meters above the ground.
	xyz_w = cam.get_3d(x_img, cam.get_boresight_depth(x_img, 2))

	# The point should end up 8 meters away along the negative y axis and 2 meters up from ground
	assert(numpy.allclose(xyz_w, numpy.array([0, -8, 2])))

	# Start with the point that is at the middle of the left part of the image.
	x_img = numpy.array([0.0, 540.0])

	# Get where that point is in the world so that it lands on the ground (h = 0)
	xyz_w = cam.get_3d(x_img, cam.get_boresight_depth(x_img, 0))

	# The point should travel 10sqrt(2) length along the camera's z-axis until it hits the ground. We already saw that
	# the y-component is -10. So if we imagine the triangle formed by the camera's z-axis from camera to ground and the
	# x-component, the point should have landed 10sqrt(2) along the NWU reference frame's postive x-axis.
	assert(numpy.allclose(xyz_w, numpy.array([10 * numpy.sqrt(2), -10, 0])))

	# Now go to the right edge of the image in the middle
	x_img = numpy.array([1920.0, 540.0])

	# Get where that point is in the world so that it lands on the ground (h = 0)
	xyz_w = cam.get_3d(x_img, cam.get_boresight_depth(x_img, 0))

	# The point should travel 10sqrt(2) length along the camera's z-axis until it hits the ground. We already saw that
	# the y-component is -10. So if we imagine the triangle formed by the camera's z-axis from camera to ground and by
	# the x-component, the point should have landed 10sqrt(2) along the NWU reference frame's negative x-axis.
	assert(numpy.allclose(xyz_w, numpy.array([-10 * numpy.sqrt(2), -10, 0])))

	# Now use every pixel down the vertical line in the middle of the image where u_img is fixed.
	for v_img in range(1080):
		x_img = numpy.array([960.0, v_img])
		# Pick a point that lands on the ground.
		xyz_w = cam.get_3d(x_img, cam.get_boresight_depth(x_img, 0))
		# The z-coordinate in the world frame for all of these should be 0 since that is what we wanted the height to
		# be. The x-coord should be zero since we're looking down the middle column of the image. Note that we aren't
		# checking the y
		assert numpy.allclose(xyz_w[[0, 2]], [0, 0])

	# Now pick the corners of the image along with all of the mid points. Choose to put the point at 2m above ground.
	for u_img in [0, 960, 1960]:
		for v_img in [0, 540, 1080]:
			x_img = numpy.array([u_img, v_img])
			xyz_w = cam.get_3d(x_img, cam.get_boresight_depth(x_img, 2))
			# The z-coordinate for all of these should be 2 since that is what we wanted the height to be. Note that
			# we aren't checking the x or y
			assert numpy.isclose(xyz_w[2], [2])


def test_get_3d_for_nx90deg_roll_camera():
	# Now let's test a camera that has roll to it. We'll keep using our camera with 0 roll, but then add three more
	# cameras that have +/-90deg and 180deg roll.
	cam_p90 = NWUCamera(1920, 1080, numpy.array([0,0,10]), numpy.array([numpy.pi/2, numpy.pi/4, -numpy.pi/2]),
						numpy.pi/2, 0.08)
	cam_n90 = NWUCamera(1920, 1080, numpy.array([0,0,10]), numpy.array([-numpy.pi/2, numpy.pi/4, -numpy.pi/2]),
						numpy.pi/2, 0.08)
	cam_180 = NWUCamera(1920, 1080, numpy.array([0,0,10]), numpy.array([numpy.pi, numpy.pi/4, -numpy.pi/2]),
						numpy.pi/2, 0.08)

	# Start with a point in the image at the center of the image for the camera with no roll. Get the point out in the
	# world that sits at on the ground.
	x_img_ref = numpy.array([960, 540], dtype=float)
	xyz_w_ref = cam.get_3d(x_img_ref, cam.get_boresight_depth(x_img_ref, 0))

	# For all three cameras with non-zero roll, the boresight pixel for all of their images should not be affected by
	# camera roll. They should all have the same point out in the world.
	assert numpy.allclose(xyz_w_ref, cam_p90.get_3d(x_img_ref, cam.get_boresight_depth(x_img_ref, 0)))
	assert numpy.allclose(xyz_w_ref, cam_180.get_3d(x_img_ref, cam.get_boresight_depth(x_img_ref, 0)))
	assert numpy.allclose(xyz_w_ref, cam_n90.get_3d(x_img_ref, cam.get_boresight_depth(x_img_ref, 0)))

	# Now get the point out in the world that sits on the ground for the point in the image of the no-roll camera that
	# sits in the middle of the top of the image. We'll call the point in the image x_img_ref.
	x_img_ref = numpy.array([960, 0], dtype=float)
	xyz_w_ref = cam.get_3d(x_img_ref, cam.get_boresight_depth(x_img_ref, 0))

	# For the camera with a +90deg roll, x_img_ref should now sit to the left 540 pixels of the boresight pixel. Make
	# sure the point in the world for this camera is the same for the camera with no roll.
	x_img = numpy.array([960-540, 540], dtype=float)
	assert numpy.allclose(xyz_w_ref, cam_p90.get_3d(x_img, cam_p90.get_boresight_depth(x_img, 0)))

	# For the camera with a 180deg roll, x_img_ref should now sit on the bottom row below the boresight pixel. Make
	# sure the point in the world for this camera is the same for the camera with no roll.
	x_img = numpy.array([960, 1080], dtype=float)
	assert numpy.allclose(xyz_w_ref, cam_180.get_3d(x_img, cam_180.get_boresight_depth(x_img, 0)))

	# For the camera with a -90deg roll, x_img_ref should now sit to the right 540 pixels of the boresight pixel. Make
	# sure the point in the world for this camera is the same for the camera with no roll.
	x_img = numpy.array([960+540, 540], dtype=float)
	assert numpy.allclose(xyz_w_ref, cam_n90.get_3d(x_img, cam_n90.get_boresight_depth(x_img, 0)))


def test_get_3d_for_camera_with_arbitrary_roll():
	# As a last test, create a series of cameras that step through a series of roll radians. Pick a point out in the
	# world, transform it into the camera frame, and then have the camera project that point back into the world. We
	# should get the exact point in the world back.
	xyz_w_ref = numpy.array([0, -9, 0])
	for roll_deg in range(-180, 180, 1):
		roll_rad = numpy.radians(roll_deg)
		cam_test = NWUCamera(1920, 1080, numpy.array([0,0,10]), numpy.array([roll_rad, numpy.pi/4, -numpy.pi/2]),
							numpy.pi/2, 0.08)

		uv = cam_test.h(xyz_w_ref, yhat_only=True)
		assert numpy.allclose(cam_test.get_3d(uv, cam_test.get_boresight_depth(uv, 0)), xyz_w_ref)
