"""A bunch of functions to make various common plots. All take a time parameter so they expire and don't hold up tests.
"""

import numpy
from matplotlib import pyplot, patches
from mpl_toolkits.mplot3d import Axes3D


def plot_frames(T:numpy.ndarray, title: str="", t: float=100):
	"""plot the body and world frames for visual inspection

	:param T: 4x4 transformation matrix from body frame to world frame
	:param t: how long to show the plot in seconds
	:param label: 
	"""
	RR = T[:3,:3]
	p = T[:3,3]

	fig = pyplot.figure()
	ax = fig.gca(projection='3d')
	ax.plot([0,5],[0,0],[0,0],c='k')
	ax.plot([0,0],[0,5],[0,0],c='k')
	ax.plot([0,0],[0,0],[0,5],c='k')

	ax.plot([p[0],p[0]+RR[0,0]],[p[1],p[1]+RR[1,0]],[p[2],p[2]+RR[2,0]],c='r')
	ax.plot([p[0],p[0]+RR[0,1]],[p[1],p[1]+RR[1,1]],[p[2],p[2]+RR[2,1]],c='g')
	ax.plot([p[0],p[0]+RR[0,2]],[p[1],p[1]+RR[1,2]],[p[2],p[2]+RR[2,2]],c='b')

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	pyplot.title(title)
	pyplot.show(block=False)
	pyplot.pause(t)
	pyplot.close()


def plot_camera_perspective(x_res: int, y_res: int, u: float, v: float, t: float=100):
	"""plot where a point lies in a camera field of fiew

	:param x_res: number of pixels the camera has left to right
	:param y_res: number of pixles the camera has up to down
	:param u: where the point lies horizontally in camera projection space
	:param v: where the point lies vertically in camera projection space
	:param t: how long to show the plot in seconds
	"""
	fig = pyplot.figure()
	ax = fig.gca()

	rect = patches.Rectangle((0,0),x_res,y_res, linewidth=1, edgecolor='k', facecolor='none')
	ax.add_patch(rect)

	pyplot.plot(u,v,'b.')

	pyplot.xlabel('u')
	pyplot.ylabel('v')
	pyplot.gca().invert_yaxis()
	pyplot.show(block=False)
	pyplot.pause(t)
	pyplot.close()


def plot_2D_track(real_xs: list, filtered_xs: list, t: float=100):
	"""Plot actual and tracked 2D location, as well as actual and tracked velocities
	for each of the two dimensions if given.

	:param real_xs: locations where the object really was
	:param filtered_xs: locations where the object was tracked to be
	:param t: how long to show the plot in seconds
	"""
	if len(real_xs[0]) != 2 and len(real_xs[0]) != 4:
		raise ValueError("State is assumed to be [x, y] or [x, y, x', y']")
	if len(real_xs) != len(filtered_xs):
		raise ValueError("Real and filtered histories need to have the same length")

	if len(real_xs[0]) == 4: # then do velocity charts too
		pyplot.figure()
		pyplot.plot(range(len(real_xs)), [x[2] for x in real_xs], 'k')
		pyplot.plot(range(len(real_xs)), [x[2] for x in filtered_xs], 'b')
		pyplot.title('x speed')

		pyplot.figure()
		pyplot.plot(range(len(real_xs)), [e[3] for e in real_xs], 'k')
		pyplot.plot(range(len(real_xs)), [e[3] for e in filtered_xs], 'b')
		pyplot.title('y speed')

	pyplot.figure() # position chart last so it ends up on top
	pyplot.plot([x[0] for x in real_xs], [x[1] for x in real_xs], 'k')
	pyplot.plot([x[0] for x in filtered_xs], [x[1] for x in filtered_xs], 'b')
	pyplot.title('positions')

	pyplot.show(block=False)
	pyplot.pause(t)
	pyplot.close()

