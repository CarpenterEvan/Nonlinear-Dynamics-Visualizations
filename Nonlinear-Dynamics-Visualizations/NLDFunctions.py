import matplotlib.pyplot as plt
import numpy as np

def plot_vector_field(f, fixed_points=None):
	x_range = [-4,4] # set x and y range
	y_range = [-4,4]
	graph_size = 10
	N = 3*graph_size-1 # Number of points on grid
	x_points = np.linspace(start = x_range[0], # make array of N points that spans x range 
							stop = x_range[1],
							num = N)
	y_points = np.linspace(start = y_range[0],
							stop = y_range[1],
							num = N)
	x,y = np.meshgrid(x_points,y_points) # creates a grid/ kronecker product from x, y arrays

	def dxdt_and_dydt(x,y): # evaluate the string f, which has x and y, so iterate through each point it has,
		return map(eval, f) # this gives the rate of change--> direction of vectors

	x_dot, y_dot = dxdt_and_dydt(x,y) # array of 

	Normalizing_Factor = np.hypot(x_dot, y_dot) # Find length of vector

	Normalizing_Factor[Normalizing_Factor==0]=1. # Corrects div/0 errors by replacing it with 1

	x_dot = 2*x_dot/Normalizing_Factor # normalize vector to length 1
	y_dot = 2*y_dot/Normalizing_Factor


	fig = plt.figure(figsize=(graph_size, graph_size), dpi=80)
	ax = fig.add_subplot(111,
							aspect='equal', # makes it square
							autoscale_on=False,
							xlim=(x_range[0], x_range[1]),
							ylim=(y_range[0], y_range[1]))
	plt.xlabel("x")
	plt.ylabel("y")
	plt.title("x'={}\n y' ={}".format(f[0],f[1]))
	plt.quiver(x,y,x_dot,y_dot, pivot="mid")
	if fixed_points !=None:
		x_val = [x[0] for x in fixed_points]
		y_val = [x[1] for x in fixed_points]
		plt.scatter(x_val, y_val)
	return plt.show()