import numpy as np
from pathlib import Path	
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LogNorm

def get_c_matrix(x_i:float, y_i:float, box_radius:float=1,
                 pixel_density:int=512) -> np.ndarray:
	'''
	Create a complex matrix centered at (x_i, y_i) extending to box_radius in each direction.
	The matrix will have pixel_density^2 total elements.
	'''
	global x_min, x_max, y_min, y_max
	x_min = x_i - box_radius
	x_max = x_i + box_radius
	y_min = y_i - box_radius
	y_max = y_i + box_radius

	re = np.linspace(x_min, x_max, pixel_density, dtype=np.float64)
	im = np.linspace(y_min, y_max, pixel_density, dtype=np.float64)
    
	c_matrix = re[np.newaxis,:] + (im[:,np.newaxis] * 1j)
	return c_matrix

def in_mandelbrot(c:np.ndarray, N_iterations:int=20) -> np.ndarray:
	escape_time_matrix = np.zeros(c.shape, dtype=int)
	'''
	Determine the "escape time" for each point in the complex matrix c. 
	Escape time is how many iterations the point takes to exceed an absolute value of 2.
	'''

	z = 1j
	for iteration in range(N_iterations):
		z = np.square(z) + c
		for row_index, row in enumerate(z):
			for item_index, item in enumerate(row):
				if abs(item) > 2:
					escape_time_matrix[row_index, item_index] = iteration
					z[row_index,item_index] = 0 
					c[row_index,item_index] = 0
					#np.delete(z[row_index], item_index)
					#np.delete(c[row_index], item_index)
	return escape_time_matrix

def main(resolve:bool):
	from time import time
	x_i, y_i = 0, 0

	y_i = -y_i
	rad = 2

	if resolve:
		N_iterations = 250
		pixel_density = 900
	else:
		N_iterations = 250
		pixel_density = 100
	start = time()
	c = get_c_matrix(x_i,y_i, box_radius=rad, pixel_density=pixel_density)
	X = in_mandelbrot(c, N_iterations=N_iterations)
	print(f"Time elapsed: {time()-start:.2f} sec")

	if resolve:
		# I don't want to save the image with the axes.
		plt.axis('off')
		plt.imsave(f"{Path(__file__).parent}/Saved/mbt_{x_i}_{y_i}_rad{rad}_pixden{pixel_density}_Niter{N_iterations}.png", X, cmap="inferno", dpi=300)
		plt.axis('on')

	plt.imshow(X, interpolation="spline36", extent=[x_min, x_max, y_min, y_max], cmap="inferno", norm=PowerNorm(0.5))

	plt.gca().set_aspect("equal")
	plt.tight_layout()
	plt.show()

if __name__ == "__main__":
	
	main(resolve=True)
	