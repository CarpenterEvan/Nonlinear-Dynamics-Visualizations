import numpy as np
from pathlib import Path	
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LogNorm
from multiprocessing import Pool
def get_starting_matrix(x_i:float, y_i:float, box_radius:float=1,
                 pixel_density:int=512) -> np.ndarray:
	'''
	Create a complex matrix centered at (x_i, y_i) extending to box_radius in each direction.
	The matrix shape is (pixeldensity, pixeldensity).
	'''
	global window
	x_min = x_i - box_radius
	x_max = x_i + box_radius
	y_min = y_i - box_radius
	y_max = y_i + box_radius
	window = [x_min, x_max, y_min, y_max]

	re = np.linspace(x_min, x_max, pixel_density, dtype=np.float64)
	im = np.linspace(y_min, y_max, pixel_density, dtype=np.float64)
    
	c_matrix = re[np.newaxis,:] + (im[:,np.newaxis] * 1j)
	return c_matrix


def escape_time_for_row(args):
		c_row, N_iterations = args
		z_row = np.zeros_like(c_row)
		escape_time_row = np.zeros_like(c_row, dtype=int)

		for iteration in range(N_iterations):
			mask = np.abs(z_row) <= 2
			z_row[mask] = z_row[mask] ** 2 + c_row[mask]

			# Iteration should be added to the escape matrix indices that:
			# have corresponding z values that were just squared 
			# (mask)
			# are now above 2
			condition1 = np.abs(z_row) > 2
			# and have not been written to yet
			condition2 = escape_time_row == 0

			escape_time_row[(mask) & (condition1) & (condition2)] = iteration

		return escape_time_row

def in_mandelbrot(c:np.ndarray, N_iterations:int=20) -> np.ndarray:
	'''
	Determine the "escape time" for each point in the complex matrix c. 
	Escape time is how many iterations the point takes to exceed an absolute value of 2.
	'''

	escape_time_matrix = np.zeros_like(c, dtype=int)

	for i in range(c.shape[1]):
		c_row = c[i]

		args = (c_row, N_iterations)
		escape_time_matrix[i] = escape_time_for_row(args)

	return escape_time_matrix

def plot_brot(escape_time_matrix, extent:list=None, cmap:str="inferno", norm=PowerNorm(1), filename:str=f"test.png"):
	'''
	Plotting commands for the Mandelbrot set
	'''
	plt.figure()
	plt.imshow(escape_time_matrix, interpolation="spline36", extent=extent, cmap=cmap, norm=norm)

	plt.show()
	# I don't want to save the image with the axes.
	plt.axis('off')
	plt.imsave(f"{Path(__file__).parent}/Saved/{filename}", escape_time_matrix, cmap=cmap)

def main(resolve:bool):
	from time import time
	x_i, y_i = 0,0
	
	#-0.8181285, 0.20082240 # near a spiral

	y_i = -y_i
	rad = 2
	
	N_iterations = 25
	pixel_density = 3_000
	


	start = time()
	starting_matrix = get_starting_matrix(x_i, y_i, box_radius=rad, pixel_density=pixel_density)
	parallel=True
	if parallel:
		with Pool() as pool:
			escape_time_matrix_rows = pool.map(
				escape_time_for_row,
				[(starting_matrix[i, :], N_iterations) for i in range(starting_matrix.shape[0])]
			)
		escape_time_matrix = np.array(escape_time_matrix_rows)
	else:
		in_mandelbrot(c=starting_matrix, N_iterations=N_iterations)
	print(f"Time elapsed: {time()-start:.2f} sec")

	file_name = f"{x_i}_{y_i}_rad{rad}_pixden{pixel_density}_Niter{N_iterations}.png"
	plot_brot(escape_time_matrix=escape_time_matrix, extent=window, filename=file_name)

if __name__ == "__main__":
	
	main(resolve=True)
	