import numpy as np
from pathlib import Path	
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LogNorm
import os
from multiprocessing import Pool
import pickle


def get_starting_matrix(x_i:float, y_i:float, box_radius:float=1,
                 pixel_density:int=512) -> np.ndarray:
	'''
	Create a complex matrix centered at (x_i, y_i) extending to box_radius in each direction.
	The matrix will have pixel_density^2 total elements.
	'''
	global window
	x_min = x_i - box_radius
	x_max = x_i + box_radius
	y_min = y_i - box_radius
	y_max = y_i + box_radius
	window = [x_min, x_max, y_min, y_max]

	re = np.linspace(x_min, x_max, pixel_density, dtype=np.float64)
	im = np.linspace(y_min, y_max, pixel_density, dtype=np.float64)
    
	starting_matrix = re[np.newaxis,:] + (im[:,np.newaxis] * 1j)
	return starting_matrix

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
	
def plot_brot(escape_time_matrix, show:bool=True, extent:list=None, cmap:str="inferno", norm=PowerNorm(1), filename:Path=Path("test.png")):
	'''
	Plotting commands for the Mandelbrot set
	'''
	plt.figure()
	plt.imshow(escape_time_matrix, interpolation="spline36", extent=extent, cmap=cmap, norm=norm)
	plt.gca().set_aspect("equal")
	plt.tight_layout()
	if show:
		plt.show()
	# I don't want to save the image with the axes.
	plt.axis('off')

	if os.path.exists(filename.parent):
		plt.imsave(filename, escape_time_matrix, cmap=cmap)
	else:
		os.system(f"mkdir -p {filename.parent}")
		plt.imsave(filename, escape_time_matrix, cmap=cmap)
	plt.close()

def in_mandelbrot(starting_matrix:np.ndarray, N_iterations:int=20, usePool:bool=True) -> np.ndarray:
	'''
	Determine the "escape time" for each point in the complex matrix c. 
	Escape time is how many iterations the point takes to exceed an absolute value of 2.
	'''
	c = starting_matrix
	escape_time_matrix = np.zeros_like(c, dtype=int)

	if usePool:
		with Pool() as pool:
			escape_time_matrix_rows = pool.map(
				escape_time_for_row,
				[(starting_matrix[i, :], N_iterations) for i in range(starting_matrix.shape[0])]
			)
		escape_time_matrix = np.array(escape_time_matrix_rows)
	else:
		for i in range(c.shape[1]):
			c_row = c[i]
			args = (c_row, N_iterations)
			escape_time_matrix[i] = escape_time_for_row(args)

	return escape_time_matrix

def main(makeFrames:bool=False):
	from time import time
	x_i, y_i = -0.8181285, 0.20082240

	y_i = -y_i
	N_iterations = 25
	pixel_density = 2_000
	N_frames = 100
	fps = 10 

	subfolder = Path(__file__).parent / Path(f"Saved/frames/zoom/pixden{pixel_density}_Niter{N_iterations}/x{x_i}_y{y_i}")
	start = time()
	if makeFrames:
		for index, rad in enumerate(np.geomspace(0.01,2,N_frames)):
			label_index = N_frames-index
			print(f"{index=}\t{rad=}")
			
			
			starting_matrix = get_starting_matrix(x_i, y_i, box_radius=rad, pixel_density=pixel_density)
			escape_time_matrix = in_mandelbrot(starting_matrix=starting_matrix, N_iterations=N_iterations)
			

			filename = subfolder / Path(f"frame{label_index:0>4}.png")
			plot_brot(escape_time_matrix=escape_time_matrix, extent=window, filename=filename, show=False)
	
	print(f"Time elapsed: {time()-start:.2f} sec")
	os.system(f"ffmpeg -r {fps} -f image2 -s {pixel_density}x{pixel_density} -pattern_type glob -i '{subfolder}/*.png' -vcodec libx264 -crf 18 -pix_fmt yuv420p {subfolder}/mandelbrot.mp4")


if __name__ == "__main__":
	main(makeFrames=True)
	
	