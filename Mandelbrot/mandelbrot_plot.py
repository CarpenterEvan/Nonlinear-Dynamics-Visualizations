import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LogNorm
from multiprocessing import Pool
from pathlib import Path	


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
		print(f"Dir not found\nmkdir -p {filename.parent}")
		os.system(f"mkdir -p {filename.parent}")
		plt.imsave(filename, escape_time_matrix, cmap=cmap)
	plt.close()

def in_mandelbrot(x_i:float, y_i:float, box_radius:float=1,
                 pixel_density:int=512, N_iterations:int=20, usePool:bool=True) -> np.ndarray:
	'''
	Determine the "escape time" for each point in the complex matrix c. 
	Escape time is how many iterations the point takes to exceed an absolute value of 2.
	'''
	starting_matrix = get_starting_matrix(x_i=x_i, y_i=y_i, box_radius=box_radius, pixel_density=pixel_density)
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
	x_i, y_i = -1.78, 0
	# -0.8181285, 0.20082240 # near spiral, slightly too high
	max_N_iterations = 550
	max_pixel_density = 1_500
	smallest_radius = 0.0001
	largest_radius = 2

	fps = 24
	seconds = 10
	N_frames = int(fps * seconds)

	y_i = -y_i # so moving based on imshow coords is more intuitive
	
	Saved = Path(__file__).parent / Path("Saved")
	
	frames = np.arange(start=1,stop=N_frames+1)


	# Experimentally paramaterizing the radius, N_iterations, and pixel density from frames
	# TODO: Zoom for radius should be parameterized as exponential
	# https://stackoverflow.com/questions/47818880/
	
	radius_values = np.geomspace(smallest_radius, largest_radius, N_frames)
	if True:
		import pandas as pd
		to_copy = pd.DataFrame(radius_values).to_clipboard()
		exit()
	pixel_density_values = (max_pixel_density - 2 * frames).astype(int)
	N_iter_values = (max_N_iterations - 2.5 * frames).astype(int)

	start = time()
	if makeFrames:
		subfolder = Saved / Path(f"frames/single_zoom/experiment/x{x_i}_y{y_i}_geomspace_different_speeds")
		loopstart = time()

		for frame in frames:
			index = frame-1
			frame_label_index = N_frames-frame
			radius = radius_values[index]
			N_iterations = N_iter_values[index]
			pixel_density = pixel_density_values[index]
			
			
			start = time()
			escape_time_matrix = in_mandelbrot(x_i=x_i, 
									  		   y_i=y_i,
									  		   box_radius=radius, 
									  		   pixel_density=pixel_density, 
									  		   N_iterations=N_iterations)
			
			print(f"{frame=: >3}/{N_frames: >3}\t{time()-start:.1f} seconds")

			filename = subfolder / Path(f"frame{frame_label_index:0>4}.png")
			plot_brot(escape_time_matrix=escape_time_matrix, extent=window, filename=filename, show=False)
		
		os.system(f"ffmpeg -r {fps}\
					-f image2\
					-s {pixel_density}x{pixel_density}\
					-pattern_type glob\
					-i '{subfolder}/*.png'\
					-vcodec libx264\
					-crf 18\
					-pix_fmt yuv420p\
					{subfolder}/mandelbrot_linear_parameterization.mp4")
		print(f"Total time elapsed: {time()-loopstart:.1f} seconds")
	else: 
		escape_time_matrix = in_mandelbrot(x_i=x_i, 
								  		   y_i=y_i,
								  		   box_radius=smallest_radius, 
								  		   pixel_density=pixel_density, 
								  		   N_iterations=N_iterations)
		
		plot_brot(escape_time_matrix=escape_time_matrix, extent=window, show=True)

	print(f"Time elapsed: {time()-start:.2f} sec")
	


if __name__ == "__main__":
	main(makeFrames=True)
