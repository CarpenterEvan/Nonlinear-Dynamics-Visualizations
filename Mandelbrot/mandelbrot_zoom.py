import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LogNorm
from multiprocessing import Pool
from pathlib import Path	


def get_starting_matrix(x_i:float, 
						y_i:float, 
						box_size:float=1,
                		resolution:tuple=(1920,1080)) -> np.ndarray:
	'''
	Create a complex matrix centered at (x_i, y_i).
	The matrix will have resolution[0] * resolution[1] total elements.
	box_size is the vertical size of the window wrt the complex plane. 
	The horizontal width of the viewing window is determined by the aspect ratio, which is calculated from resolution. 
	'''
	global window

	aspect_ratio = resolution[0]/resolution[1]

	x_min = x_i - box_size * aspect_ratio
	x_max = x_i + box_size * aspect_ratio
	y_min = y_i - box_size
	y_max = y_i + box_size
	window = [x_min, x_max, y_min, y_max]
	re = np.linspace(x_min, x_max, resolution[0], dtype=np.float64)
	im = np.linspace(y_min, y_max, resolution[1], dtype=np.float64)
    
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

			escape_time_row[(mask) & (condition1) & (condition2)] = iteration# - np.log(np.log(np.abs(z_row[mask])))/np.log(N_iterations)

		return escape_time_row
def in_mandelbrot(x_i:float, y_i:float, box_size:float=1,
                 resolution:tuple=(), N_iterations:int=20, usePool:bool=True) -> np.ndarray:
	'''
	Determine the "escape time" for each point in the complex matrix c. 
	Escape time is how many iterations the point takes to exceed an absolute value of 2.
	'''
	starting_matrix = get_starting_matrix(x_i=x_i, y_i=y_i, box_size=box_size, resolution=resolution)
	c = starting_matrix
	print(c)
	escape_time_matrix = np.full(c.shape, complex(1,1), dtype=complex)
	print(escape_time_matrix)
	if usePool:
		with Pool(4) as pool:
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

def plot_brot(escape_time_matrix, show:bool=True, extent:list=None, cmap:str="inferno", norm=PowerNorm(0.5), filename:Path=Path("test.png")):
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
	print(f"Saving image at: {filename}")
	plt.imsave(filename, escape_time_matrix, cmap=cmap)
	plt.close()


def main(makeFrames:bool=False, show=False):
	from time import time

	x_i, y_i = 0, 0
	# -0.8181285, 0.20082240 # near spiral, slightly too high
	zoom_from_radius = 100
	zoom_to_radius = 1
	#-0.743643887037158704752191506114774, 0.131825904205311970493132056385139 suggested by copilot
	
	N_iterations = 500 #if animiated, this is the initial number of iterations. 

	smallest_size = 200
	largest_size = 2

	fps = 4
	resolution = (1500,1500)# (1920, 1080)# 1080p
	seconds = 10
	N_frames = int(fps * seconds)

	y_i = -y_i # so moving based on imshow coords is more intuitive
	
	Saved = Path(__file__).parent / Path("Saved")


	# Experimentally paramaterizing the N_iterations and pixel density from frames
	# TODO: Zoom for radius should be parameterized as exponential?
	# https://stackoverflow.com/questions/47818880/
	
	# make lists of meta parameters	
	frames = np.arange(start=1,stop=N_frames+1)
	box_size_values = np.geomspace(smallest_size, largest_size, N_frames)
	N_iter_values = (  ((N_frames-frames)**2)/N_frames + N_iterations ).astype(int)

	#(max_N_iterations - ((max_N_iterations-50)/ N_frames) * frames).astype(int)

	####################################################################################
	start = time()
	if makeFrames:
		subfolder = Saved / Path(f"frames/single_zoom/experiment/x{x_i}_y{y_i}_geomspace_different_speeds")
		loopstart = time()
		try:
			for frame in frames:
				index = frame-1
				frame_label_index = N_frames-index
				box_size = box_size_values[index]
				N_iterations = N_iter_values[index]

				
				start = time()
				escape_time_matrix = in_mandelbrot(x_i=x_i, 
												   y_i=y_i,
												   box_size=box_size, 
												   resolution=resolution, 
												   N_iterations=N_iterations)
				
				print(f"{frame=: >3}/{N_frames: >3}\t{round(time()-start,1): >5} seconds",end="\r")

				filename = subfolder / Path(f"frame{frame_label_index:0>4}.png")
				plot_brot(escape_time_matrix=escape_time_matrix, extent=window, filename=filename, show=False)
			
			os.system(f"ffmpeg -r {fps}\
			 			-y\
						-f image2\
						-s {resolution[0]}x{resolution[1]}\
						-pattern_type glob\
						-i '{subfolder}/*.png'\
						-vcodec libx264\
						-crf 18\
						-pix_fmt yuv420p\
						{subfolder}/mandelbrot_linear_parameterization.mp4")
			print("\n")
			print(f"Total time elapsed: {time()-loopstart:.1f} seconds")
		except KeyboardInterrupt:
			print("Exiting...")
	else: 
		box_size = box_size_values[0]
		N_iterations = N_iter_values[0]
		escape_time_matrix = in_mandelbrot(x_i=x_i, 
								  		   y_i=y_i,
								  		   box_size=min(smallest_size,largest_size), 
								  		   resolution=resolution, 
								  		   N_iterations=N_iterations)
		
		plot_brot(escape_time_matrix=escape_time_matrix, extent=window, show=show)

	print(f"Time elapsed: {time()-start:.2f} sec")
	


if __name__ == "__main__":
	main(makeFrames=False)
