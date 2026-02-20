import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LogNorm
from multiprocessing import Pool
from pathlib import Path
from time import time
from PIL import Image
import sys

saved_folder = Path(__file__).parent / Path("Saved")

def get_starting_matrix(x_0:float, 
						y_0:float, 
						box_size:float=1,
                		resolution:tuple=(1920,1080)) -> np.ndarray:
	'''
	Create a complex matrix centered at (x_0, y_0).
	The matrix will have resolution[0] * resolution[1] total elements.
	box_size is the vertical size of the window wrt the complex plane. 
	The horizontal width of the viewing window is determined by the aspect ratio, which is calculated from resolution. 
	'''
	global window

	aspect_ratio = resolution[0]/resolution[1]

	x_min = x_0 - box_size * aspect_ratio
	x_max = x_0 + box_size * aspect_ratio
	y_min = y_0 - box_size
	y_max = y_0 + box_size
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

def calculate_escape_time_matrix(x_0:float, y_0:float, box_size:float=1,
                 resolution:tuple=(), N_iterations:int=20, usePool:bool=True) -> np.ndarray:
	'''
	Determine the "escape time" for each point in the complex matrix c. 
	Escape time is how many iterations the point takes to exceed an absolute value of 2.
	'''
	starting_matrix = get_starting_matrix(x_0=x_0, y_0=y_0, box_size=box_size, resolution=resolution)
	c = starting_matrix

	escape_time_matrix = np.full(c.shape, complex(1,1), dtype=complex)

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

def map_ETM_to_image(escape_time_matrix, 
					 show:bool=True,
					 extent:list=None, 
					 cmap:str="inferno", 
					 norm=PowerNorm(0.5), 
					 filename:Path=Path("test.png")):
	'''
	Maps the escape time matrix (ETM) to an image using `matplotlib.pyplot.imshow`
	'''
	plt.figure()
	plt.imshow(escape_time_matrix, interpolation="spline36", extent=extent, cmap=cmap, norm=norm)
	plt.gca().set_aspect("equal")
	plt.tight_layout()
	if show:
		plt.show()
	# I don't want to save the image with the axes.
	plt.axis('off')

	if filename is not None:
		if not os.path.exists(filename.parent):
			print(f"Dir not found\nmkdir -p {filename.parent}")
			os.system(f"mkdir -p {filename.parent}")
		else:
			plt.imsave(filename, escape_time_matrix, cmap=cmap)
		plt.imsave(filename, escape_time_matrix, cmap=cmap)
	plt.close()
	return filename

def FFmpegCommand(subfolder:Path, fps:int=24, resolution:tuple=(1920,1800), Timed:bool=False):
	'''
	Using FFmpeg to group the frames together into a video.\n
	Saved png frames should be named like frame0001.png, frame0002.png, etc.
	- `subfolder` is the path where the  Takes in the *.png files in the subfolder.
	- `fps` frames per second.
	-`resolution` number of pixels in the (width, height) of the video.
	- `Timed` Boolean to determine if the FFmpeg command will be timed.
	'''

	FFmpeg_start_time = time() if Timed else None
	ffmpeg_command = f"ffmpeg -r {fps}\
			 			-y\
						-f image2\
						-s {resolution[0]}x{resolution[1]}\
						-pattern_type glob\
						-i '{subfolder}/frame*.png'\
						-vcodec libx264\
						-crf 18\
						-pix_fmt yuv420p\
						{subfolder}/mandelbrot_linear_parameterization.mp4"
	os.system(ffmpeg_command)

	if Timed:
		FFmpeg_end_time = time()
		print(f"\nTotal time elapsed by FFmpeg: {FFmpeg_end_time-FFmpeg_start_time:.1f} seconds")

def generate_video(show=False, Timed:bool=True, onlyFirstAndLastFrame:bool=False):	
# **** Image Parameters ****
	x_0, y_0 = 0, 0 #Focus point.
		# -0.8181285, 0.20082240 # near spiral, slightly too high
		#-0.743643887037158704752191506114774, 0.131825904205311970493132056385139 suggested by copilot for some reason

	y_0 = -y_0 #This means moving the focus based on the imshow coords is more intuitive.
	N_iterations = 500 
# ~❦~❦ Video Parameters ~❦~❦

	# Zoom radius
	zoom_from_radius = 2
	zoom_to_radius = 3
	resolution = (1500,1500)# (1920, 1080)# 1080p
	seconds = 5
	fps = 4
	N_frames = int(seconds * fps)
	frames = np.arange(start=1,stop=N_frames+1)

	# Transformations of parameters
	N_iter_values = (  ((N_frames-frames)**2)/N_frames + N_iterations ).astype(int)
		# If animiated, N_iterations is the initial number of iterations. 
		# The N_iterations will change as a function of the frame number as:
	box_size_values = np.geomspace(zoom_from_radius, zoom_to_radius, N_frames)

		# Experimentally paramaterizing the N_iterations and pixel density from frames
		# TODO: Zoom for radius should be parameterized as exponential?
		# https://stackoverflow.com/questions/47818880/
# File making
	total_time_start = time() if Timed else None
	
	frames_folder = saved_folder / Path("frames")

	print(f"Making {N_frames} frames for video...")

	print(f"Focus at approximately {x_0:.2f},{y_0:.2f}...")

	print(f"Zoom from {zoom_from_radius} to {zoom_to_radius}...")


	default_leaf_directory = frames_folder / Path(f"x_{x_0}_y_{y_0}_from_{str(zoom_from_radius).replace('.','p')}_to_{str(zoom_to_radius).replace('.','p')}/")
	print(f"These frames will be saved in {default_leaf_directory} ...")

	new_leaf_directory = input(f"Create unique leaf directory name? [no]: ")

	if len(new_leaf_directory) > 0:
		leaf_directory = frames_folder / new_leaf_directory
	elif len(new_leaf_directory) == 0:
		leaf_directory = default_leaf_directory
	else:
		print("I'm not sure how you got here...")
		leaf_directory = default_leaf_directory
# Option to just make frames
	if onlyFirstAndLastFrame:
		image_paths = []
		for index in (0,-1):
			box_size = box_size_values[index]
			N_iterations = N_iter_values[index]
			test_frame = leaf_directory / Path(f"test_frame_{index}.png")
			escape_time_matrix = calculate_escape_time_matrix(x_0=x_0, 
											y_0=y_0,
											box_size=box_size, 
											resolution=resolution, 
											N_iterations=N_iterations)
		
			tmp_image = map_ETM_to_image(escape_time_matrix=escape_time_matrix, 
							extent=window, 
							show=show, 
							filename=test_frame)
			image_paths.append(tmp_image)

		PILImages = [Image.open(i) for i in image_paths]


		widths, heights = zip(*(image.size for image in PILImages))
		total_width = sum(widths)
		max_height = max(heights)
		combined_image = Image.new('RGB', (total_width, max_height))
		x_offset = 0
		for image in PILImages:
			combined_image.paste(image, (x_offset,0))
			x_offset += image.size[0]
		combined_image_filename = leaf_directory / Path("First_and_Last_Frame.png")
		combined_image.save(combined_image_filename)
		for image in PILImages:
			image.close()
			os.system(f"rm {image.filename}")
		return print(f"Image saved at {combined_image_filename}")
# Main video loop
	try:
		# Main frame-making loop
		for frame in frames:	
				#I want code to start at 0 but naming frames to start at 1
			frame_number = frame-1 
			#frame_label= N_frames-frame_number

			current_box_size = box_size_values[frame_number]
			current_N_iterations = N_iter_values[frame_number]
			filename = leaf_directory / Path(f"frame{frame_number:0>4}.png")

			calculation_time_start = time()
			escape_time_matrix = calculate_escape_time_matrix(x_0=x_0, y_0=y_0,
												box_size = current_box_size, 
												N_iterations = current_N_iterations,
												resolution = resolution)
			map_ETM_to_image(escape_time_matrix=escape_time_matrix, extent=window, filename=filename, show=False)
			print(f"{frame=: >4}/{N_frames: >4}\t{round(time()-calculation_time_start,0): >5} seconds",end="\r")
		print("\nFrames done! running FFmpeg command to create video...")
		FFmpegCommand(subfolder=leaf_directory, fps=fps, resolution=resolution, Timed=Timed)
	except KeyboardInterrupt:
		print("Ketboard Interrupt. Exiting...")
		exit()


	

	if Timed:
		total_time_end = time()
		print(f"Time elapsed: {total_time_end-total_time_start:.2f} sec")



if __name__ == "__main__":

	generate_video(onlyFirstAndLastFrame=False)
