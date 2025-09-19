def get_c_matrix(x_i:float, y_i:float, box_radius:float=1,
                 pixel_density:int=512) -> np.ndarray:
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
	z = 0
	for iteration in range(N_iterations):
		z = z * z + c
		for row_index, row in enumerate(z):
			for item_index, item in enumerate(row):
				if abs(item) > 2:
					escape_time_matrix[row_index, item_index] = iteration
					z[row_index,item_index] = 0
					c[row_index,item_index] = 0
	return escape_time_matrix

def main():
	x_i, y_i = -0.6643, 0.460221
	rad = 0.1

	start = time()

	c = get_c_matrix(x_i,y_i, box_radius=rad, pixel_density=1000)
	X = in_mandelbrot(c, N_iterations=200)

	print(f"Time elapsed: {time()-start:.2f} sec")

	plt.imshow(X, interpolation="gaussian", extent=[x_min, x_max, y_min, y_max], cmap="inferno", norm=PowerNorm(0.2))
	plt.gca().set_aspect("equal")
	plt.tight_layout()
	plt.show()
	plt.axis('off')
	plt.imsave(f"{Path(__file__).parent}/Saved/mbt_{x_i}_{y_i}_rad{rad}.png", X, cmap="inferno", dpi=300)

if __name__ == "__main__":
	from pathlib import Path
	import numpy as np
	import matplotlib.pyplot as plt
	from matplotlib.colors import PowerNorm
	from time import time

	main()