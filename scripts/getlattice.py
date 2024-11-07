import numpy as np  
  
def generate_cube_points(center, length, width, height, spacing):  
    # Center coordinates of the cube  
    x_center=center[0]
    y_center=center[1]
    z_center=center[2]



    # Number of points along each dimension  
    num_x = int(length / spacing) + 1  
    num_y = int(width / spacing) + 1  
    num_z = int(height / spacing) + 1  
      
    # Generate the grid points  
    x_points = np.linspace(x_center - length / 2, x_center + length / 2, num_x)  
    y_points = np.linspace(y_center - width / 2, y_center + width / 2, num_y)  
    z_points = np.linspace(z_center - height / 2, z_center + height / 2, num_z)  
      
    # Create a meshgrid  
    x_grid, y_grid, z_grid = np.meshgrid(x_points, y_points, z_points, indexing='ij')  
      
    # Reshape to get all points as a 2D array of shape (num_points, 3)  
    points = np.vstack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel())).T  
      
    return points  
  
# Example usage  
         
center=[-16.4,-3.4,0.0]
length=0
width=5.44
height=1.25
spacing=0.05

                
  
cube_points = generate_cube_points(center, length, width, height, spacing)  
  
# Print the first few points to verify  
print(cube_points.shape)
print(cube_points[:,0])