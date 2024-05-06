#Create a cell system
import matplotlib.pyplot as plt
import helper_functions
import numpy as np
class OccupancyCellMap():
    def __init__(self, occupancy_grid, cell_size, threshold=0.5):
        self.grid = occupancy_grid # The actual occupancy values of the grid, scaling from 0 to 1
        self.visited_grid = np.full(occupancy_grid.shape, 0.0)
        self.cell_size = cell_size # The physical size of the grid cells (dimensions are immaterial for now)
        self.threshold = threshold # the threshold of the grid cells for them to be considered occupied
    
    def point_to_cell(self, coords):
        # Gives the index of a cell in the data array given the physical coordinates
        return (round(coords[0] / self.cell_size), round(coords[1] / self.cell_size))
    
    def cell_to_point(self, index):
        # Returns the x,y point in 2-space of a grid cell index 
        return (round(index[0] * self.cell_size), round(index[1] * self.cell_size))

    def return_occupancy(self, index):
        # Returns the occupancy value of the requested index
        return self.grid[index[0], index[1]]
    
    def set_occupancy(self, index, occupancy):
        #self.grid[index[0]-1:index[0]+1, index[1]-1:index[1]+1] = np.exp(-self.grid[index[0]-1:index[0]+1, index[1]-1:index[1]+1])
        self.grid[index[0], index[1]] += occupancy
        if occupancy > 0:
            self.visited_grid[index[0], index[1]] = 1.0

        
    def scale_values(self):
        self.grid[self.grid > self.threshold] = 1
        self.grid[self.grid < self.threshold] = 0
        self.grid[self.grid == self.threshold] = 0.5

    def plot_the_map(self, alpha=1, min_val=0, origin='lower'):
        #plt.imshow(self.grid, vmin=min_val, vmax=1, origin=origin, interpolation='none', alpha=alpha, cmap = "Accent")
        self.scale_values()
        plt.imshow(self.grid, vmin=min_val, vmax=1, origin=origin, interpolation='none', alpha=alpha, cmap = "Accent") # The multiplication is the issue! Both should work independantly!
        

