import helper_functions
import robot
import enviroment
import occupancy_map
import numpy as np
import matplotlib.pyplot as plt
import potential_navigation
plt.ion()


scan_range = 60
cell_repulsion_range = 50

#Map_Test = occupancy_map.OccupancyCellMap(helper_functions.png_to_grid("ExampleMap.png", 1000, 1000), 1)
Map_Test = occupancy_map.OccupancyCellMap(np.full((1000 + (scan_range * 2), 1000 + (scan_range * 2)), 1), 1) # Do not set the last part to 0.5 it screws up the grid :(

shapely_map = helper_functions.png_to_shapely_map("mazetraining1.jpg")
map_area = enviroment.Enviroment(shapely_map)
#print(len(Map_Test.grid))
#print(Map_Test.grid)    

#print(Map_Test.point_to_cell((15,15)))
#Map_Test.set_occupancy((15,15), 0)
#print(Map_Test.return_occupancy((1,1)))

#Map_Test.plot_the_map()

TestBot = robot.Robot(map_area, Map_Test, np.array([750.0,650.0]), 100, scan_range, 50)


potential_path = potential_navigation.PotentialExploration(TestBot, Map_Test, cell_repulsion_range, 0.1)
potential_path.simulate_path()
Map_Test.plot_the_map()

plt.pause(30)