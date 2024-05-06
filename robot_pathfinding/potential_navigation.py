import occupancy_map
import helper_functions
import robot
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
class PotentialExploration(robot.Robot):
    def __init__(self, robot, map, grid_range, dt):
        self.robot = robot
        self.map = map
        self.grid_range = grid_range
        self.dt = dt

    def simulate_path(self):
        force_matrix = calculate_pulls(self.grid_range)
        y = np.array([self.robot.pose[0], self.robot.pose[1], 0.0, 3.0])
    
        for i in range(self.robot.max_steps):
            #print(self.robot.pose)

            y += self.timestep(y, force_matrix)
            y[2:] = np.clip(y[2:], -3.0, 3.0)
            #print(y)
            self.robot.move((y[2], y[3]))
            self.robot.enviroment.live_plot(self.robot)
            self.robot.timestep += 1
            #time.sleep(0.1)
            #print(y)
            #print(self.robot.path)

                   
    def timestep(self, y, force_matrix):
        y_new = np.zeros(4)
        robot_index = self.map.point_to_cell(self.robot.pose)
        #self.map.scale_values()
        grid_slice = self.map.grid[robot_index[0] - self.grid_range : robot_index[0] + self.grid_range + 1, robot_index[1] - self.grid_range : robot_index[1] + self.grid_range + 1]
        visited_grid_slice = self.map.visited_grid[robot_index[1] - self.grid_range : robot_index[1] + self.grid_range + 1, robot_index[0] - self.grid_range : robot_index[0] + self.grid_range + 1]
        #print(np.count_nonzero(visited_grid_slice == 1.0))
        force_x_from_prev, force_y_from_prev = sum_path_repulsion(self, self.robot.pose, self.robot.path, lambda x: 5 * np.exp((-x ** 2) / 100))
        #print(force_x_from_prev, force_y_from_prev)
        force_x = np.sum(force_matrix[:,:,0] * (visited_grid_slice * grid_slice.T)) + force_x_from_prev
        force_y = np.sum(force_matrix[:,:,1] * (visited_grid_slice * grid_slice.T)) + force_y_from_prev
        #print (force_x, force_y)
        #print( force_x, force_y)
        accel = np.array([force_x, force_y])
        y_new[2:] += accel * self.dt
        y_new[:2] += (y[2:] + y_new[2:]) * self.dt
        #print(y_new)
        return y_new

#def points():
                                                                                                                                                                                                                                                                    
def create_distance_matrix(grid_range, cell_size):
    distance_matrix = np.zeros((grid_range * 2 + 1, grid_range * 2 + 1, 2))
    for j in range(grid_range * 2 + 1):
        for i in range(grid_range * 2 + 1):

            distance_matrix[i,j,0] = -(j - (round(grid_range)))
            distance_matrix[i,j,1] = -(i - (round(grid_range)))
    
    distance_matrix *= cell_size # Scaling factor
    #print(distance_matrix[:,:,0])
    return distance_matrix


def force_equation(distance_vector, distance_mag, equation1, equation2):
    scaling = equation1(distance_mag) + equation2(distance_mag)
    
    distance_vector[:,:,0] *= scaling
    distance_vector[:,:,1] *= scaling
    #print(distance_vector)
    return distance_vector, scaling
    #return equation(distance_mag) * distance_vector # Gives an acceleration vector

def calculate_pulls(grid_range):
    distances = create_distance_matrix(grid_range,1)
    #print(distances.shape)
    distance_x = np.array(distances[:,:,0])
    distance_y = np.array(distances[:,:,1])
    
    distance_mag = (distance_x ** 2 + distance_y ** 2) ** 0.5
    #print(distance_mag)
    #print(distance_x.shape())
    force_vectorised = np.vectorize(force_equation)
    #print(distances[:,:,0])
    #print(distances)
    force_matrix, scaling = force_equation(distances, distance_mag, lambda x: 5000 * np.exp((-x ** 2) / 20), lambda x: 150 * np.exp((-x ** 2) / 50))
    #print(force_matrix)
    #print(force_matrix)
    
    # Temp pandas debugging
    #m,n,r = distance_mag.shape
    #out_arr = force_matrix.tolist()
    #out_df = pd.DataFrame(out_arr)
    #df = pd.DataFrame (force_matrix)
    #filepath = 'forcematrix.xlsx'
    #out_df.to_excel(filepath, index=False)  
    
    return force_matrix


def sum_path_repulsion(self, pose, previous_steps, equation):
    distance_matrix = previous_steps[:] - pose
    #print(previous_steps)
    distance_mag = (distance_matrix[:,0] ** 2 + distance_matrix[:,1] ** 2) ** 0.5
    
    scaling = -equation(distance_mag)
    
    distance_matrix[:,0] *= scaling
    distance_matrix[:,1] *= scaling
    
    force_x = np.sum(distance_matrix[:,0])
    force_y = np.sum(distance_matrix[:,1])
    return force_x, force_y

#print(create_distance_matrix(2, 2))
#print(force_equation(2, lambda x: 50 / x ** 4))
