import matplotlib.pyplot as plt
import enviroment
import numpy as np
import shapely
import shapely.ops
import occupancy_map

class Robot(enviroment.Enviroment, occupancy_map.OccupancyCellMap):
    def __init__(self, enviroment, occupancy_grid, start_position, sensor_num, sensor_range, max_steps, timestep=1, path = []):
        
        fig = plt.figure(figsize = (6,6))
        ax = fig.add_subplot(111)
        ax.set_box_aspect(1)
        ax.grid()
        plt.gca().set_aspect('equal')
        
        self.ax = ax
        self.pose = start_position
        self.enviroment = enviroment
        self.occupancy_grid = occupancy_grid
        self.max_steps = max_steps
        self.path = np.array([self.pose, self.pose])
        self.sensor_range = sensor_range
        self.timestep = timestep
        self.intersection_points = []
        #self.sensor_vectors = []
        #self.sensor_readings = np.ones(sensor_num) + 99
        self.update_sensors()
        
    def move(self, pose_change):
        self.pose += pose_change
        self.path = np.resize(self.path, [self.timestep + 1, 2])
        self.path[self.timestep] = self.pose
        self.update_sensors()
         
    def update_sensors(self):
        #real_pose = self.sensor_offsets + self.pose
        #temp_sensors = []
        #for sensor in real_pose[0]:
        #    temp_sensor = shapely.LineString([tuple(self.pose), tuple(sensor)])
        #    temp_sensors.append(temp_sensor)
        
        n = 100
        r = self.sensor_range
        # Dictates the spacing of points
        intersection_points = []
        for i in range(n-1):
            t = np.linspace(0, 2*np.pi, n, endpoint=True)
            x = r * np.cos(t)
            y = r * np.sin(t)
            
            sensor_pose = (self.pose[0] + x[i], self.pose[1] + y[i])
            #point_interpolation = 
            sensor_vector = shapely.LineString([self.pose, sensor_pose])
            
            
            #print(intersection)
            #print(sensor_pose)
        
            if sensor_vector.intersects(self.enviroment.map_area):
                
                intersection = shapely.intersection(sensor_vector, self.enviroment.map_area)
                closest_intersection_point = shapely.ops.nearest_points(intersection, shapely.Point(self.pose))[0]

                intersection_points.append((closest_intersection_point.x, closest_intersection_point.y))
                #intersection_distance = np.linalg.norm(np.array([intersection.x, intersection.y]) - self.pose)
                sensor_vector = shapely.LineString([self.pose, (closest_intersection_point.x, closest_intersection_point.y)])
                num_points = 50
                new_points = [(sensor_vector.interpolate(i/float(num_points - 1), normalized=True).y, sensor_vector.interpolate(i/float(num_points - 1), normalized=True).x) for i in range(num_points)]
                #print(new_points)
                indexes = [self.occupancy_grid.point_to_cell(new_points[i]) for i in range(len(new_points))]
                #print(indexes)
                for i in indexes:
                    self.occupancy_grid.set_occupancy(i, -0.10)
            else:
                num_points = 50
                new_points = [(sensor_vector.interpolate(i/float(num_points - 1), normalized=True).y, sensor_vector.interpolate(i/float(num_points - 1), normalized=True).x) for i in range(num_points)]
                #print(new_points)
                indexes = [self.occupancy_grid.point_to_cell(new_points[i]) for i in range(len(new_points))]
                #print(indexes)
                for i in indexes:
                    self.occupancy_grid.set_occupancy(i, -0.1)
            for i in intersection_points:
                coords = self.occupancy_grid.point_to_cell((i[1], i[0]))
                self.occupancy_grid.set_occupancy(coords, 0.3)
        self.enviroment.refresh(self)
        #self.intersection_points = intersection_points
        