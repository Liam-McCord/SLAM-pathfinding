import matplotlib.pyplot as plt
import enviroment
import numpy as np
import shapely
import shapely.plotting


class Enviroment():
    def __init__(self, map_area):
        self.map_area = map_area
        
    def refresh(self, robot):
        plt.clf()
        shapely.plotting.plot_polygon(self.map_area)
        shapely.plotting.plot_line(shapely.LineString(robot.path),color="r")
    def live_plot(self, robot):
        plt.ion
        plt.scatter(robot.path[:,0], robot.path[:,1])
        plt.show()
        plt.pause(0.01)
        #plt.clf()
        