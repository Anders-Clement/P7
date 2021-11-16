#! /usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32

class stupidClass():
    "Dense class for calculating density from a lidar scan"
    def __init__(self) -> None:
        self.listner_frame = "laser"
        rospy.init_node('visualizerofgraphs', anonymous=False)
        self.lidar_msgs = rospy.Subscriber('/scan', LaserScan, self.reciever_of_scans)
        self.denisty_msg = rospy.Publisher('/density', Float32)
        self.last_five = np.zeros(10) 
        self.previouse_value = 0.0

    def PolyArea(self, x,y):
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

    def low_pass(self):
        return (1.0 / len(self.last_five)) * np.sum(self.last_five)

    def reciever_of_scans(self, msg):
        d3points = []

        angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        for ranges, angle in zip(msg.ranges, angles):
            if ranges < msg.range_min or ranges > msg.range_max:
                continue

            d3points.append( (ranges * np.cos(angle), ranges * np.sin(angle)) )

        d3points = np.array(d3points)

        self.last_five[:-1] = self.last_five[1:]
        self.last_five[-1] = self.PolyArea(d3points[:, 0], d3points[:, 1])
        new_msg = Float32()
        new_msg.data = self.low_pass()
        self.denisty_msg.publish(new_msg)

        # plt.clf()
        # plt.plot(d3points[:,0], d3points[:,1])
        # plt.draw()
        # plt.pause(0.1)

if __name__=="__main__":
    new_class = stupidClass()
    rospy.spin()