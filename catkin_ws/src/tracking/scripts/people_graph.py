import matplotlib.pyplot as plt
import numpy as np
import cv2

import rospy
from rospy.core import rospywarn
import tf2_ros

from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PointStamped

class myViewerClass():
    def __init__(self) -> None:
        self.x_lim = 120
        self.update_interval = 1
        self.figure, self.ax = plt.subplots()
        self.ax.set_xlim((0, self.x_lim))

        self.selected_id = 2

        self.length_data = [0 for _ in range(self.x_lim)]
        self.angle_data = [0 for _ in range(self.x_lim)]


    def reciever_of_data(self, msg):
        for marker in msg.markers:
            laser_point=PointStamped()
            laser_point.header = marker.header
            laser_point.point = marker.pose.position.point
            newPoint = laser_point
            if marker.header.frame_id != "base_link":
                newPoint=self.listener.transformPoint("base_link",laser_point)
                print("Transformed")

            self.length_data.pop(0)
            self.length_data.append(np.sqrt( newPoint.point.x**2 + newPoint.point.y**2))

            self.angle_data.pop(0)
            self.angle_data.append(np.arctan2( newPoint.point.y, newPoint.point.X))
            break

    def update_graph(self):
        if not rospy.is_shutdown():
            self.ax.clear()
            self.ax.plot(self.length_data)
        


    def main(self):
        self.people_msgs = rospy.Subscriber('/visualization_markers/world_state', MarkerArray, self.reciever_of_data)

        rospy.init_node('visualizerofgraphs', anonymous=True)

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        rospy.Timer(rospy.Duration(self.update_interval), self.update_graph)

        rospy.spin()


if __name__=="__main__":
    myStuff = myViewerClass()
    myStuff.main()
