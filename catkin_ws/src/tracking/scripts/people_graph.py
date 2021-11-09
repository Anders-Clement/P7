import matplotlib.pyplot as plt
import numpy as np
import cv2

import rospy
from rospy.core import rospywarn
import tf2_ros

from geometry_msgs.msg import QuaternionStamped
from wire_msgs.msg import WorldState, ObjectState, Property
from problib.msg import PDF

class myViewerClass():
    def __init__(self) -> None:
        self.figure = plt.figure()
        self.x_lim = 120
        self.data = [0 for _ in range(120)]

    def reciever_of_data(self, msg):
        print(msg)


        self.data.pop(0)
        self.data.append()


    def main(self):
        self.people_msgs = rospy.Subscriber('/people_detections', QuaternionStamped, self.reciever_of_data)

        rospy.init_node('talker', anonymous=True)

if __name__=="__main__":
    myStuff = myViewerClass()
    myStuff.main()
