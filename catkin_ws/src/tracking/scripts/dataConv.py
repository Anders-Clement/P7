#! /usr/bin/python3

from logging import fatal
from math import atan2
import matplotlib.pyplot as plt
import numpy as np
import cv2

import rospy
from rospy.core import rospywarn
import tf2_ros
import time

from geometry_msgs.msg import QuaternionStamped
from geometry_msgs.msg import PointStamped

from transtonumpy import msg_to_se3
import copy

class ourFilter():
	def __init__(self):
		rospy.init_node('dataConv', anonymous=True)
		self.tfBuffer = tf2_ros.Buffer()
		self.data = []
		self.start_time = None
		self.last_time = rospy.get_time()

	def reciever_of_data(self, msg):
		laser_point=PointStamped()
		laser_point.header = msg.header
		laser_point.point.x = msg.quaternion.x
		laser_point.point.y = msg.quaternion.y
		laser_point.point.z = msg.quaternion.z
		newPoint = copy.deepcopy(laser_point)
		newPoint2 = copy.deepcopy(laser_point)
		time_now = msg.header.stamp

		if self.start_time == None or time_now.to_sec() < self.start_time:
			self.start_time = time_now.to_sec()

		self.last_time = time.time()
		trans2 = None
		targetFrame = "odom"
		angleFrame = "base_link"
		try:
			trans = self.tfBuffer.lookup_transform(targetFrame, laser_point.header.frame_id, time_now, rospy.Duration(1))
			trans = msg_to_se3(trans)
			newPoint_stuff = np.matmul(trans, np.array([laser_point.point.x, laser_point.point.y, laser_point.point.z, 1]).T )
			newPoint.point.x = newPoint_stuff[0]
			newPoint.point.y = newPoint_stuff[1]
			newPoint.point.z = newPoint_stuff[2]

			trans2 = self.tfBuffer.lookup_transform(angleFrame, laser_point.header.frame_id, time_now, rospy.Duration(1))
			trans2 = msg_to_se3(trans2)
			newPoint_stuff = np.matmul(trans2, np.array([laser_point.point.x, laser_point.point.y, laser_point.point.z, 1]).T )
			newPoint2.point.x = newPoint_stuff[0]
			newPoint2.point.y = newPoint_stuff[1]
			newPoint2.point.z = newPoint_stuff[2]
			
			trans3 = self.tfBuffer.lookup_transform(targetFrame, "flat_body", time_now, rospy.Duration(1))
			next_tuple = (time_now.to_sec(), newPoint.point.x, newPoint.point.y, newPoint.point.z, 
											trans3.transform.translation.x, trans3.transform.translation.y, trans3.transform.translation.z, atan2(newPoint2.point.y, newPoint2.point.x))

			self.data.append(next_tuple)
		except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
			print(e)
			print("Noped the fuck out of that transform")

	def main(self):
		self.people_msgs = rospy.Subscriber('people_detections', QuaternionStamped, self.reciever_of_data)
		self.listener = tf2_ros.TransformListener(self.tfBuffer)
		#new_rate = rospy.Rate(10)
		while not rospy.is_shutdown():
			if (time.time() - self.last_time) > 50:
				self.data = sorted(self.data, key=lambda a: a[0])

				if len(self.data) == 0:
					continue

				first_time = self.data[0][0]
				for x in range(len(self.data)):
					self.data[x] = (self.data[x][0] - first_time, self.data[x][1], self.data[x][2], self.data[x][3], self.data[x][4], self.data[x][5], self.data[x][6], self.data[x][7]) 

				np_data = np.array(self.data)
				print(np_data)

				np.savetxt("foo.csv", np_data, delimiter=",")

			time.sleep(1)
			

if __name__=="__main__":
	myStuff = ourFilter()
	myStuff.main()
