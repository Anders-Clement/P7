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

from std_msgs.msg import Float32
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
		self.human_velocity = [] #timestamp, speed
		self.human_dist = [] #timestamp, distance

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
			next_tuple = (time_now.to_sec(), 
							newPoint.point.x, 
							newPoint.point.y, 
							newPoint.point.z, 
							trans3.transform.translation.x, 
							trans3.transform.translation.y, 
							trans3.transform.translation.z, 
							atan2(newPoint2.point.y, newPoint2.point.x))

			self.data.append(next_tuple)
		except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
			print(e)
			print("Noped the fuck out of that transform")

	def velocity_reciever(self, msg):
		self.human_velocity.append([rospy.get_time(), msg.data])

	def dist_reciever(self, msg):
		self.human_dist.append([rospy.get_time(), msg.data])


	def main(self):
		self.people_msgs = rospy.Subscriber('/good_detections', QuaternionStamped, self.reciever_of_data)
		self.human_vel_msgs = rospy.Subscriber('/human_velocity', Float32, self.velocity_reciever)
		self.human_dist_msgs = rospy.Subscriber('/distance_to_robot', Float32, self.dist_reciever)
		self.listener = tf2_ros.TransformListener(self.tfBuffer)
		#new_rate = rospy.Rate(10)
		while not rospy.is_shutdown():
			time.sleep(1)

		self.data = sorted(self.data, key=lambda a: a[0])

		if len(self.data) == 0:
			return

		np_data = np.array(self.data)
		first_time = copy.deepcopy(np_data[0, 0])
		np_data[:, 0] = np_data[:, 0] - first_time

		hu_vel = np.array(self.human_velocity)
		hu_vel[:, 0] = hu_vel[:, 0] - first_time

		hu_dist = np.array(self.human_dist)
		hu_dist[:,0] = hu_dist[:, 0] - first_time

		all_data = np.zeros((max(np_data.shape[0], hu_vel.shape[0], hu_dist.shape[0]), 12))
		all_data[:np_data.shape[0], 0:8] = np_data
		all_data[:hu_vel.shape[0], 8:10] = hu_vel
		all_data[:hu_dist.shape[0], 10:12] = hu_dist

		np.savetxt("foo.csv", all_data, delimiter=",", header="stamp,people_x,people_y,people_z,spot_x,spot_y,spot_z,angle,stamp,human_vel,stamp,human_dist")
	

if __name__=="__main__":
	myStuff = ourFilter()
	myStuff.main()
