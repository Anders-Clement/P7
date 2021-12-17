#! /usr/bin/python3

from logging import fatal
import matplotlib.pyplot as plt
import numpy as np
import cv2

import rospy
from rospy.core import rospywarn
import tf2_ros

from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import PointStamped

from transtonumpy import msg_to_se3

from numba import njit
import copy


def data_behandling(new_data, inx):
	length_array = 0
	for x in new_data:
		if x[0] == inx:
			length_array += 1

	new_list = np.zeros((length_array,5))
	counting_index = 0
	for data in new_data:
		if data[0] == inx:
			new_list[counting_index, 0] = data[0]
			new_list[counting_index, 1] = data[1]
			new_list[counting_index, 2] = data[2] # X
			new_list[counting_index, 3] = data[3] # Y
			new_list[counting_index, 4] = data[4] # Z
			counting_index += 1

	

	average = copy.deepcopy(new_list[:, 2:])
	# averaged_list = [0.0 for _ in range(len(new_list))]
	average_window = average[0, :]

	for sample_index, row_values in enumerate(average):
		average_window = np.add(average_window*0.5, (average_window - row_values)*0.5)
		average[sample_index] = average_window

	vel_list = [0.0 for _ in range(len(average))]
	for sample_index in range(1, len(average)):
		vel_list[sample_index] = np.sqrt( (average[sample_index, 0] - average[sample_index - 1, 0])**2\
			+ (average[sample_index, 1] - average[sample_index - 1, 1])**2\
			+ (average[sample_index, 2] - average[sample_index - 1, 2])**2 ) / (new_list[sample_index, 1] - new_list[sample_index - 1, 1])


	average_vel = copy.deepcopy(vel_list)
	average_window_vel = average_vel[0]

	for sample_index, row_values in enumerate(vel_list):
		average_window_vel = np.add(average_window_vel*0.95, row_values*0.05)
		average_vel[sample_index] = average_window_vel

	return new_list, vel_list, average


class myViewerClass():
	def __init__(self):
		self.x_lim_secs = 360
		self.update_interval = 1
		self.figure, self.ax = plt.subplots(2)

		self.starting_time = 1000000000000
		self.last_time = 0

		self.selected_id = []

		self.tfBuffer = tf2_ros.Buffer()
		
		self.data = []
		self.last_length = 0


	def reciever_of_data(self, msg):
		for marker in msg.markers:
			#if self.selected_id == -1:
			#	self.selected_id = marker.id
			#	print("Select id {}".format(self.selected_id))

			if marker.text not in self.selected_id:
				self.selected_id.append(marker.text)

			if marker.header.stamp.to_sec() > self.starting_time:
				if marker.header.stamp.to_sec() < self.last_time:
					continue

				self.last_time = marker.header.stamp.to_sec()

				if marker.header.stamp.to_sec() > self.starting_time + self.x_lim_secs:
					continue

				break_bool = -1
				for index, ids in enumerate(self.selected_id):
					if str(ids) in marker.text:
						break_bool = index

				if break_bool == -1:
					continue

				laser_point=PointStamped()
				laser_point.header = marker.header
				laser_point.point = marker.pose.position
				newPoint = laser_point
				time_now = marker.header.stamp
				#time_now = rospy.get_rostime()

				#print(newPoint.header.frame_id)

				trans2 = None
				targetFrame = 'odom'
				if marker.header.frame_id != targetFrame:
					try:
						if marker.header.frame_id == "base_link":
							marker.header.frame_id = "flat_body"

						#self.listener.waitforTransform(targetFrame, marker.header.frame_id, time_now, rospy.Duration(3))

						trans = self.tfBuffer.lookup_transform(targetFrame, marker.header.frame_id, time_now, rospy.Duration(2))
						trans = msg_to_se3(trans)
						newPoint_stuff = np.matmul(trans, np.array([laser_point.point.x, laser_point.point.y, laser_point.point.z, 1]).T )
						newPoint.point.x = newPoint_stuff[0]
						newPoint.point.y = newPoint_stuff[1]
						newPoint.point.z = newPoint_stuff[2]

						trans2 = self.tfBuffer.lookup_transform(targetFrame, "flat_body", time_now, rospy.Duration(2))
						#print(trans2)
					except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
						print('Could not get transform from %s to %s', targetFrame, marker.header.frame_id)
						print(e)
						continue
				
				if trans2 != None:
					self.data.append((100,
									time_now.to_sec() - self.starting_time,
									# np.arctan2(trans2.transform.translation.y, trans2.transform.translation.x),
									# np.sqrt(trans2.transform.translation.x**2 + trans2.transform.translation.y**2), 
									trans2.transform.translation.x, trans2.transform.translation.y, trans2.transform.translation.z
									))
				else:
					print("Failed to get robot localization")

				self.data.append((break_bool, 
								marker.header.stamp.to_sec() - self.starting_time, 
								# np.arctan2(newPoint.point.y, newPoint.point.x), 
								# np.sqrt( newPoint.point.x**2 + newPoint.point.y**2),
								newPoint.point.x, newPoint.point.y, newPoint.point.z
								))

			elif marker.header.stamp.to_sec() < self.starting_time:
				self.starting_time = marker.header.stamp.secs
				self.data = []
				self.last_time = self.starting_time
				print("Clearing")

			# elif marker.header.stamp.secs == self.starting_time:
			# 	self.data = []
			# 	self.last_time = self.starting_time

	def update_graph(self):
		while not rospy.is_shutdown():
			plt.pause(0.01)
			
			
			#self.data = sorted(self.data, key= lambda a: a[1])
			#self.ax[2].clear()

			#new_data = np.array(self.data)
			if len(self.data) == 0:
				continue

			
			new_data = copy.deepcopy(self.data)
			new_list, vel_list, averaged_list = data_behandling(new_data, 100)
			if len(new_list) == self.last_length:
				plt.draw()
				plt.pause(1)
				continue
			
			self.last_length = len(new_list)

			self.ax[0].clear()
			self.ax[1].clear()

			self.ax[0].plot(new_list[:, 2:3], new_list[:, 3:4], label="Spot")
			self.ax[1].plot(new_list[:, 1:2], np.asfarray(vel_list), label="Spot")
			# self.ax[1].plot(new_list[:, 1:2], np.asfarray(averaged_list), label="Spot")

			for inx in range(len(self.selected_id)):

				new_list, vel_list, averaged_list = data_behandling(new_data, inx)

				self.ax[0].plot(new_list[:, 2:3], new_list[:, 3:4], label="Human")
				self.ax[1].plot(new_list[:, 1:2], np.asfarray(vel_list), label="Human")
				self.ax[0].plot(averaged_list[:, 0:1], averaged_list[:, 1:2], label="Average Human")


			self.ax[0].set_ylabel("Distance (m)")
			self.ax[0].set_xlabel("Time (secs)")
			self.ax[0].set_title("Distance from transform")

			# self.ax[1].set_ylabel("Speed (m/s)")
			# self.ax[1].set_xlabel("Time (secs)")
			# self.ax[1].set_title("Averaged distance")
			
			#self.ax[2].set_ylim([0,10])
			self.ax[1].set_ylabel("Speed (m/s)")
			self.ax[1].set_xlabel("Time (secs)")
			self.ax[1].set_title("Velocity")

			self.ax[0].legend()
			self.ax[1].legend()

			plt.draw()
		


	def main(self):
		self.people_msgs = rospy.Subscriber('/visualization_markers/world_state', MarkerArray, self.reciever_of_data)

		rospy.init_node('visualizerofgraphs', anonymous=True)
		self.listener = tf2_ros.TransformListener(self.tfBuffer)

		print("Go")

		self.update_graph()


if __name__=="__main__":
	myStuff = myViewerClass()
	myStuff.main()
