#! /usr/bin/python3.6 

import sys

import jetson.inference
import jetson.utils

import cv2
import numpy as np

import rospy
from rospy.core import rospywarn
import tf2_ros
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import QuaternionStamped
from visualization_msgs.msg import Marker
import message_filters

import matplotlib.pyplot as plt
from transtonumpy import msg_to_se3

from numba import njit

import time

labels = {1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light", 11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench", 16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe", 27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase", 34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite", 39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard", 43: "tennis racket",
		  44: "bottle", 46: "wine glass", 47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl", 52: "banana", 53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut", 61: "cake", 62: "chair", 63: "couch", 64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet", 72: "tv", 73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard", 77: "cell phone", 78: "microwave", 79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator", 84: "book", 85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear", 89: "hair drier", 90: "toothbrush"}
frame_id_rot = {"frontleft_fisheye": cv2.ROTATE_90_CLOCKWISE, 
				"frontright_fisheye": cv2.ROTATE_90_CLOCKWISE,
				"back_fisheye": -1, 
				"right_fisheye": cv2.ROTATE_180, 
				"left_fisheye": -1}
#sys.stdout.close()
image_queue = [None, None, None, None, None]
camera_info_msg = [None for _ in range(10)]
sub_once = [None for _ in range(10)]
robot_id = 0


# deproject a list of points, using camera intrinsics
def deproject_points(points, K):
	u = points[:,0]
	v = points[:,1]
	z = points[:,2]
	x_over_z = (u - K[0, 2]) / K[0, 0]
	y_over_z = (v - K[1, 2]) / K[1, 1]
	x = x_over_z * z
	y = y_over_z * z

	return np.array([x,y,z])

# deproject a point using camera intrinsics
def deprojection(u, v, depth, dep_intrinsic):
	depth_z = depth[int(u), int(v)]

	x_over_z = (int(u) - dep_intrinsic[0, 2]) / dep_intrinsic[0, 0]
	y_over_z = (int(v) - dep_intrinsic[1, 2]) / dep_intrinsic[1, 1]
	z = depth_z #/ np.sqrt(x_over_z**2 + y_over_z**2)
	x = x_over_z * z
	y = y_over_z * z

	return np.array([x, y, z])



def inference_on_image(frame, net):
	img = jetson.utils.cudaFromNumpy(frame)

	# detect objects in the image (with overlay)
	detections = net.Detect(img)

	# print the detections
	# print("detected {:d} objects in image".format(len(detections)))

	return detections


# Define a callback for the Image message
def image_callback(img_msg, depth_msg, queue_id):
	# log some info about the image topic
	# rospy.loginfo(img_msg.header)
	# rospy.loginfo(depth_msg.header)
	image_queue[queue_id] = (img_msg, depth_msg)


# callback for camera_info, unregister when we have camera_info
def camera_info_sub(msg, index):
	# print(type(msg))
	camera_info_msg[index] = msg
	sub_once[index].unregister()


''' publish marker to rviz, default duration is 3 seconds'''
def pub_marker(pub, position, frame_id):
	global robot_id
	robotMarker = Marker()
	robotMarker.header.frame_id = frame_id
	robotMarker.header.stamp    = rospy.get_rostime()
	robotMarker.ns = "robot" + str(robot_id)
	robotMarker.id = 0
	robotMarker.type = 2 # sphere
	robotMarker.action = 0
	robotMarker.pose.position.x = position[0]
	robotMarker.pose.position.y = position[1]
	robotMarker.pose.position.z = position[2]
	robotMarker.pose.orientation.x = 0
	robotMarker.pose.orientation.y = 0
	robotMarker.pose.orientation.z = 0
	robotMarker.pose.orientation.w = 1.0
	robotMarker.scale.x = 0.1
	robotMarker.scale.y = 0.1
	robotMarker.scale.z = 0.1

	robotMarker.color.r = 0.0
	robotMarker.color.g = 1.0
	robotMarker.color.b = 0.0
	robotMarker.color.a = 1.0

	robotMarker.lifetime = rospy.Duration(3)
	pub.publish(robotMarker)
	robot_id += 1


''' return points in depth image where  bin_edges[min_index] < z < bin_edges[max_index]'''
@njit()
def getPoints(cutout, bin_edges, min_index, max_index):
	points = []
	for y in range(cutout.shape[0]):
		for x in range(cutout.shape[1]):
			if cutout[y, x] < bin_edges[max_index] and cutout[y,x] > bin_edges[min_index]:
				points.append([x,y,cutout[y,x]])
	return points


def main():
	net = jetson.inference.detectNet("ssd-mobilenet-v2", sys.argv, 0.5)
	# Initialize the ROS Node named 'opencv_example', allow multiple nodes to be run with this name
	rospy.init_node('jetson_inference_node', anonymous=True)

	tfBuffer = tf2_ros.Buffer()
	listener = tf2_ros.TransformListener(tfBuffer)

	# Initalize a subscriber to the "/camera/rgb/image_raw" topic with the function "image_callback" as a callback
	sub_image_frontleft = message_filters.Subscriber("/spot/camera/frontleft/image", Image)
	sub_depth_frontleft = message_filters.Subscriber("/spot/depth/frontleft/image", Image)
	ts = message_filters.TimeSynchronizer([sub_image_frontleft, sub_depth_frontleft], 10)
	ts.registerCallback(image_callback, (0))

	sub_image_frontright = message_filters.Subscriber("/spot/camera/frontright/image", Image)
	sub_depth_frontright = message_filters.Subscriber("/spot/depth/frontright/image", Image)
	ts2 = message_filters.TimeSynchronizer([sub_image_frontright, sub_depth_frontright], 10)
	ts2.registerCallback(image_callback, (1))

	sub_image_left = message_filters.Subscriber("/spot/camera/left/image", Image)
	sub_depth_left = message_filters.Subscriber("/spot/depth/left/image", Image)
	ts3 = message_filters.TimeSynchronizer([sub_image_left, sub_depth_left], 10)
	ts3.registerCallback(image_callback, (2))

	sub_image_right = message_filters.Subscriber("/spot/camera/right/image", Image)
	sub_depth_right = message_filters.Subscriber("/spot/depth/right/image", Image)
	ts5 = message_filters.TimeSynchronizer([sub_image_right, sub_depth_right], 10)
	ts5.registerCallback(image_callback, (3))

	sub_image_back = message_filters.Subscriber("/spot/camera/back/image", Image)
	sub_depth_back = message_filters.Subscriber("/spot/depth/back/image", Image)
	ts4 = message_filters.TimeSynchronizer([sub_image_back, sub_depth_back], 10)
	ts4.registerCallback(image_callback, (4))

	marker_pub = rospy.Publisher('robotMarker', Marker, queue_size=10)
	detection_pub = rospy.Publisher('people_detections', QuaternionStamped, queue_size=10)

	camera_info = ["/spot/camera/frontleft/camera_info",
				   "/spot/depth/frontleft/camera_info",
				   "/spot/camera/frontright/camera_info",
				   "/spot/depth/frontright/camera_info",
				   "/spot/camera/left/camera_info",
				   "/spot/depth/left/camera_info",
				   "/spot/camera/right/camera_info",
				   "/spot/depth/right/camera_info",
				   "/spot/camera/back/camera_info",
				   "/spot/depth/back/camera_info"]

	for i, topic in enumerate(camera_info):
		sub_once[i] = rospy.Subscriber(
			topic, CameraInfo, camera_info_sub, (i), queue_size=1)

	rate = rospy.Rate(1000.0)
	time_list = np.zeros(50)
	index = 0
	while not rospy.is_shutdown():
		for camera_num in range(0, 5):
			time_start = time.time()
			if rospy.is_shutdown():
				break

			# cehck for cam msg, skip if empty
			camera_info_index = camera_num*2
			if image_queue[camera_num] is None:
				continue
			img_msg, depth_msg = image_queue[camera_num]
			image_queue[camera_num] = None

			if camera_info_msg[camera_info_index] is None:
				print('lacking camera_info from ', img_msg.header.frame_id)
				continue
			if camera_info_msg[camera_info_index + 1] is None:
				print('lacking camera_info from ', depth_msg.header.frame_id)
				continue
			
			# get transform between cameras, necessary for registration
			try:
				trans = tfBuffer.lookup_transform(
					img_msg.header.frame_id, depth_msg.header.frame_id, rospy.Time())
			except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
				rate.sleep()
				continue

			# ensure depth and color image are from 'same' time
			if abs(img_msg.header.stamp.secs - depth_msg.header.stamp.secs) > 1:
				rospy.logerr(
					"Depth and Image timestamps out of alignement")
				continue
			
			# retrieve images from msgs
			im_orginal = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
				img_msg.height, img_msg.width, -1)
			depth_original = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(
				depth_msg.height, depth_msg.width, -1)

			

			camera_info_greyscale = camera_info_msg[camera_info_index]
			camera_info_depth = camera_info_msg[camera_info_index + 1]

			# undistort and register depth image to greyscale image
			# we register to grayscale image, thus further deprojections must be 
			# made from the grayscale camera
			depth_undistorted = cv2.undistort(depth_original, np.array(
				camera_info_depth.K).reshape(3, 3), camera_info_depth.D, None, None)

			depth_registered = cv2.rgbd.registerDepth(np.array(camera_info_depth.K).reshape(3, 3), 
														np.array(camera_info_greyscale.K).reshape(3, 3), 
														camera_info_greyscale.D, np.array(msg_to_se3(trans)), 
														depth_undistorted, (camera_info_greyscale.width, 
														camera_info_greyscale.height))

			# convert to rgb for detections, and rotate, since some cameras are rotated
			im_color = cv2.cvtColor(im_orginal, cv2.COLOR_GRAY2RGB)
			detect_image = im_color
			if frame_id_rot[img_msg.header.frame_id] != -1:
				detect_image = cv2.rotate(im_color, frame_id_rot[img_msg.header.frame_id])
			
			# detect and loop over the detections
			detections = inference_on_image(detect_image, net)
			for det in detections:
				if det.ClassID == 1: # class 1 == human
					
					# extract bounding box in original image
					# we extract top half, and center half of the bounding box
					# optionally uncomment cv2.rectangle to draw bounding box'es and the cutout
					height_box, width_box = (0, 0), (0, 0)
					if frame_id_rot[img_msg.header.frame_id] == -1:
						width = int(det.Right - det.Left)
						height = int(det.Bottom - det.Top)
						height_box = [int(det.Top), int(det.Bottom - height*0.5)]
						width_box =  [int(det.Left + width*0.25), int(det.Right) - int(width*0.25)]
						# cv2.rectangle(im_color, (int(det.Left), int(det.Top)), (int(det.Right), int(det.Bottom)), (0, 255, 0), thickness=2)

					elif frame_id_rot[img_msg.header.frame_id] == cv2.ROTATE_90_CLOCKWISE:
						height = int((camera_info_greyscale.height - det.Left) - (camera_info_greyscale.height - det.Right))
						width = int(det.Bottom - det.Top)
						height_box = [camera_info_greyscale.height - int(det.Right) + int(height*0.25),
									 camera_info_greyscale.height - int(det.Left) - int(height*0.25)]
						width_box = [int(det.Top), int(det.Bottom) - int(width/2)]
						# cv2.rectangle(im_color, (int(det.Top), camera_info_greyscale.height - int(det.Right)),(int(det.Bottom) , camera_info_greyscale.height - int(det.Left)), (0, 255, 0), thickness=2)

					elif frame_id_rot[img_msg.header.frame_id] == cv2.ROTATE_180:
						height = int(det.Bottom) - int(det.Top)
						width = int(det.Right - det.Left)
						height_box = [camera_info_greyscale.height - int(det.Bottom) + int(height*0.5), camera_info_greyscale.height - int(det.Top)]
						width_box = [camera_info_greyscale.width - int(det.Right) + int(width*0.25), camera_info_greyscale.width - int(det.Left) - int(width*0.25)]
						# cv2.rectangle(im_color, (camera_info_greyscale.width - int(det.Right), camera_info_greyscale.height - int(det.Bottom)), (camera_info_greyscale.width - int(det.Left) , camera_info_greyscale.height - int(det.Top)), (0, 255, 0), thickness=2)
					
					# cv2.rectangle(im_color, (width_box[0], height_box[0]), (width_box[1], height_box[1]), (0, 0, 255), thickness=2)

					
					# cut out corresponding depth image
					cutout = np.array(depth_registered[height_box[0]:height_box[1], width_box[0]:width_box[1]])
					# calculate histogram of depth values, using bins of size 200 mm
					bins = int(np.max(cutout)/200)
					if bins < 1:
						continue
					hist, bin_edges = np.histogram(cutout.reshape(-1), bins=bins)

					# find the largest bin
					largest_value, largest_index = 0, 0
					for i in range(1, len(hist)):
						if hist[i] > largest_value:
							largest_value = hist[i]
							largest_index = i
					
					# and retrieve the points from the largest bin
					min_index = largest_index
					max_index = largest_index + 1
					points2 = getPoints(cutout, bin_edges, min_index, max_index)

					# convert to list, and shift points by the bounding box's upper left corner
					# since points are found in the cutout depth image
					points2 = np.array(points2)
					if len(points2) <= 1:
						continue
					points2[:,0] = points2[:,0] + width_box[0]
					points2[:,1] = points2[:,1] + height_box[0]
					
					# deproject the points, using grayscale camera intrinsics, since we use the registered depth image!
					deprojected_points = deproject_points(points2, np.array(camera_info_greyscale.K).reshape(3,3))
					# calculate the detected position as center of mass
					detection_pos2 = [np.average(deprojected_points[0,:]), np.average(deprojected_points[1,:]), np.average(deprojected_points[2,:])]
					detection_pos2 = np.array(detection_pos2)
				
					# DEBUGGING: plots registered depth, and scatter plot of pointcloud and CoM shown
					# plt.close('all')
					# fig, ax = plt.subplots(2,2)
					# ax[0,0].imshow(im_color)
					# ax[0,1].imshow(depth_registered)
					# ax[1,0].imshow(cutout)
					# from mpl_toolkits import mplot3d
					# fig = plt.figure()
					# ax = plt.axes(projection='3d')
					# ax.set_xlabel('x')
					# ax.set_ylabel('y')
					# ax.set_zlabel('z')
					# ax.scatter3D(deprojected_points[0], deprojected_points[1], deprojected_points[2])
					# ax.scatter3D(detection_pos2[0], detection_pos2[1], detection_pos2[2])
					# plt.imshow(#im_color)
					# plt.show()

					# convert to meters from millimeters
					detection_pos2 = detection_pos2 / 1000

					# transform to targetFrame, such that we can project to the x-y plane
					targetFrame = 'base_link'
					try:
						trans = tfBuffer.lookup_transform(targetFrame, img_msg.header.frame_id, rospy.Time())
						trans = msg_to_se3(trans)
						point_transformed = np.matmul(trans, np.array([detection_pos2[0], detection_pos2[1], detection_pos2[2], 1]).T )
					except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
						rospywarn('Could not get transform from %s to %s', targetFrame, depth_msg.header.frame_id)
						rate.sleep()
						continue
					
					# send marker to rviz, note z-coordinate is set to 0
					pub_marker(marker_pub, [point_transformed[0], point_transformed[1], 0], targetFrame)
					# send detection to wire, note z-coordinate is set to 0
					detection_msg = QuaternionStamped()
					detection_msg.quaternion.x = point_transformed[0]
					detection_msg.quaternion.y = point_transformed[1]
					detection_msg.quaternion.z = 0
					detection_msg.quaternion.w = det.Confidence
					detection_msg.header = depth_msg.header
					detection_msg.header.frame_id = targetFrame
					detection_pub.publish(detection_msg)
			
			#print("Hz: ", camera_num, ": ", 1 / (end_time - time_start) )


			end_time = time.time()

			time_list[index] = 1 / (end_time - time_start)
			index += 1

			if index == 50:
				index = 0
				print("std: ", np.std(time_list), " mean: ", np.mean(time_list), " max: ", max(time_list), " min: ", min(time_list))

					

			
if __name__ == "__main__":
	main()	