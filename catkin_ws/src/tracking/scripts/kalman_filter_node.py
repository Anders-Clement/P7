import rospy
import tf2_ros

import numpy as np
import math

from std_msgs.msg import Float32
from geometry_msgs.msg import QuaternionStamped
from visualization_msgs.msg import Marker

from transtonumpy import msg_to_se3
from kalman_filter import kalmanFilter


class kalmanFilterNode:

    def __init__(self) -> None:
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)
        self.distance_to_robot_pub = rospy.Publisher('/distance_to_robot', Float32, queue_size=10)
        self.human_velocity_pub = rospy.Publisher('/human_velocity', Float32, queue_size=10)
        self.detection_sub = rospy.Subscriber('/people_detections', QuaternionStamped, self.detection_callback, queue_size=10)
        self.marker_pub = rospy.Publisher('kalman_filter_visualization_markers', Marker, queue_size=10)
        self.kalmanFilter = None
        self.lastGoodDetectionTime = rospy.Time.now()
        self.TIMEOUTLIMIT = 10
        
        

    def detection_callback(self, msg):
        # initialize kalman filter if not done yet, of if KF is lost
        if self.kalmanFilter is None or rospy.Time.now() - self.lastGoodDetectionTime > self.TIMEOUTLIMIT:
            # get position of spot, to initialize kalman filter
            try:
                self.trans = self.tfBuffer.lookup_transform('base_link', 'odom', rospy.Time.now())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                print("could not look up transform")
                return
            transformation_matrix = msg_to_se3(self.trans)
            # initialize the filter 2 meter behind the robot
            startPos = np.array([-2,0,0,1])
            transformedPos = np.matmul(transformation_matrix, startPos)
            self.kalmanFilter = kalmanFilter(rospy.Time.now())
            self.kalmanFilter.x = np.array([transformedPos[0],transformedPos[1],0,0,0,0])

        if self.kalmanFilter.predict(rospy.Time.now(), np.array([msg.quaternion.x, msg.quaternion.y, 0]) ):
            # find distance to robot and publish it
            dist = np.linalg.norm(np.array(self.trans.translation)[:2] - self.kalmanFilter.x[:2])
            distance_to_robot_msg = Float32()
            distance_to_robot_msg.data = dist
            self.distance_to_robot_pub.publish(distance_to_robot_msg)
            self.pub_KF_x_pos()
            velocity_of_human = math.sqrt(self.kalmanFilter.x[3]**2+self.kalmanFilter.x[4]**2)
            self.human_velocity_pub(velocity_of_human)

        self.pub_KF_pred_x_pos()
    

    def get_marker(self, color, xyPos, id):
        self.robotMarker = Marker()
        self.robotMarker.header.frame_id = 'odom'
        self.robotMarker.header.stamp    = rospy.get_rostime()
        self.robotMarker.ns = "kalman_filter"
        self.robotMarker.type = 2 # sphere
        self.robotMarker.action = 0
        self.robotMarker.pose.position.x = xyPos[0]
        self.robotMarker.pose.position.y = xyPos[1]
        self.robotMarker.pose.position.z = self.trans.translation.z
        self.robotMarker.pose.orientation.x = 0
        self.robotMarker.pose.orientation.y = 0
        self.robotMarker.pose.orientation.z = 0
        self.robotMarker.pose.orientation.w = 1.0
        self.robotMarker.scale.x = 0.1
        self.robotMarker.scale.y = 0.1
        self.robotMarker.scale.z = 0.1
        self.robotMarker.color.r = color[0]
        self.robotMarker.color.g = color[1]
        self.robotMarker.color.b = color[2]
        self.robotMarker.color.a = 1.0
        self.robotMarker.id = id
        self.robotMarker.lifetime = rospy.Duration(self.TIMEOUTLIMIT)


    def pub_KF_x_pos(self):
        marker = self.get_marker([255,0,0], self.kalmanFilter.x[:2],0)
        self.marker_pub.publish(marker)


    def pub_KF_pred_x_pos(self):
        marker = self.get_marker([0,255,0], self.kalmanFilter.x_pred[:2],1)
        self.marker_pub.publish(marker)    


if __name__ == '__main__':
    rospy.init_node('kalman_filter_node')
    node = kalmanFilterNode()
    rospy.spin()

