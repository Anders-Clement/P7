#! /usr/bin/python3.6 

import rospy
import tf2_ros

import numpy as np
import math

from geometry_msgs.msg import Twist
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
        self.spot_vel_sub = rospy.Subscriber('/spot/cmd_vel', Twist)
        self.kalmanFilter = None
        self.lastGoodDetectionTime = rospy.Time.now().to_sec()
        self.TIMEOUTLIMIT = 10
        
    def Spot_vel_callback(msg):
        global spot_velocity
        spot_velocity = np.linalg.norm([msg.twist.twist.linear.x, msg.twist.twist.linear.y])
    
    def getIdealDistance(velocity):
        #di = 0.593*velocity**2 - 0.405*velocity + 1.78
        a = 0.73
        b = 1.32
        di = a*velocity + b
        return di

    def detection_callback(self, msg):
        try:
            self.trans = self.tfBuffer.lookup_transform('base_link', 'odom', msg.header.stamp)
            transformation_matrix_odom_body = msg_to_se3(self.trans)
            transformation_matrix_body_odom = np.linalg.inv(transformation_matrix_odom_body)
            # initialize kalman filter if not done yet, of if KF is lost
            if self.kalmanFilter is None or rospy.Time.now().to_sec() - self.lastGoodDetectionTime > self.TIMEOUTLIMIT:
                # initialize the filter at ideal distance behind spot given spot volocity
                spot_velocity_now = spot_velocity
                IdealDis = self.getIdealDistance(spot_velocity_now)

                startPos = np.array([IdealDis,0,0,1])
                transformedPos = np.matmul(transformation_matrix_body_odom, startPos)
                self.kalmanFilter = kalmanFilter(rospy.Time.now().to_sec())
                self.kalmanFilter.x = np.array([transformedPos[0],transformedPos[1],0,spot_velocity_now,0,0])
                print('init KF, x_y: ', self.kalmanFilter.x[:2])         

            detectionPos = np.matmul(transformation_matrix_body_odom, [msg.quaternion.x, msg.quaternion.y, 0, 1])
            detectionPos[2] = 0  
            detectionPos = detectionPos[:3] 
            if self.kalmanFilter.predict(rospy.Time.now().to_sec(), np.array(detectionPos) ):
                self.lastGoodDetectionTime = rospy.Time.now().to_sec()
                # find distance to robot and publish it
                # SPOTPOS IS WRONG
                spotPos = np.array([self.trans.transform.translation.x, self.trans.transform.translation.y])
                dist = np.linalg.norm(np.array(spotPos - self.kalmanFilter.x[:2]))
                distance_to_robot_msg = Float32()
                distance_to_robot_msg.data = dist
                self.distance_to_robot_pub.publish(distance_to_robot_msg)
                print('good detection, KF.x: ', self.kalmanFilter.x, detectionPos, spotPos, dist)
                velocity_of_human = math.sqrt(self.kalmanFilter.x[3]**2+self.kalmanFilter.x[4]**2)
                self.human_velocity_pub(velocity_of_human)


            self.pub_KF_x_pos() 
            self.pub_KF_pred_x_pos()
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            print('failed transform kalman filter')
            return
 
    

    def get_marker(self, color, xyPos, id):
        marker = Marker()
        marker.header.frame_id = 'odom'
        marker.header.stamp    = rospy.get_rostime()
        marker.ns = "kalman_filter"
        marker.type = 2 # sphere
        marker.action = 0
        marker.pose.position.x = xyPos[0]
        marker.pose.position.y = xyPos[1]
        marker.pose.position.z = self.trans.transform.translation.z
        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = 0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 1.0
        marker.id = id
        marker.lifetime = rospy.Duration(self.TIMEOUTLIMIT)
        return marker


    def pub_KF_x_pos(self):
        if self.kalmanFilter is not None:
            x_pos = self.kalmanFilter.x[:2]
            marker = self.get_marker([255,0,0], x_pos,0)
            self.marker_pub.publish(marker)


    def pub_KF_pred_x_pos(self):
        if self.kalmanFilter is not None:
            pred_pos = self.kalmanFilter.x_pred[:2]
            marker = self.get_marker([0,255,0], pred_pos,1)
            self.marker_pub.publish(marker)    


if __name__ == '__main__':
    rospy.init_node('kalman_filter_node')
    node = kalmanFilterNode()
    rospy.spin()

