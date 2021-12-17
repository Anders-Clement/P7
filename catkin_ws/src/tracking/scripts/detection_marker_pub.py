#! /usr/bin/python3.6 


import rospy as rospy
from geometry_msgs.msg import QuaternionStamped
from visualization_msgs.msg import Marker


class markerPub():
    def __init__(self) -> None:
        self.detectionSub = rospy.Subscriber('/people_detections', QuaternionStamped, self.detection_callback, queue_size=50)
        self.markerPub = rospy.Publisher('robotMarker', Marker, queue_size=10)
        self.robot_id = 0

    def detection_callback(self, msg):
        robotMarker = Marker()
        robotMarker.header = msg.header
        #robotMarker.header.stamp = rospy.get_rostime()
        robotMarker.ns = "robot" + str(self.robot_id)
        robotMarker.id = 0
        robotMarker.type = 2 # sphere
        robotMarker.action = 0
        robotMarker.pose.position.x = msg.quaternion.x
        robotMarker.pose.position.y = msg.quaternion.y
        robotMarker.pose.position.z = msg.quaternion.z
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

        robotMarker.lifetime = rospy.Duration(300000)
        self.markerPub.publish(robotMarker)
        self.robot_id += 1



if __name__ == '__main__':
    rospy.init_node('detection_marker_pub')
    mp = markerPub()
    rospy.spin()