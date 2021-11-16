#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32

def raw_callback(data):
    global max_velocity
    velocity = data.linear.x
   
    if velocity > max_velocity:
        data.linear.x = max_velocity
        pub.publish(data)
    else:
        pub.publish(data)

def limit_callback(data):
    global max_velocity
    max_velocity = data.data

pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
rospy.Subscriber("cmd_vel_raw", Twist, raw_callback, queue_size = 1)
rospy.Subscriber("vel_limit", Float32, limit_callback, queue_size = 1)

max_velocity = 1.6

if __name__ == '__main__':
    
    rospy.init_node('cmd_vel_limit_node', anonymous=True)
    rospy.spin()