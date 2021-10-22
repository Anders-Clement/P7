#!/usr/bin/env python

from sensor_msgs.msg import Joy
from std_srvs.srv import Trigger, TriggerResponse
import rospy

def cb(data):
    ServiceToCall = ""
    if data.buttons[4] and data.buttons[6] and data.buttons[2]:
        ServiceToCall = "/spot/estop/gentle"
    elif data.buttons[0]:
        ServiceToCall= "/spot/sit"
    elif data.buttons[2]:
        ServiceToCall= "/spot/stand"
    

    
    rospy.wait_for_service(ServicetoCall)
    try:
        service = rospy.ServiceProxy(ServiceToCall, Trigger)
        resp = service()
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)


def listener():
    rospy.Subscriber('teleop_sub', Joy, cb)
    rospy.spin()


if __name__ == '__main__':
    rospy.init_node('teleop_services_node')
    listener()

