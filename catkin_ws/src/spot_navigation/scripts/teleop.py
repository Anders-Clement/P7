#!/usr/bin/env python

from sensor_msgs.msg import Joy
from std_srvs.srv import Trigger, TriggerResponse
import rospy

def cb(data):
    ServiceToCall = ""
    #print("enter callback")
    
    if data.buttons[1] and data.buttons[4] and data.buttons[5]:
        ServiceToCall = "/spot/estop/gentle"
        print("emergency STOP")
        rospy.wait_for_service(ServiceToCall)
        
        try:
            service = rospy.ServiceProxy(ServiceToCall, Trigger)
            resp = service()
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    elif data.buttons[3]:
        ServiceToCall= "/spot/sit"
        print("Sit down")
        rospy.wait_for_service(ServiceToCall)
        try:
            service = rospy.ServiceProxy(ServiceToCall, Trigger)
            resp = service()
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
    elif data.buttons[1]:
        ServiceToCall= "/spot/stand"
        print("Standing up")
        rospy.wait_for_service(ServiceToCall)
        
        try:
            service = rospy.ServiceProxy(ServiceToCall, Trigger)
            resp = service()
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)



def listener():
    rospy.Subscriber('joy', Joy, cb)
    rospy.spin()


if __name__ == '__main__':
    rospy.init_node('teleop_services_node')
    listener()

