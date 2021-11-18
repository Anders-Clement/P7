#!/usr/bin/env python

from sensor_msgs.msg import Joy
from std_srvs.srv import Trigger, TriggerResponse
from std_msgs.msg import Float32
import rospy


vel_limit = 1.6
vel_limit_pub = None
last_vel_change_time = None

def cb(data):
    global vel_limit, last_vel_change_time, vel_limit_pub
    # vel_limit buttons
    if data.buttons[0]:
        if rospy.Time.now().to_sec() - last_vel_change_time > .2:
            last_vel_change_time = rospy.Time.now().to_sec()
            vel_limit = vel_limit + 0.25
            if vel_limit > 1.6:
                vel_limit = 1.6
            vel_limit_pub.publish(Float32(vel_limit))       

    if data.buttons[2]: 
        if rospy.Time.now().to_sec() - last_vel_change_time > 1:
            last_vel_change_time = rospy.Time.now().to_sec()
            vel_limit = vel_limit - 0.25
            if vel_limit < 0.25:
                vel_limit = 0.25
            vel_limit_pub.publish(Float32(vel_limit))

    
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
    rospy.Subscriber('/joy', Joy, cb, queue_size=1)
    global vel_limit_pub
    vel_limit_pub = rospy.Publisher('/vel_limit', Float32, queue_size=10)
    global last_vel_change_time 
    last_vel_change_time = rospy.Time.now().to_sec()
    rospy.spin()


if __name__ == '__main__':
    rospy.init_node('teleop_services_node')
    listener()

