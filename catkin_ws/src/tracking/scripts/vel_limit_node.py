#! /usr/bin/python3.6 

import rospy as rospy
from std_msgs.msg import Float32


MIN_DIST = 0.45
MAX_DIST = 1.20

spring_Pgain = 5.0
spring_Igain = 1.0

spring_val = 0
iError = 0

MAX_SPEED = 1.6
MIN_SPEED = 0.8


MAX_DENSITY = 10.0
MIN_DENSITY = 0.5

SPOT_SPEED = 0


pub = rospy.Publisher('vel_limit', Float32, queue_size=5)



def DistCallBack(msg):
    
    global spring_val, iError
    
    if( MIN_DIST > msg.data):
        error = msg.data - MIN_DIST
        iError = error + iError
        spring_val = error * spring_Pgain + spring_Igain * iError

    elif( msg.data > MAX_DIST):
        error = msg.data - MAX_DIST
        iError = error + iError
        spring_val = error * spring_Pgain + spring_Igain * iError

    else:
        iError = 0
    

def DensityCalĺback(msg):
    global SPOT_SPEED

    if(msg.data > MAX_DENSITY):
        SPOT_SPEED = MIN_SPEED + spring_val

    elif(MIN_DENSITY > msg.data):
        SPOT_SPEED = MAX_SPEED + spring_val

    else:
        SPOT_SPEED = (MIN_SPEED-MAX_SPEED)/(MAX_DENSITY-MIN_DENSITY)*msg.data + spring_val
    


def do_stuff():
    
    rospy.init_node('vel_limit_node', anonymous=True)

    rospy.Subscriber("distance_to_robot", Float32, DistCallback, queue_size=5)

    rospy.Subscriber("density", Float32, DensityCallback, queue_size=5)

    pub.Publisher(Float32(SPOT_SPEED))  

    rospy.spin()



if __name__ == '__main__':
    try:
        do_stuff()
    except rospy.ROSInterruptException:
        pass
