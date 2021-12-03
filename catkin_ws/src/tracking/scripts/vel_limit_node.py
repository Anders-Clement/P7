#! /usr/bin/python3.6 

import rospy as rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
import numpy as np
import message_filters
import matplotlib.pyplot as plt

Max_speed = 1.4
spot_velocity = 0
person_velocity = 0
person_distance = 0

#Proportional controller gain
PosConstGain = 2
Kp1 = 1
Kp2 = 0.5

#Cost function controller gains
CostPosGain = 1
CostVelGain = 1

update_freq = 10

pub = rospy.Publisher('vel_limit', Float32, queue_size=5)

def getIdealDistance(velocity):
    #di = 0.593*velocity**2 - 0.405*velocity + 1.78
    a = 0.65
    b = 1.42
    di = a*velocity + b
    return di

def PController(pos, vel):
    global spot_velocity, PosConstGain, Kp1, Kp2

    idealPos = getIdealDistance(vel)

    velChange = 1.0/update_freq*((idealPos-pos)*Kp1+(vel-spot_velocity )*Kp2 + PosConstGain)
    vel_limit = spot_velocity + velChange
    if(vel_limit > 1.4):
        vel_limit = 1.4
    #posChange = (idealPos-pos)*PosPGain
    #vel_limit = spot_velocity + posChange
    return vel_limit

def CostFuncController(pos,vel):
    global spot_velocity, CostPosGain, CostVelGain
    gradient = [ 0.269531186697266*vel - 0.0952452172863454*pos - 0.154084299810647 , -0.0952452172863454*vel + 0.130468813302734*pos - 0.172481857335664 ]
    velChange =  1.0/update_freq*(gradient[0]*CostVelGain + gradient[1]*CostPosGain)
    vel_limit = spot_velocity + velChange 
    
    if(vel_limit > 1.4):
        vel_limit = 1.4
    return vel_limit

def Human_vel_Callback(human_vel_msg):
    global person_velocity
    person_velocity = float(human_vel_msg.data)    

def Spot_vel_callback(msg):
    global spot_velocity
    spot_velocity = np.linalg.norm([msg.twist.twist.linear.x, msg.twist.twist.linear.y])
    
def DistCallBack(dist_msg):
    global person_distance
    person_distance = float(dist_msg.data)

def do_stuff():
    rospy.init_node('vel_limit_node')
    rospy.Subscriber("spot/odometry", Odometry, Spot_vel_callback)
    rospy.Subscriber("distance_to_robot", Float32)
    rospy.Subscriber("human_velocity", Float32)

    controller_type = rospy.get_param('controller', 'cost')
    gui_type = rospy.get_param('controller', 'plot')

    fig, ax = None, None
    last_list = [0 for _ in range(50)]
    x_axis = [x/update_freq for x in range(50)]
    if gui_type == "plot":
        fig, ax = plt.subplots(1)


    if controller_type != 'cost' and controller_type != 'pid':
        rospy.logerr("Valid types: cost, pid")
        return

    rate = rospy.Rate(update_freq)
    while not rospy.is_shutdown():
        vel_limit = 0.0
        if controller_type[0] == 'p':
            vel_limit = PController(person_distance, person_velocity)
        else:
            vel_limit = CostFuncController(person_distance, person_distance)

        pub.publish(vel_limit)

        if ax != None:
            plt.cla()
            last_list.pop(0)
            last_list.insert(-1, vel_limit)
            ax.plot(x_axis, last_list)
            plt.draw()
            plt.pause(0.01)
        rate.sleep()



if __name__ == '__main__':
    try:
        do_stuff()
    except rospy.ROSInterruptException:
        pass
