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


# Ideal vel controller
ideal_vel = 0
acc_max = 0.2
dec_max = -0.5
dist_threshold = 0.5


#Proportional controller gain
PosConstGain = 1
idealSpeedConst = 1.4
Kp1 = 1.0
Kp2 = .5
Kp3 = .5

#Cost function controller gains
CostK1 = .1 #Velocity
CostK2 = .25 #Distance


update_freq = 5.0

pub = rospy.Publisher('vel_limit', Float32, queue_size=5)

fig2, ax2 = None, None

grade1 = [0 for _ in range(1000)]
grade2 = [0 for _ in range(1000)]

x_axis2 = [x/update_freq for x in range(1000)]


def getIdealDistance(velocity):
    #di = 0.593*velocity**2 - 0.405*velocity + 1.78
    a = 0.65
    b = 1.42
    di = a*velocity + b
    return di

def IdealVelController(pos, vel):
    global ideal_vel, dec_max, acc_max, dist_threshold, spot_velocity
    idealDist = getIdealDistance(vel)
    Ts = 1 / update_freq
    error = idealDist-pos

    velchange = 0
    if error < -dist_threshold:
        velchange = Ts*dec_max
    elif error > dist_threshold:
        velchange = Ts*acc_max

    ideal_vel = ideal_vel + velchange
    if ideal_vel > 1.4: 
        ideal_vel = 1.4
    if ideal_vel < 0:
        ideal_vel = 0
    
    #print('vel_lim: ', ideal_vel, ' pos: ', pos, ' vel: ',vel, ' idealPos: ', idealDist, ' spot_vel: ', spot_velocity)

    return ideal_vel, idealDist, pos, vel, spot_velocity


def PController(pos, vel):
    global spot_velocity, PosConstGain, Kp1, Kp2, Kp3, idealSpeedConst

    idealPos = getIdealDistance(vel)

    # old proportional model:
    #velChange = 1.0/update_freq*((idealPos-pos)*Kp1+(vel-spot_velocity )*Kp2 + PosConstGain)
    # new proportional model:
    Ts = 1 / update_freq
    velChange =  ((idealPos - pos)*Kp1 + (vel - spot_velocity)*Kp2 + (idealSpeedConst - spot_velocity)*Kp3)*Ts
    vel_limit = spot_velocity + velChange
    #print('vel_lim: ', vel_limit, ' pos: ', pos, ' vel: ',vel, ' idealPos: ', idealPos, ' spot_vel: ', spot_velocity)

    if(vel_limit > 1.4):
        vel_limit = 1.4
    elif vel_limit < 0.0:
        vel_limit = 0.0
    #posChange = (idealPos-pos)*PosPGain
    #vel_limit = spot_velocity + posChange
    return vel_limit, idealPos, pos, vel, spot_velocity

def CostFuncController(pos,vel):
    global spot_velocity, CostPosGain, CostVelGain, CostK1, CostK2, grade1, grade2, x_axis2, fig2, ax2
    #gradient = [ 0.269531186697266*vel - 0.0952452172863454*pos - 0.154084299810647 , -0.0952452172863454*vel + 0.130468813302734*pos - 0.172481857335664 ]
    gradient = [1.74772627686573*CostK1*(0.609859100958587*vel - 0.83539725616591*pos + 1.10440853035332) + 1.96*CostK2*(1.02040816326531*vel - 1.42857142857143), CostK1*(-1.46004573622269*vel + 2.0*pos - 2.64403197928144)]
    gradient = -np.array(gradient)
    velChange =  1.0/update_freq*(gradient[0] + gradient[1])
    vel_limit = spot_velocity + velChange

    if fig2 == None:
        fig2, ax2 = plt.subplots(2)
    

    ax2[0].cla()
    ax2[1].cla() 
    grade1.pop(0)
    grade1.append(gradient[0])

    grade2.pop(0)
    grade2.append(gradient[1])

    ax2[0].plot(x_axis2, grade1)
    ax2[1].plot(x_axis2, grade2)  

    idealPos = getIdealDistance(vel)

    #print('cost: vel_lim: ', vel_limit, ' pos: ', pos, ' vel: ',vel, ' idealPos: ', idealPos, ' spot_vel: ', spot_velocity)
    
    if(vel_limit > 1.4):
        vel_limit = 1.4
    elif vel_limit < 0.0:
        vel_limit = 0.0
        
    return vel_limit, idealPos, pos, vel, spot_velocity

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
    global spot_velocity
    rospy.init_node('vel_limit_node')
    rospy.Subscriber("spot/odometry", Odometry, Spot_vel_callback)
    rospy.Subscriber("distance_to_robot", Float32, DistCallBack)
    rospy.Subscriber("human_velocity", Float32, Human_vel_Callback)

    controller_type = rospy.get_param('controller')
    gui_type = rospy.get_param('gui')

    fig, ax = None, None
    last_list = [0 for _ in range(1000)]

    spot_vel_log = [0 for _ in range(1000)]
    ideal_dist_log = [0 for _ in range(1000)]

    person_vel_log = [0 for _ in range(1000)]
    person_dist_log = [0 for _ in range(1000)]

    x_axis = [x/update_freq for x in range(1000)]
    if gui_type == "plot":
        fig, ax = plt.subplots(2)


    # if controller_type != 'cost' and controller_type != 'pid' and controller_type != "bin":
    #     rospy.logerr("Valid types: cost, pid")
    #     return

    rate = rospy.Rate(update_freq)
    while not rospy.is_shutdown():

        controller_type =  rospy.get_param('controller')

        vel_limit = 0.0
        idealpos = 0.0
        pos = 0.0
        vel = 0.0
        spot_local = 0.0
        global person_distance, person_velocity
        if controller_type[0] == 'p':
            vel_limit, idealpos, pos, vel, spot_local = PController(person_distance, person_velocity)
        elif controller_type[0] == 'b':
            vel_limit, idealpos, pos, vel, spot_local = IdealVelController(person_distance, person_velocity) 
        elif controller_type[0] == 'c':
            vel_limit, idealpos, pos, vel, spot_local = CostFuncController(person_distance, person_velocity)
        else:
            rate.sleep()
            continue

        pub.publish(vel_limit)

        if fig != None:
            ax[0].cla()
            ax[1].cla() 
            last_list.pop(0)
            last_list.append(vel_limit)

            spot_vel_log.pop(0)
            spot_vel_log.append(spot_local)

            ideal_dist_log.pop(0)
            ideal_dist_log.append(idealpos)

            person_vel_log.pop(0)
            person_vel_log.append(vel)

            person_dist_log.pop(0)
            person_dist_log.append(pos)

            ax[0].plot(x_axis, last_list, label="vel limit")
            ax[0].plot(x_axis, person_vel_log, label="person velocity")
            ax[0].plot(x_axis, spot_vel_log, label="spot velocity")  
            ax[1].plot(x_axis, person_dist_log, label="person dist")
            ax[1].plot(x_axis, ideal_dist_log, label="ideal dist")
            ax[1].legend()
            ax[0].legend()  
            plt.draw()
            plt.pause(0.01)

        rate.sleep()



if __name__ == '__main__':
    try:
        do_stuff()
    except rospy.ROSInterruptException:
        pass
