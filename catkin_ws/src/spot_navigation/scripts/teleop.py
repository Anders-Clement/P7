#!/usr/bin/env/python3

from rospy.core import rospywarn
from sensor_msgs.msg import Joy
from std_srvs.srv import Trigger, TriggerResponse
from std_msgs.msg import Float32
import rospy
import numpy as np

def genRampTest(seconds, min, max, frequency, delayToZero=10):
    test = np.zeros((seconds*frequency + 2, 2))

    for i in range(len(test) - 1):
        test[i][0] = i / frequency
        if i > 0:
            test[i][1] = min + (max-min)/ ((seconds*frequency)/(i))
        else:
            test[i][1] = min

    test[len(test)-1] = [seconds + delayToZero, 0]
    return np.array(test)

def genStepTest(max_vel):
    step_list = []
    step_list.append([0, max_vel])
    step_list.append([60, 0])
    step_list.append([70, max_vel*0.66])
    step_list.append([130, 0])
    step_list.append([140, max_vel*0.33])
    step_list.append([200, 0])
    step_list.append([210, max_vel/2])
    step_list.append([270, max_vel])
    step_list.append([330, 0])

    return np.array(step_list)


def genStairCaseTest(min, max):
    stair_list = []
    step_size = (max-min) / 3
    stair_list.append([0, min+step_size])
    stair_list.append([10, min])
    stair_list.append([20, min+step_size])
    stair_list.append([30, min+step_size*2])
    stair_list.append([40, min+step_size*3])
    stair_list.append([50, min+step_size*2])
    stair_list.append([60, min+step_size])
    stair_list.append([70, min])
    stair_list.append([80, 0])

    return np.array(stair_list)

def genTests():
    testList = []
    testList.append(genStepTest(1.6))
    testList.append(genStairCaseTest(1,1.6))
    testList.append(genRampTest(5, 0, 1.6, 10))

    return testList






vel_limit = 1.6
vel_limit_pub = None
last_vel_change_time = None
change_vel_limit_delay = 0.1
not_ready_for_test = True

def cb(data):
    global vel_limit, last_vel_change_time, vel_limit_pub, change_vel_limit_delay
    # vel_limit buttons
    if data.buttons[0]:
        global not_ready_for_test
        if not_ready_for_test:
            not_ready_for_test = False
    if data.buttons[13] and not_ready_for_test:
        if rospy.Time.now().to_sec() - last_vel_change_time > change_vel_limit_delay:
            last_vel_change_time = rospy.Time.now().to_sec()
            vel_limit = vel_limit + 0.25
            if vel_limit > 1.6:
                vel_limit = 1.6
            vel_limit_pub.publish(Float32(vel_limit))       

    if data.buttons[14] and not_ready_for_test: 
        if rospy.Time.now().to_sec() - last_vel_change_time > change_vel_limit_delay:
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
    global vel_limit_pub, vel_limit
    vel_limit_pub = rospy.Publisher('/vel_limit', Float32, queue_size=10)
    vel_limit_pub.publish(Float32(vel_limit))
    global last_vel_change_time 
    global not_ready_for_test

    while not rospy.is_shutdown():
        last_vel_change_time = rospy.Time.now().to_sec()
        not_ready_for_test = True

        testList = genTests()
        testList.pop(0)
        rate = rospy.Rate(100)
        for test in testList:
            while not_ready_for_test:
                rospy.sleep(1)
            print('starting test!')
            startTime = rospy.Time.now()  
            i = 0

            while not rospy.is_shutdown():
                now = rospy.Time.now()
                if (now - startTime).to_sec() > test[i][0]:
                    print('next step in test: ', test[i], ' out of ', len(test), ' steps')
                    vel_limit = test[i][1]
                    vel_limit_pub.publish(Float32(vel_limit)) 
                    i += 1
                    if i >= len(test):
                        break

                try:
                    rate.sleep()
                except Exception as e:
                    print(e)   
            
            not_ready_for_test = True

        print('tests are done, returning to manual control. Press "x" to start testing again. (starts immediately)')


if __name__ == '__main__':
    rospy.init_node('teleop_services_node')
    listener()

