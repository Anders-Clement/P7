import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from math import sqrt

from numpy.lib.function_base import gradient

def loadGradientData():
    x,y,gradients = pickle.load(open('gradients_large.pickle','rb'))
    return x,y,gradients 


def getGradientFromFunc(vel, dist):
    x = vel
    y = dist
    # gradient = [2.02957850094677*(0.158367116365784*x + 0.115611616503873*y - 0.492713141932434)/sqrt((-0.32141849463293*x - 0.234642851315963*y + 1)**2 
    #             + 0.532933387965517*(-0.32141849463293*x - 0.234642851315963*y + 1)**2) + 0.862409286678319*(0.467436688907025*x - 0.640304173095752*y 
    #             + 0.846492355066264)/sqrt((-0.552204265176659*x + 0.756420503107354*y - 1)**2 + 0.532933387965517*(0.552204265176658*x - 0.756420503107354*y 
    #             + 1)**2), 
    #             2.02957850094677*(0.115611616503873*x + 0.0843991238671465*y - 0.359691861029668)/sqrt((-0.32141849463293*x - 0.234642851315963*y + 1)**2 
    #             + 0.532933387965517*(-0.32141849463293*x - 0.234642851315963*y + 1)**2) + 0.862409286678319*(-0.640304173095752*x + 0.877101528000476*y 
    #             - 1.15954224455494)/sqrt((-0.552204265176659*x + 0.756420503107354*y - 1)**2 + 0.532933387965517*(0.552204265176658*x - 0.756420503107354*y 
    #             + 1)**2)]

    #0.1 0.3
   # gradient = [ 0.0695311866972662*x - 0.0952452172863454*y - 0.174084299810647 , -0.0952452172863454*x + 0.130468813302734*y - 0.172481857335664 ]
   # 0.1 0.1
    gradient = [ 0.0695311866972662*x - 0.0952452172863454*y + 0.0259157001893534 , -0.0952452172863454*x + 0.130468813302734*y - 0.172481857335664 ]

    # quadratic vel cost, *0.1
    gradient = [ 0.269531186697266*x - 0.0952452172863454*y - 0.154084299810647 , -0.0952452172863454*x + 0.130468813302734*y - 0.172481857335664 ]
    return -np.array(gradient) # minus to get gradient descent


def getGradientFromPos(vel, dist, x_steps, y_steps, gradients):
    x_max = np.max(x_steps)
    y_max = np.max(y_steps)
    x_index = int((vel / x_max)*len(x_steps))
    y_index = int((dist / y_max)*len(y_steps))
    if len(x_steps) > x_index >= 0 and  len(y_steps) > y_index >= 0:
        gradient = gradients[x_index, y_index]
        gradient = -gradient # gradient descent
        return np.array([gradient[1], -gradient[0]])
    else:
        return False
        

def getIdealDistance(velocity):
    #di = 0.593*velocity**2 - 0.405*velocity + 1.78
    a = 0.73
    b = 1.32
    di = a*velocity + b
    return di


def sim_human_follow_spot(Ts, simTime, x, Kpd_person=.5, Kpv_person=.5, Kp_vel_spot = 0.5, Kp_dist_spot = 0.5,
         spotDeltaXdot='gradient', useSpotForIdealDist=False, includeStop=False):
    x_steps, y_steps, gradients = loadGradientData()
    time = 0
    x_log = []
    time_log = []
    ideal_dist = []
    gradient_log = []
    ideal_dist.append(x[0]) # x is initialized at ideal distance
    max_spot_vel = 1.4 # max velocity
    spot_acc_lim = 1 # as read from odometry data

    if includeStop and spotDeltaXdot == 'proportional':
        targetSpotVel = np.append(np.ones(shape=(int((simTime/2)/Ts)))*max_spot_vel, np.zeros(shape=(int((simTime/2)/Ts))))
    else:
        targetSpotVel = np.ones(int(simTime/Ts))*max_spot_vel

    for i in range(int(simTime/Ts)):
        time_log.append(time)
        x_log.append(x)
        time = time + Ts
        A = np.array([[1, Ts, 0,0],
                     [0,1,0,0],
                     [0,0,1,Ts],
                     [0,0,0,1]])
        if spotDeltaXdot == 'simple':
            if includeStop and time > simTime / 2: 
                deltaXdot = -spot_acc_lim*Ts
            else:
                deltaXdot = spot_acc_lim*Ts

        elif spotDeltaXdot == 'proportional':
            deltaXdot = (targetSpotVel[i] - x[1])*2*Ts
        elif spotDeltaXdot == 'gradient':
            dist = x[0] - x[2]
            vel = x[3]
            #gradient = getGradientFromPos(vel, dist, x_steps, y_steps, gradients)
            gradient = getGradientFromFunc(vel, dist)
            if gradient is False:
                print('failed to get gradient, vel,dist: ', vel, dist)
                return np.array(x_log), np.array(time_log), np.array(ideal_dist), np.array(gradient_log)
                exit(-1)
            deltaXdot = gradient[0]*Kp_vel_spot*Ts + gradient[1]*Kp_dist_spot*Ts
            gradient_log.append(np.append(gradient, deltaXdot/Ts))

        if abs(deltaXdot) > spot_acc_lim*Ts:
            if deltaXdot > 0:
                deltaXdot = spot_acc_lim*Ts
            else:
                deltaXdot = -spot_acc_lim*Ts
        # limit Spot's velocity to max_spot_vel which is Spot's max velocity
        if x[1] + deltaXdot > max_spot_vel: 
            deltaXdot = max_spot_vel - x[1]
        if x[1] + deltaXdot < 0:
            deltaXdot = -x[1]

        d = float(x[0] - x[2])
        if useSpotForIdealDist:
            v = x[1]
        else:
            v = x[3]
        di = getIdealDistance(v) 
        ideal_dist.append(di)
        Khuman = (d-di)*Kpd_person*Ts + (x[1]-x[3])*Kpv_person*Ts
        B = np.array([0,deltaXdot,0, Khuman], dtype=float)
        B = np.reshape(B, (4,1))
        new_x = np.matmul(A,x) + B
        x = new_x
    x_log.append(x)
    time_log.append(time)
    return (np.array(x_log), np.array(time_log), np.array(ideal_dist), np.array(gradient_log))


def sim_spot_with_gradient_human_stops(Ts, simTime, x, human_stop_time, instant_stop=False, Kpd_person=.5, Kpv_person=.5, Kp_vel_spot = 0.5, Kp_dist_spot = 0.5):
    x_steps, y_steps, gradients = loadGradientData()
    time = 0
    x_log = []
    time_log = []
    ideal_dist = []
    gradient_log = []
    ideal_dist.append(x[0]) # x is initialized at ideal distance
    max_spot_vel = 1.4 # max velocity
    spot_acc_lim = 1 # as read from odometry data
    human_acc_lim = .5
    human_second_start_time = human_stop_time + 10

    for i in range(int(simTime/Ts)):
        time_log.append(time)
        x_log.append(x)
        time = time + Ts
        A = np.array([[1, Ts, 0,0],
                     [0,1,0,0],
                     [0,0,1,Ts],
                     [0,0,0,1]])

        dist = x[0] - x[2]
        vel = x[3]
        #gradient = getGradientFromPos(vel, dist, x_steps, y_steps, gradients)
        gradient = getGradientFromFunc(vel, dist)
        if gradient is False:
            print('failed to get gradient, vel,dist: ', vel, dist)
            return np.array(x_log), np.array(time_log), np.array(ideal_dist), np.array(gradient_log)
            exit(-1)
        deltaXdot = gradient[0]*Kp_vel_spot*Ts + gradient[1]*Kp_dist_spot*Ts
        gradient_log.append(np.append(gradient, deltaXdot/Ts))

        if abs(deltaXdot) > spot_acc_lim*Ts:
            if deltaXdot > 0:
                deltaXdot = spot_acc_lim*Ts
            else:
                deltaXdot = -spot_acc_lim*Ts
        # limit Spot's velocity to max_spot_vel which is Spot's max velocity
        if x[1] + deltaXdot > max_spot_vel: 
            deltaXdot = max_spot_vel - x[1]
        if x[1] + deltaXdot < 0:
            deltaXdot = -x[1]

        # finally, ensure human is below 1.4:
        if x[3] > 1.4:
            deltaXdot = 0

        d = float(x[0] - x[2])
        v = x[3]
        di = getIdealDistance(v) 
        ideal_dist.append(di)
        Khuman = (d-di)*Kpd_person*Ts + (x[1]-x[3])*Kpv_person*Ts
        if human_second_start_time > time > human_stop_time:
            Khuman = 0
        B = np.array([0,deltaXdot,0, Khuman], dtype=float)
        B = np.reshape(B, (4,1))
        new_x = np.matmul(A,x) + B

        if human_second_start_time > time > human_stop_time:
            if instant_stop:
                new_x[3] = 0
            else:
                new_x[3] += -human_acc_lim*Ts
                if new_x[3] < 0:
                    new_x[3] = 0

        if time > human_second_start_time:
            if new_x[3] > 1:
                new_x[3] = 1

        x = new_x
    x_log.append(x)
    time_log.append(time)
    return (np.array(x_log), np.array(time_log), np.array(ideal_dist), np.array(gradient_log))


def simSimpleProportionalControl(Ts, simTime, x, human_stop_time, Kp, instant_stop=False, Kpd_person=.5, Kpv_person=.5, Kp_vel_spot = 0.5, Kp_dist_spot = 0.5):
    time = 0
    x_log = []
    time_log = []
    ideal_dist = []
    gradient_log = []
    ideal_dist.append(x[0]) # x is initialized at ideal distance
    max_spot_vel = 1.4 # max velocity
    spot_acc_lim = 1 # as read from odometry data
    human_acc_lim = .5
    human_second_start_time = human_stop_time + 10

    for i in range(int(simTime/Ts)):
        time_log.append(time)
        x_log.append(x)
        time = time + Ts
        A = np.array([[1, Ts, 0,0],
                     [0,1,0,0],
                     [0,0,1,Ts],
                     [0,0,0,1]])

        dist = x[0] - x[2]
        vel = x[3]
        #gradient = getGradientFromPos(vel, dist, x_steps, y_steps, gradients)
        gradient = getGradientFromFunc(vel, dist)
        if gradient is False:
            print('failed to get gradient, vel,dist: ', vel, dist)
            return np.array(x_log), np.array(time_log), np.array(ideal_dist), np.array(gradient_log)
            exit(-1)
        gradient = np.array([0,0])
        deltaXdot = (-(dist-getIdealDistance(vel)) + (x[3]-x[1])*0.5 + Kp)*Ts
        gradient_log.append(np.append(gradient, deltaXdot))

        if abs(deltaXdot) > spot_acc_lim*Ts:
            if deltaXdot > 0:
                deltaXdot = spot_acc_lim*Ts
            else:
                deltaXdot = -spot_acc_lim*Ts
        # limit Spot's velocity to max_spot_vel which is Spot's max velocity
        if x[1] + deltaXdot > max_spot_vel: 
            deltaXdot = max_spot_vel - x[1]
        if x[1] + deltaXdot < 0:
            deltaXdot = -x[1]

        # finally, ensure human is below 1.4:
        if x[3] > 1.4:
            deltaXdot = 0

        d = float(x[0] - x[2])
        v = x[3]
        di = getIdealDistance(v) 
        ideal_dist.append(di)
        Khuman = (d-di)*Kpd_person*Ts + (x[1]-x[3])*Kpv_person*Ts
        if human_second_start_time > time > human_stop_time:
            Khuman = 0
        B = np.array([0,deltaXdot,0, Khuman], dtype=float)
        B = np.reshape(B, (4,1))
        new_x = np.matmul(A,x) + B

        if human_second_start_time > time > human_stop_time:
            if instant_stop:
                new_x[3] = 0
            else:
                new_x[3] += -human_acc_lim*Ts
                if new_x[3] < 0:
                    new_x[3] = 0

        if time > human_second_start_time:
            if new_x[3] > 1:
                new_x[3] = 1

        x = new_x
    x_log.append(x)
    time_log.append(time)
    return (np.array(x_log), np.array(time_log), np.array(ideal_dist), np.array(gradient_log))



def plot_xlog_time_log(data, show=True, Title='Empty'):
    x_log, time_log, ideal_dist, gradient_log = data
    if len(gradient_log) > 1:
        fig, ax = plt.subplots(4)
        ax[3].plot(time_log[:len(gradient_log)], gradient_log[:,0], label='gradient vel')
        ax[3].plot(time_log[:len(gradient_log)], gradient_log[:,1], label='gradient dist')
        ax[3].plot(time_log[:len(gradient_log)], gradient_log[:,2], label='deltaXdot')
        ax[3].set_xlabel('time [s]')
        ax[3].set_ylabel('magnitude')
        ax[3].grid()
        ax[3].legend()    
        ax[3].sharex(ax[2])



    else:
        fig, ax = plt.subplots(3)
    fig.suptitle(Title)
    distance = x_log[:,0] - x_log[:,2]
    ax[0].plot(time_log, x_log[:,0], label='spot_pos')
    ax[2].plot(time_log, x_log[:,1], label='spot_vel')
    ax[0].plot(time_log, x_log[:,2], label='person_pos')
    ax[2].plot(time_log, x_log[:,3], label='person_vel')
    ax[1].plot(time_log, distance, label='distance spot<=>person')
    ax[1].plot(time_log, ideal_dist, label='ideal distance')

    #ax[1].set_ylim((0,3.5))
    #ax[2].set_ylim((0,2))
    ax[1].set_xlabel('time [s]')
    ax[1].set_ylabel('distance [m]')
    ax[0].set_xlabel('time [s]')
    ax[2].set_xlabel('time [s]')
    ax[0].set_ylabel('position [m]')
    ax[2].set_ylabel('velocity [m/s]')

    ax[0].grid()
    ax[2].grid()
    ax[1].grid()
    ax[1].sharex(ax[0])
    ax[2].sharex(ax[1])
    ax[0].legend()
    ax[2].legend()
    ax[1].legend()
    if show:
        plt.show()


def main():
    idealDist = getIdealDistance(0)
    x = np.array([idealDist,0,0,0], dtype=np.float32) # start at ideal distance
    x = np.reshape(x,(4,1))

    Ts = .01
    start = time.time()
    #for i in range(1,4):
    Kp = .5
    Kp_dist = (2/5)
    Kp_vel = (1 - Kp_dist) * Kp
    Kp_dist *= Kp
    #print(Kp_vel, Kp_dist)
    #data = sim_human_follow_spot(Ts,30,x, Kpd_person=.5, Kpv_person=.5, Kp_vel_spot=Kp_vel, Kp_dist_spot=Kp_dist, spotDeltaXdot='proportional', includeStop=True)
    #plot_xlog_time_log(data, show=False, Title='Person following Spot')

    end = time.time()
    print('calculation time: ', end-start)
    Kp_vel = 1
    Kp_dist = 3
    data = sim_human_follow_spot(Ts,30,x, spotDeltaXdot='gradient', Kp_dist_spot = Kp_dist, Kp_vel_spot=Kp_vel)
    plot_xlog_time_log(data, show=False, Title='simple human')

    data = sim_spot_with_gradient_human_stops(Ts, 60, x, 30, Kp_dist_spot=Kp_dist, Kp_vel_spot=Kp_vel, instant_stop=False)
    plot_xlog_time_log(data, show=False, Title='advanced human')

    # data = simSimpleProportionalControl(Ts, 50, x, 15, .1, Kp_dist_spot=30, Kp_vel_spot=30, instant_stop=False)
    # plot_xlog_time_log(data, show=False, Title='proportional control .1')
    # data = simSimpleProportionalControl(Ts, 50, x, 15, .3, Kp_dist_spot=30, Kp_vel_spot=30, instant_stop=False)
    # plot_xlog_time_log(data, show=False, Title='proportional control .3')
    #data = simSimpleProportionalControl(Ts, 60, x, 25, 2, Kp_dist_spot=30, Kp_vel_spot=30, instant_stop=False)
    #plot_xlog_time_log(data, show=False, Title='proportional control 2')
    plt.show()


def test():
    x_steps, y_steps, gradients = loadGradientData()
    gradient = getGradientFromPos(0,0, x_steps, y_steps, gradients)
    gradient2 = getGradientFromFunc(0,0)
    print(gradient, gradient2)
    gradient = getGradientFromPos(0.5, 0.5, x_steps, y_steps, gradients)
    gradient2 = getGradientFromFunc(0.5,0.5)
    print(gradient, gradient2)
    gradient = getGradientFromPos(1.4, 1.4, x_steps, y_steps, gradients)
    gradient2 = getGradientFromFunc(1.4,1.4)
    print(gradient, gradient2)
    gradient = getGradientFromPos(0, np.max(x_steps), x_steps, y_steps, gradients)
    gradient2 = getGradientFromFunc(0, np.max(x_steps))
    print(gradient, gradient2)
    gradient = getGradientFromPos(np.max(y_steps), np.max(x_steps), x_steps, y_steps, gradients)
    gradient2 = getGradientFromFunc(np.max(y_steps),np.max(x_steps))
    print(gradient, gradient2)
    gradient = getGradientFromPos(np.max(y_steps), 0, x_steps, y_steps, gradients)
    gradient2 = getGradientFromFunc(np.max(y_steps),0)
    print(gradient, gradient2)


if __name__ == '__main__':
   main()
   #test()