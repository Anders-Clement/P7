import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

def loadGradientData():
    x,y,gradients = pickle.load(open('catkin_ws/src/tracking/scripts/gradients.pickle','rb'))
    return x,y,gradients 


def getGradientFromPos(vel, dist, x_steps, y_steps, gradients):
    x_max = np.max(x_steps)
    y_max = np.max(y_steps)
    x_index = int((vel / x_max)*len(x_steps))
    y_index = int((dist / y_max)*len(y_steps))
    if len(x_steps) > x_index >= 0 and  len(y_steps) > y_index >= 0:
        gradient = gradients[x_index, y_index]
        gradient = -gradient # gradient descent
        return np.array([gradient[1],gradient[0]])
    else:
        return False
        

def getIdealDistance(velocity):
    #di = 0.593*velocity**2 - 0.405*velocity + 1.78
    a = 0.73
    b = 1.32
    di = a*velocity + b
    return di


def sim(Ts, simTime, x, Kpd = 1, Kpv = 1, Kp_vel = 0.5, Kp_dist = 0.5,
         spotDeltaXdot='gradient', useSpotForIdealDist=False, includeStop=False):
    x_steps, y_steps, gradients = loadGradientData()
    time = 0
    x_log = []
    time_log = []
    ideal_dist = []
    ideal_dist.append(x[0]) # x is initialized at ideal distance
    target_spot_vel = 1.4 # max velocity
    spot_acc_lim = 1 # as read from odometry data

    if includeStop and spotDeltaXdot == 'proportional':
        targetSpotVel = np.append(np.ones(shape=(int((simTime/2)/Ts)))*target_spot_vel, np.zeros(shape=(int((simTime/2)/Ts))))
    else:
        targetSpotVel = np.ones(int(simTime/Ts))*target_spot_vel

    for i in range(int(simTime/Ts)):
        time_log.append(time)
        x_log.append(x)
        time = time + Ts
        A = np.array([[1, Ts, 0,0],
                     [0,1,0,0],
                     [0,0,1,Ts],
                     [0,0,0,1]])
        if spotDeltaXdot == 'simple':
            deltaXdot = spot_acc_lim*Ts
        elif spotDeltaXdot == 'proportional':
            deltaXdot = (targetSpotVel[i] - x[1])*2*Ts
        elif spotDeltaXdot == 'gradient':
            dist = x[0] - x[2]
            vel = x[3]
            gradient = getGradientFromPos(vel, dist, x_steps, y_steps, gradients)
            if gradient is False:
                print('failed to get gradient, vel,dist: ', vel, dist)
                exit(-1)
            deltaXdot = gradient[0]*Kp_vel*Ts + gradient[1]*Kp_dist*Ts

        if abs(deltaXdot) > spot_acc_lim*Ts:
            if deltaXdot > 0:
                deltaXdot = spot_acc_lim*Ts
            else:
                deltaXdot = -spot_acc_lim*Ts
        # limit Spot's velocity to target_spot_vel
        if x[1] + deltaXdot > target_spot_vel: 
            deltaXdot = target_spot_vel - x[1]

        d = float(x[0] - x[2])
        if useSpotForIdealDist:
            v = x[1]
        else:
            v = x[3]
        di = getIdealDistance(v) # replace with either 2nd order or 1st order polynomial
        ideal_dist.append(di)
        Khuman = (d-di)*Kpd*Ts + (x[1]-x[3])*Kpv*Ts
        B = np.array([0,deltaXdot,0, Khuman], dtype=float)
        B = np.reshape(B, (4,1))
        new_x = np.matmul(A,x) + B
        x = new_x
    x_log.append(x)
    time_log.append(time)
    return np.array(x_log), np.array(time_log), np.array(ideal_dist)


def plot_xlog_time_log(x_log, ideal_dist, time_log, show=True, Title='Empty'):
    fig, ax = plt.subplots(3)
    fig.suptitle(Title)
    distance = x_log[:,0] - x_log[:,2]
    ax[0].plot(time_log, x_log[:,0], label='spot_pos')
    ax[2].plot(time_log, x_log[:,1], label='spot_vel')
    ax[0].plot(time_log, x_log[:,2], label='person_pos')
    ax[2].plot(time_log, x_log[:,3], label='person_vel')
    ax[1].plot(time_log, distance, label='distance spot<=>person')
    ax[1].plot(time_log, ideal_dist, label='ideal distance')
    ax[1].set_ylim((0,3.5))
    ax[2].set_ylim((0,2))
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

    Ts = 0.001
    start = time.time()
    #for i in range(1,4):
    Kp = .5
    Kp_dist = (2/5)
    Kp_vel = (1 - Kp_dist) * Kp
    Kp_dist *= Kp
    print(Kp_vel, Kp_dist)
    x_log, time_log, ideal_dist = sim(Ts,20,x, Kpd=.5, Kpv=.5, Kp_vel=Kp_vel, Kp_dist=Kp_dist)
    plot_xlog_time_log(x_log, ideal_dist, time_log, show=False, Title='Kp_vel: ' + str(Kp_vel)+ ', Kp_dist: ' + str(Kp_dist))

    end = time.time()
    print('calculation time: ', end-start)
    x_log, time_log, ideal_dist = sim(Ts,20,x, Kpd=.5, Kpv=.5, spotDeltaXdot='proportional')
    #plot_xlog_time_log(x_log, ideal_dist, time_log, show=False, Title='Kp_vel: ' + str(Kp_vel)+ ', Kp_dist: ' + str(Kp_dist))

    plt.show()


def test():
    x_steps, y_steps, gradients = loadGradientData()
    gradient = getGradientFromPos(0,0, x_steps, y_steps, gradients)
    print(gradient)
    gradient = getGradientFromPos(0.5, 0.5, x_steps, y_steps, gradients)
    print(gradient)
    gradient = getGradientFromPos(1.4, 1.4, x_steps, y_steps, gradients)
    print(gradient)
    gradient = getGradientFromPos(0, np.max(x_steps), x_steps, y_steps, gradients)
    print(gradient)
    gradient = getGradientFromPos(np.max(y_steps), np.max(x_steps), x_steps, y_steps, gradients)
    print(gradient)
    gradient = getGradientFromPos(np.max(y_steps), 0, x_steps, y_steps, gradients)
    print(gradient)


if __name__ == '__main__':
   main()
   #test()