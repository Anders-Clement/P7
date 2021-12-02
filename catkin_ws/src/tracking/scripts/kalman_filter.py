import numpy as np
import matplotlib.pyplot as plt
import copy
import pickle

class kalmanFilter:
    def __init__(self, startTime=0) -> None:
        # state vector
        self.x = np.array([0,0,0,0,0,0])

        # note this changes with dt, which is left out here
        self.F = np.array([[1,0,0,0,0,0],
                            [0,1,0,0,0,0],
                            [0,0,1,0,0,0],
                            [0,0,0,1,0,0],
                            [0,0,0,0,1,0],
                            [0,0,0,0,0,1]])
        
        # process covariance
        self.P = np.ones((6,6),np.float32)*0.1 + np.eye(6)*0.9
        # process noise
        self.V = np.ones((6,6),np.float32)*0.1 + np.eye(6)*0.9
        # measurement matrix
        self.H = np.array([ [1,0,0,0,0,0],
                            [0,1,0,0,0,0],
                            [0,0,1,0,0,0]])
        # measurement covariance
        self.W = np.eye(3)*0.9 + np.ones((3,3))*0.1
        self.W = self.W * 10 # between 2.5 and 25 is ok, 1, 50 are beyond working
        # measurement noise
        self.lastTime = startTime
        self.distanceTreshold = 1.5


    def predict(self, time, measurement):
        # update F with proper dt
        dt = time - self.lastTime
        self.F = np.array([[1,0,0,dt,0,0],
                            [0,1,0,0,dt,0],
                            [0,0,1,0,0,dt],
                            [0,0,0,1,0,0],
                            [0,0,0,0,1,0],
                            [0,0,0,0,0,1]])
        # predict
        self.x_pred = np.matmul(self.F, self.x)
        self.P_pred = np.matmul(self.F, np.matmul(self.P, self.F.T)) + self.V     
        
        # assume measurement is good, if within 2 meters of predicted kalman pos
        dist = np.abs(np.linalg.norm(self.x_pred[:2] - measurement[:2]))
        if dist < self.distanceTreshold: #1.3:
            self.update(measurement, time)
            return True
        
        return False

    def update(self, measurement, time):
        self.v = measurement - np.matmul(self.H, self.x_pred)
        self.S = np.matmul(self.H, np.matmul(self.P,self.H.T)) + self.W
        self.R = np.matmul(self.P, np.matmul(self.H.T, np.linalg.inv(self.S)))

        self.x = self.x_pred + np.matmul(self.R, self.v)
        self.P = self.P_pred - np.matmul(self.R, np.matmul(self.H, self.P_pred))
        self.lastTime = time


if __name__ == '__main__':
    OUTPUTFILENAME = 'person5.pickle'
    data = np.genfromtxt('nov-19-2021-14-57-person5.csv', delimiter=',')
    if not len(data) > 5:
        exit(-1) 
    start_value = 0
    data = data[start_value:]

    KF = kalmanFilter(data[0][0])
    KF.distanceTreshold = 50
    KF.x = np.array([73, -85,0,0,0,0]) 
   

    # person 5
    start_value = 175 
    data = data[start_value:] 
    KF = kalmanFilter(data[0][0]) 
    KF.x = np.array([86, -81, 0,0,0,0]) 
    KF.distanceTreshold = 1 


    # KF.x = np.array([-81, 104,0,0,0,0]) # person6
    # KF.distanceTreshold = 2 # person6
    # KF.x = np.array([85, -86,0,0,0,0]) # person4.csv
    # KF.x = np.array([48,-61,0,0,0,0]) # person6_del1.csv
    # KF.x = np.array([12, -18,0,0,0,0]) # person1.csv data = data[600:]
    # KF.x = np.array([-45, 118,0,0,0,0]) # data2.csv
    # KF.x = np.array([-44, -58,0,0,0,0]) # data3.csv
    # KF.x = np.array([-70, -114,0,0,0,0]) # data3.csv spot 2
    # KF.x = np.array([45, -50,0,0,0,0]) # data3.csv spot 2
    # KF.x = np.array([-43, -79,0,0,0,0]) # data4.csv
    # KF.x = np.array([-60, -94,0,0,0,0]) # data4.csv


    PLOT_CONTINOUSLY = False

    x_log = []
    y_log = []
    distance_to_spot_log = []
    spot_v_log = []
    time_log_filter = []
    time_log_spot = []
    time_log_angle = []
    colors = []

    fig, ax = plt.subplots(3)
    for i, measurement in enumerate(data):
        print(i, KF.x, measurement[0])

        if i > 0:
            dt = data[i,0] - data[i-1,0]
            if dt > 0:
                dpos = np.linalg.norm(data[i,4:7] - data[i-1,4:7])
                spot_v_log.append(dpos/dt)
                time_log_spot.append(data[i,0])
            #  Person 4 

            # if i == 900-start_value:
            #     KF.x = np.array([5.3, -107,0,0,0,0])
            #     KF.distanceTreshold = 2
            #     KF.P = np.ones((6,6),np.float32)*0.1 + np.eye(6)*0.9
            # if i == 3417-start_value:
            #     KF.x = np.array([-191, -198,0,0,0,0])
            #     KF.distanceTreshold = 1.5
            #     KF.P = np.ones((6,6),np.float32)*0.1 + np.eye(6)*0.9
            if i == 60: 

                KF.distanceTreshold = 3 

            if i == 150: 

              KF.distanceTreshold = 1 

            
            

            if i == 300: 

                KF.distanceTreshold = 3 

            
            

            if i == 400: 

                KF.distanceTreshold = 1.5 

            
            

            if i == 2560: 

                KF.P = np.ones((6,6),np.float32)*0.1 + np.eye(6)*0.9  

                KF.x = np.array([-183,-220,0,0,0,0]) 




            if i == 3400: 

                KF.x = np.array([-114, -160, 0,0,0,0]) 

                KF.P = np.ones((6,6),np.float32)*0.1 + np.eye(6)*0.9  

                KF.distanceTreshold = 1 

            if i < 100: 

                if measurement[2] < -82 or measurement[2] > -80: 

                    continue 
        y = np.array(measurement[1:4])
        y[2] = 0
        if KF.predict(measurement[0], y):
            colors.append(1)
            x_log.append(KF.x)
            y_log.append(y)
            distance = np.linalg.norm(measurement[1:3] - measurement[4:6])
            distance_to_spot_log.append(distance)
            time_log_filter.append(measurement[0])
            if measurement[7] < 0.0:
                measurement[7] += np.pi*2

            time_log_angle.append(measurement[7])
        # else:
        #     colors.append(0)
        


        if len(x_log) == 0:
            continue
        if PLOT_CONTINOUSLY and measurement[0] > 220:
            # plt.clf()
            # plt.scatter(np.array(y_log)[:,0], np.array(y_log)[:,1], c=colors)
            # plt.scatter(np.array(x_log)[:,0], np.array(x_log)[:,1], c=colors)
            # plt.scatter(KF.x[0], KF.x[1])
            # plt.scatter(KF.x_pred[0], KF.x_pred[1])
            #plt.scatter( KF.x[0] , KF.x[1] , s=KF.P[0,0] ,  facecolors='none', edgecolors='blue' ) 
            ax[0].clear()
            ax[1].clear()
            ax[2].clear()
            ax[0].scatter(data[i-20:i,1], data[i-20:i,2], label='detections')
            #plt.scatter(np.array(y_log)[:,0], np.array(y_log)[:,1], c=colors)
            #ax[0].scatter(np.array(x_log)[:,0], np.array(x_log)[:,1], c=colors, label='kalman filter')
            ax[0].scatter(KF.x_pred[0], KF.x_pred[1], label='predicted position')
            ax[0].set_xlim((-100, -80))
            ax[0].set_ylim((-145, -135))

            #ax[0].set_xlim(KF.x_pred[0] - 2, KF.x_pred[0] + 2)
            #ax[0].set_ylim(KF.x_pred[1] - 2, KF.x_pred[1] + 2)
            y_to_plot = []
            for i in range(len(x_log)):
                y_to_plot.append(np.linalg.norm(np.array(x_log[i])[3:5]))
            ax[1].plot(time_log_filter, y_to_plot, label='kalman filter')
            ax[1].plot(time_log_spot, spot_v_log, label='spot')
            ax[2].plot(time_log_filter, time_log_angle, label="angle")
            ax[0].set_xlabel('x-coordinate in meters')
            ax[0].set_ylabel('y-coordinate in meters')
            ax[0].legend()
            ax[1].set_xlabel('Time in seconds')
            ax[1].set_ylabel('Velocity in m/s')
            ax[1].legend()
            ax[2].legend()
            plt.draw()
            plt.pause(0.0001)


    # data_to_save = []
    # # extract data
    # for i in range(len(x_log)):
    #     timestamp = time_log_filter[i]
    #     kalman_velocity = np.linalg.norm(np.array(x_log[i])[3:5])
    #     spot_velocity = 0
    #     for j in range(len(time_log_spot)):
    #         if time_log_spot[j] == timestamp:
    #             spot_velocity = spot_v_log[j]
    #     data_to_save.append([timestamp, kalman_velocity, spot_velocity])

    # np.savetxt('outputData.csv', np.array(data_to_save), delimiter=',')
    data_to_save = [x_log, distance_to_spot_log, time_log_angle, time_log_filter, spot_v_log, time_log_spot]
    pickle.dump(data_to_save, open(OUTPUTFILENAME,'wb'))

    ax[0].clear()
    ax[1].clear()
    ax[0].scatter(data[:i,1], data[:i,2], label='detections')
    #plt.scatter(np.array(y_log)[:,0], np.array(y_log)[:,1], c=colors)
    ax[0].scatter(np.array(x_log)[:,0], np.array(x_log)[:,1], c=colors, label='kalman filter')
    ax[0].scatter(KF.x_pred[0], KF.x_pred[1], label='predicted position')
    y_to_plot = []
    for i in range(len(x_log)):
        y_to_plot.append(np.linalg.norm(np.array(x_log[i])[3:5]))
    ax[1].plot(time_log_filter, y_to_plot, label='kalman filter')
    ax[1].plot(time_log_spot, spot_v_log, label='spot')
    ax[0].set_xlabel('x-coordinate in meters')
    ax[0].set_ylabel('y-coordinate in meters')
    ax[0].legend()
    ax[1].set_xlabel('Time in seconds')
    ax[1].set_ylabel('Velocity in m/s')
    ax[1].legend()
    ax[2].plot(time_log_filter, distance_to_spot_log, label="distance")
    ax[2].plot(time_log_filter, time_log_angle, label="angle")
    ax[2].set_xlabel('Time in seconds')
    ax[2].set_ylabel('Distance (m), Angle(rad)')
    ax[2].legend()
    ax[2].sharex(ax[1])


    plt.show()
        #plt.waitforbuttonpress()


    