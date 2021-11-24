import matplotlib.pyplot as plt
import numpy as np
import pickle




if __name__ == '__main__':
    personNames = ['person1','person2','person3', 'person4','person5','person6']
    colors = ['red', 'green', 'orange', 'cyan', 'magenta', 'black']
    picklesToLoad = ['person1.pickle', 'person2.pickle','person3.pickle', 'person4.pickle', 'person5.pickle', 'person6.pickle']
    delay = 10
    testTimeStep = 330 + delay
    testTimeStaircase = 80 + delay
    testTimeRamp = 45 + delay
    testTimes = [testTimeStep, testTimeStaircase, testTimeRamp]
    # startTimes = [[26.8, 383.5, 483.3], #person1
    #              [2.6, 352, 446], # person2
    #              [10, 377, 482], # person3
    #              [35.1, 436, 542], #person4
    #              [9.8, 387, 500], # person5
    #              [12.1, 365, 458], #person6
    #              ] 

    startTimes = [[26.8, 382, 483.3], #person1
                [3.6, 352, 446.75], # person2
                [9.5, 378, 481.25], # person3
                [35.1, 436, 543], #person4
                [10.8, 387, 500], # person5
                [12.1, 364, 458.25], #person6
                ] 


    dataSets = []
    for pickleToLoad in picklesToLoad:
        data = pickle.load(open(pickleToLoad, 'rb'))
        dataSets.append(data)

    for i, dataSet in enumerate(dataSets):
        x_log, distance_to_spot_log, time_log_angle, time_log_filter, spot_v_log, time_log_spot = dataSet
        startStopTimeIndeces = []
        startIndex = None
        stopIndex = None
        testNum = 0
        for j, time in enumerate(time_log_filter):
            if time > startTimes[i][testNum] and startIndex is None:
                startIndex = j
            if time > startTimes[i][testNum] + testTimes[testNum] and stopIndex is None:
                stopIndex = j

            if startIndex is not None and stopIndex is not None:
                startStopTimeIndeces.append((startIndex, stopIndex))
                startIndex = None
                stopIndex = None
                testNum += 1
                if testNum == 3:
                    break

        # same, but for spot, due to different lengths of lists
        startStopTimeIndecesSpot = []
        startIndex = None
        stopIndex = None
        testNum = 0
        for j, time in enumerate(time_log_spot):
            if time > startTimes[i][testNum] and startIndex is None:
                startIndex = j
            if time > startTimes[i][testNum] + testTimes[testNum] and stopIndex is None:
                stopIndex = j

            if startIndex is not None and stopIndex is not None:
                startStopTimeIndecesSpot.append((startIndex, stopIndex))
                startIndex = None
                stopIndex = None
                testNum += 1
                if testNum == 3:
                    break


        # lastStopTime = 0
        # lastStartTime = None
        # lastStopIndex = 0
        # startIndex = 0
        # offset = 0
        # for j, startStopIndex in enumerate(startStopTimeIndeces):
        #     offset = 0
        #     if j == 0 and startStopIndex[0] == 0:
        #         offset = time_log_filter[startStopIndex[0]] - time_log_spot[startStopTimeIndecesSpot[j][0]]

        #     if j == 1 and i == 2:
        #         offset = time_log_filter[startStopIndex[0]] - time_log_spot[startStopTimeIndecesSpot[j][0]]
        #         offset = 6
        #     sumToSubtract = time_log_filter[startStopIndex[0]] - lastStopTime - offset
        #     for l in range(lastStopIndex, len(time_log_filter)):
        #         time_log_filter[l] -= sumToSubtract

        #     lastStopTime = time_log_filter[startStopIndex[1]]
        #     lastStopIndex = startStopIndex[1]
        #     startIndex = startStopIndex[1] - startStopIndex[0]


        lastStopTime = 0
        lastStartTime = None
        lastStopIndex = 0
        lastStopIndexSpot = 0
        startIndex = 0
        for j, startStopIndexSpot in enumerate(startStopTimeIndecesSpot):
            offset = 0
            if j == 0 and startStopIndexSpot[0] == 0:
                offset = time_log_filter[startStopIndexSpot[0]] - time_log_spot[startStopTimeIndecesSpot[j][0]]

            sumToSubtract = time_log_spot[startStopIndexSpot[0]] - lastStopTime
            for l in range(lastStopIndexSpot, len(time_log_spot)):
                time_log_spot[l] -= sumToSubtract
            for l in range(lastStopIndex, len(time_log_filter)):
                time_log_filter[l] -= sumToSubtract - offset
            lastStopTime = time_log_spot[startStopIndexSpot[1]]
            lastStopIndex = startStopTimeIndeces[j][1]
            lastStopIndexSpot = startStopIndexSpot[1]
            startIndex = startStopIndexSpot[1] - startStopIndexSpot[0]


        # for l, startStopIndex in enumerate(startStopTimeIndeces):
        #     for j in range(len(time_log_filter)):
        #         if j < startStopIndex[0]:
        #             time_log_filter[j] -= time_log_filter[startStopIndex[0]] + l*testTimes[l]
        #print(startStopTimeIndeces, startStopTimeIndecesSpot)
        new_x_log = []
        new_distance_to_spot_log = []
        new_time_log_angle = []
        new_time_log_filter = []
        for j, startStop in enumerate(startStopTimeIndeces):
            new_x_log += x_log[startStop[0]:startStop[1]]
            new_distance_to_spot_log += distance_to_spot_log[startStop[0]:startStop[1]]
            new_time_log_angle += time_log_angle[startStop[0]:startStop[1]]
            new_time_log_filter += time_log_filter[startStop[0]:startStop[1]]

        new_spot_v_log = []
        new_time_log_spot = []
        for j, startStop in enumerate(startStopTimeIndecesSpot):
            new_spot_v_log += spot_v_log[startStop[0]:startStop[1]]
            new_time_log_spot += time_log_spot[startStop[0]:startStop[1]]

        dataSets[i] = [new_x_log, new_distance_to_spot_log, new_time_log_angle, new_time_log_filter, new_spot_v_log, new_time_log_spot]


    fig, ax = plt.subplots(3)
    for j, dataSet in enumerate(dataSets):
        x_log, distance_to_spot_log, time_log_angle, time_log_filter, spot_v_log, time_log_spot = dataSet

        for i in range(len(time_log_angle)):
            time_log_angle[i] = (time_log_angle[i]-np.pi) * (180/np.pi)
            # old x-y plot
            #ax[0].scatter(data[:i,1], data[:i,2], label='detections')
            #plt.scatter(np.array(y_log)[:,0], np.array(y_log)[:,1], c=colors)
            #ax[0].scatter(np.array(x_log)[:,0], np.array(x_log)[:,1], label='kalman filter')
            #ax[0].scatter(KF.x_pred[0], KF.x_pred[1], label='predicted position')
        y_to_plot = []
        for i in range(len(x_log)):
            y_to_plot.append(np.linalg.norm(np.array(x_log[i])[3:5]))

        if j == 2:
            ax[0].plot(time_log_spot, spot_v_log, label='spot', color='blue')
        ax[0].plot(time_log_filter, y_to_plot, label=personNames[j], color=colors[j])
        ax[0].set_xlabel('Time in seconds')
        ax[0].set_ylabel('Speed in m/s')
        ax[0].legend()
        ax[1].plot(time_log_filter, distance_to_spot_log, label="dist." + personNames[j], color=colors[j])
        ax[1].set_xlabel('Time in seconds')
        ax[1].set_ylabel('Distance (m)')
        ax[2].set_xlabel('Time in seconds')
        ax[2].plot(time_log_filter, time_log_angle, label="angle " + personNames[j], color=colors[j])
        ax[2].set_ylabel('Angle (degrees)')
        ax[1].legend()
        ax[2].legend()
        ax[1].sharex(ax[0])
        ax[2].sharex(ax[0])

    plt.show()


    


    

    