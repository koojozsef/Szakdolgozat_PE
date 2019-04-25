from __future__ import print_function
import sys
import cv2
from random import randint
import numpy as np


trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
trackerType = trackerTypes[4]

def createTrackerByName(trackerType):
    # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)
    return tracker


def main():
    #createTrackerByName(trackerTypes[0])

    videoPath = "D:/joci/projects/Szakdoga_PE/Szakdoga/Dataset/Video/traffic.mp4"

    # Create a video capture object to read videos
    cap = cv2.VideoCapture(videoPath)


    # Read first frame
    success, frame = cap.read()
    # quit if unable to read the video file
    if not success:
        print('Failed to read video')
        sys.exit(1)

    frame = cv2.resize(frame, (1280, 720))

    # Select boxes
    bboxes = []
    colors = []

    # OpenCV's selectROI function doesn't work for selecting multiple objects in Python
    # So we will call this function in a loop till we are done selecting all objects
    while True:
        # draw bounding boxes over objects
        # selectROI's default behaviour is to draw box starting from the center
        # when fromCenter is set to false, you can draw box starting from top left corner
        bbox = cv2.selectROI('MultiTracker', frame)
        bboxes.append(bbox)
        colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
        print("Press q to quit selecting boxes and start tracking")
        print("Press any other key to select next object")
        k = cv2.waitKey(0) & 0xFF
        if (k == 113):  # q is pressed
            break

    print('Selected bounding boxes {}'.format(bboxes))

    # Create MultiTracker object
    multiTracker = cv2.MultiTracker_create()

    # Initialize MultiTracker
    for bbox in bboxes:
        multiTracker.add(createTrackerByName(trackerType), frame, bbox)

    kalman_x = cv2.KalmanFilter(2, 1)

    kalman_x.transitionMatrix = np.array([[1., 1.], [0., 1.]])
    kalman_x.measurementMatrix = .5 * np.ones((1, 2))
    kalman_x.processNoiseCov = 1e-5 * np.eye(2)
    kalman_x.measurementNoiseCov = 1e-1 * np.ones((1, 1))
    kalman_x.errorCovPost = 1. * np.ones((2, 2))

    state_x = 1. * np.ones((2, 1))

    kalman_y = cv2.KalmanFilter(2, 1)

    kalman_y.transitionMatrix = np.array([[1., 1.], [0., 1.]])
    kalman_y.measurementMatrix = .5 * np.ones((1, 2))
    kalman_y.processNoiseCov = 1e-5 * np.eye(2)
    kalman_y.measurementNoiseCov = 1e-1 * np.ones((1, 1))
    kalman_y.errorCovPost = 1. * np.ones((2, 2))

    state_y = 1. * np.ones((2, 1))

    # Process video and track objects
    KEY_PRESSED = 1
    j = 0
    while cap.isOpened():
        if KEY_PRESSED == 0:
            k = cv2.waitKey() & 0xff
            KEY_PRESSED = 1
            print(f"key pressed: {k}")
        else:
            KEY_PRESSED = 0
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.resize(frame, (1280, 720))

            # get updated location of objects in subsequent frames
            success, boxes = multiTracker.update(frame)

            # draw tracked objects

            for i, newbox in enumerate(boxes):
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                state_x[0] = p1[0] + ((p2[0] - p1[0]) / 2)
                state_y[0] = p1[1] + ((p2[1] - p1[1]) / 2)

                if j == 0:
                    kalman_y.statePost = p2[1]/2 * np.ones((2, 1))
                    kalman_x.statePost = p2[0]/2 * np.ones((2, 1))
                    print(kalman_x.statePost)
                    j = 1

                cv2.circle(frame, (state_x[0], state_y[0]), 1, colors[i], 1, 1)

                pred_x = kalman_x.predict()
                kalman_x.correct(np.dot(kalman_x.measurementMatrix, state_x))

                pred_y = kalman_y.predict()
                kalman_y.correct(np.dot(kalman_y.measurementMatrix, state_y))

                print(pred_x, ", ", pred_y)

                cv2.circle(frame, (pred_x[0], pred_y[0]), 5, colors[i], 2, 1)
                cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

            # show frame
            cv2.imshow('MultiTracker', frame)

            # quit on ESC button
            if k == 27:  # Esc pressed
                break

if __name__ == "__main__":
    main()
