from __future__ import print_function
import sys
import cv2
from random import randint
import numpy as np


trackerTypes = ['KCF', 'TLD', 'MEDIANFLOW', 'MOSSE']
trackerType = trackerTypes[0]

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

    videoPath = "D:\joci\projects\Szakdoga_PE\Szakdoga\Dataset\Video\\traffic.mp4".replace("\\", "/")

    # Create a video capture object to read videos

    cap = cv2.VideoCapture(videoPath)


    # Read first frame
    for i in range(160):
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
        # draw bounding boxes over objects
        # selectROI's default behaviour is to draw box starting from the center
        # when fromCenter is set to false, you can draw box starting from top left corner
    bbox = (553, 258, 33, 22)
    bboxes.append(bbox)
    colors.append([255, 0, 0])

    print('Selected bounding boxes {}'.format(bboxes))

    # Create MultiTracker object
    multiTracker = cv2.MultiTracker_create()

    # Initialize MultiTracker
    for bbox in bboxes:
        multiTracker.add(createTrackerByName(trackerType), frame, bbox)

    # Process video and track objects
    KEY_PRESSED = 1
    j = 0
    n = 0
    while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            frame_r = cv2.resize(frame, (1280, 720))

            # get updated location of objects in subsequent frames
            success, boxes = multiTracker.update(frame_r)

            # draw tracked objects

            for i, newbox in enumerate(boxes):
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))

                cv2.rectangle(frame_r, p1, p2, colors[i], 2, 1)

            # show frame
            if n == 0 or n == 27 or n == 51:
                j = j+1
                frame_crop = frame_r[191:191+130, 540:540+180]
                cv2.imshow('MultiTracker', frame_crop)
                cv2.waitKey()

                destPath = f"D:\joci\projects\Szakdoga_PE\Szakdoga\Szakdolgozat_PE_Doc/01_doc/01_images\objectTrack\mixing/{trackerType}_%02d" % j + ".png"
                cv2.imwrite(destPath, frame_crop)


            n = n + 1
            # quit on ESC button

if __name__ == "__main__":
    main()
