#!/usr/bin/python3

import jetson.inference
import jetson.utils

import argparse
import sys

import cv2 
import numpy as np 
labels = {1: "person", 2 :"bicycle", 3 :"car", 4 :"motorcycle", 5 :"airplane", 6 :"bus", 7 :"train", 8 :"truck", 9 :"boat", 10: "traffic light", 11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench", 16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe", 27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase", 34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite", 39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard", 43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl", 52: "banana", 53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut", 61: "cake", 62: "chair", 63: "couch", 64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet", 72: "tv", 73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard", 77: "cell phone", 78: "microwave", 79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator", 84: "book", 85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear", 89: "hair drier", 90: "toothbrush"}

net = jetson.inference.detectNet("ssd-mobilenet-v2", sys.argv, 0.5)

# create video sources
cap = cv2.VideoCapture(2)


if __name__ == "__main__":
	while True:
		# capture the next image
		
		retval, frame = cap.read()
		img = jetson.utils.cudaFromNumpy(frame)

		# detect objects in the image (with overlay)
		detections = net.Detect(img)

		# print the detections
		print("detected {:d} objects in image".format(len(detections)))
		for det in detections:
			if det.ClassID == 1:
				cv2.rectangle(frame, (int(det.Left), int(det.Top)), (int(det.Right), int(det.Bottom)), (0, 0, 255))
				cv2.circle(frame, (int(det.Center[0]), int(det.Center[1])), 10, (255, 0, 0))

		
		cv2.imshow("Capture", frame)
		cv2.waitKey(10)

