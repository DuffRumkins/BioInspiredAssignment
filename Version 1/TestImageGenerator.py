import numpy as np
import cv2 as cv
import Constants

GRID_SIZE, SCALE, NUM_SQUARES = Constants.GRID_SIZE,\
                                Constants.SCALE,\
                                Constants.NUM_SQUARES


img = np.zeros((GRID_SIZE*SCALE,GRID_SIZE*SCALE),np.uint8)

for square in range(NUM_SQUARES):
    square_size = np.random.randint(1,GRID_SIZE+1)*SCALE
    square_top_left = np.random.randint(0,GRID_SIZE,2)*SCALE
    square_bottom_right = square_top_left + square_size - 1 #subtract 1 because cv.rectangle() includes last index
    square_intensity = 255*np.random.randint(0,2)
    #print (square_size,square_top_left,square_bottom_right,square_intensity)
    cv.rectangle(img, tuple(square_top_left), tuple(square_bottom_right),
                 int(square_intensity),-1)
    
    
if cv.imwrite("TestImages/TestImage.png", img) == True:
    print ("A new test image has been succesfully generated")
else:
    print ("The new test image could not be saved")

