import numpy as np
import cv2 as cv

import Constants_v2

IMG_SIZE, SCALE, NUM_SQUARES, NUM_IMAGES = Constants_v2.IMG_SIZE,\
                                           Constants_v2.SCALE,\
                                           Constants_v2.NUM_SQUARES,\
                                           Constants_v2.NUM_IMAGES
GRID_SIZE = int(IMG_SIZE/SCALE)

#i is the range of numbers from 1 to GRID_SIZE - 1 that will be used for determining the action space
i = np.arange(1,GRID_SIZE)
#determine action space so it only consists of actions that create unique squares
action_space = 2*GRID_SIZE**3 - GRID_SIZE + np.sum(2*i**2 - 4*GRID_SIZE*i + i - GRID_SIZE)

#initialise the grid as a square matrix of ones of size GRID_SIZE
grid = 6*np.ones((GRID_SIZE,GRID_SIZE))
#iterate through each of the rows in the grid and set each grid point to the number of unique actions possible from that point
for row in range(GRID_SIZE):
    grid[row:,row:] = (GRID_SIZE-row)*np.ones(grid[row:,row:].shape)

def determine_square(action):
        #iterate through each of the rows in the grid
        for row in range(GRID_SIZE):
            #check if the action belongs to the current row
            if action < np.sum(grid[row]):
                #iterate through each of the columns in the current row
                for column in range(GRID_SIZE):
                    #determine value of the point at the current row and column
                    point = grid[row,column]
                    #check if the square starts at this point
                    if action < point:
                        #if so return the row, column and the square size as a tuple
                        return int(row), int(column), int(action + 1)
                    #if the square does not belong to this point subtract the value of the point from the action
                    else:
                        action -= point
            #if the square does not correspond to this row subtract the sum of the column from the action
            else:
                action -= np.sum(grid[row])
        return False


for i in range(NUM_IMAGES):
    img = np.zeros((GRID_SIZE*SCALE,GRID_SIZE*SCALE),np.uint8)

    for square in range(NUM_SQUARES):
        action = np.random.randint(0,action_space)

        row, column, size = determine_square(int(action))
        
        #from row and column determine top left and bottom right corner of the to-be-drawn square
        square_top_left = np.array([column*SCALE,row*SCALE]) #in format (x,y) for cv.rectangle()
        square_bottom_right = square_top_left+size*SCALE - 1 #subtract 1 because cv.rectangle() includes last index
        #the intensity (for grayscale images) is taken as the average intensity of the pixels in the same square in the reference image
        #we add 1 to the square_bottom_right values to compensate for the 1 being previously subtracted
        square_intensity = np.random.randint(0,256)
        #the square is drawn onto the image, note the weird syntax for cv.rectangle() --> see above comments
        cv.rectangle(img, tuple(square_top_left), tuple(square_bottom_right),
                     int(square_intensity),-1)
    
    
    if cv.imwrite(f"TestImages/TestImage{i}.png", img) == True:
        print ("A new test image has been succesfully generated")
    else:
        print ("The new test image could not be saved")

