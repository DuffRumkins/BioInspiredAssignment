import numpy as np
import cv2 as cv
import Constants

GRID_SIZE, SCALE, NUM_SQUARES = Constants.GRID_SIZE,\
                                Constants.SCALE,\
                                Constants.NUM_SQUARES


ref = cv.imread("TestImages/TestImage.png", cv.IMREAD_GRAYSCALE)

cv.imshow("Reference Image", ref)

img = np.zeros(np.shape(ref),np.uint8)

grid = input("Would you like to see the grid on the reference image? (yes/no) ")

ref_grid = ref.copy()

if grid[0] == 'y':
    for line in range(1,GRID_SIZE):
        ref_grid[line*SCALE,:] = 255
        ref_grid[:,line*SCALE] = 255

    cv.imshow("Image", ref_grid)
    cv.waitKey(1)
    

play = True

while play:

    point = int(input(f"At which grid point would you like to place a square? (1-{GRID_SIZE**2}) ")) - 1

    while point not in range(GRID_SIZE**2):
        print ("Bad input, try again")
        point = int(input(f"At which grid point would you like to place a square? (1-{GRID_SIZE**2}) ")) - 1
    
    point_row = int(point/GRID_SIZE)
    point_column = point%GRID_SIZE


    size = int(input(f"What size square would you like to draw? (1-{GRID_SIZE}) "))

    while size not in range(1,GRID_SIZE+1):
        print ("Bad input, try again")
        size = int(input(f"What size square would you like to draw? (1-{GRID_SIZE}) "))

    print (point_row, point_column)
    square_top_left = np.array([point_column*SCALE,point_row*SCALE]) #in format (x,y) for cv.rectangle()
    square_bottom_right = square_top_left+size*SCALE
    square_intensity = np.mean(ref[square_top_left[1]:square_bottom_right[1],
                                   square_top_left[0]:square_bottom_right[0]])
    cv.rectangle(img, tuple(square_top_left), tuple(square_bottom_right),
                 int(square_intensity),-1)


    cv.destroyAllWindows()
    cv.imshow("Image", img)

    cv.imshow("Reference Image", ref_grid)
    cv.waitKey(1)


    stop = input("Would you like to finish? (yes/no)")

    if stop != '' and stop[0] == 'y':
        play = False
