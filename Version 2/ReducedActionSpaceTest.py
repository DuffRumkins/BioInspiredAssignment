import numpy as np
import Constants_v2

IMG_SIZE, SCALE, NUM_SQUARES = Constants_v2.IMG_SIZE,\
                                Constants_v2.SCALE,\
                                Constants_v2.NUM_SQUARES
GRID_SIZE = int(IMG_SIZE/SCALE)
GRID_SIZE = 6

i = np.arange(1,GRID_SIZE)
action_space = 2*GRID_SIZE**3 - GRID_SIZE + np.sum(2*i**2 - 4*GRID_SIZE*i + i - GRID_SIZE)

grid = 6*np.ones((GRID_SIZE,GRID_SIZE))
for row in range(GRID_SIZE):
    grid[row:,row:] = (GRID_SIZE-row)*np.ones(grid[row:,row:].shape)

def determine_square(index,grid):
    for row in range(GRID_SIZE):
        if index < np.sum(grid[row]):
            for column in range(GRID_SIZE):
                point = grid[row,column]
                if index < point:
                    return row, column, index
                else:
                    index -= point
        else:
            index -= np.sum(grid[row])
    return False
