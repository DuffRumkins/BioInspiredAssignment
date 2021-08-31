import math
import random
import numpy as np
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

#check if iPython is being used (Google Colab) and if so import display
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display

import Constants_v2

IMG_SIZE, SCALE, NUM_SQUARES = Constants_v2.IMG_SIZE,\
                                Constants_v2.SCALE,\
                                Constants_v2.NUM_SQUARES
GRID_SIZE = int(IMG_SIZE/SCALE)
                            

#i is the range of numbers from 1 to GRID_SIZE - 1 that will be used for determining the action space
i = np.arange(1,GRID_SIZE)
#determine action space so it only consists of actions that create unique squares
action_space = 2*GRID_SIZE**3 - GRID_SIZE + np.sum(2*i**2 - 4*GRID_SIZE*i + i - GRID_SIZE)


#define the Deep Q-Network (the policy network) as a class that extends the nn.Module
class DQN(nn.Module):
    def __init__(self, img_height, img_width):
        #the super call delegates the function call to the parent class (nn.Module) this is apparently needed to initialize the nn.Module properly
        super().__init__()

        """#first layer is fully connected (Linear) and takes img_height*img_width inputs and has 24 outputs
        #if we were taking RGB/BGR images as inputs we would need to multiply this by 3 to account for all channels
        self.fc1 = nn.Linear(in_features = img_height*img_width, out_features=24)
        #second layer is also fully connected and takes 24 inputs (outputs from previous channel) and has 32 outputs
        self.fc2 = nn.Linear(in_features=24, out_features= 32)
        #output layer is also fully connected and has as many outputs as possible actions (number of gridpoints * number of different square sizes)
        self.out = nn.Linear(in_features = 32, out_features= GRID_SIZE**3)"""

        #first layer is fully connected (Linear) and takes img_height*img_width inputs and has 24 outputs
        #if we were taking RGB/BGR images as inputs we would need to multiply this by 3 to account for all channels
        self.fc1 = nn.Linear(in_features = IMG_SIZE*IMG_SIZE, out_features=24)
        #second layer is also fully connected and takes 24 inputs (outputs from previous channel) and has 32 outputs
        self.fc2 = nn.Linear(in_features=24, out_features= 32)

        self.fc3 = nn.Linear(in_features=32, out_features= 32)
        
        #output layer is also fully connected and has as many outputs as possible actions (action_space)
        self.out = nn.Linear(in_features = 32, out_features= action_space)

        #CHANGE SETUP OF THIS NETWORK -> CONVOLUTIONAL LAYERS (pool())

        """self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.out = nn.Linear(in_features = img_height*img_width*24, out_features= GRID_SIZE**3)"""

        """self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()

        self.pool = nn.MaxPool2d(kernel_size=SCALE)

        self.conv2 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        #self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        #self.relu3 = nn.ReLU()
        #self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        #self.relu4 = nn.ReLU()
        self.out = nn.Linear(in_features = int((IMG_SIZE*IMG_SIZE)/SCALE)*4, out_features= GRID_SIZE**3)"""
        
    #all PyTorch neural networks required an implementation of forward() which defines a forward pass to the network
    #forward() takes an image tensor t
    def forward(self, t):
        """#the image tensore must first be flattened (made 1D)
        t = t.flatten(start_dim = 1)
        #t is then passed through the layers using ReLU activation functions
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)"""

        #the image tensore must first be flattened (made 1D)
        t = t.flatten(start_dim = 1)
        #t is then passed through the layers using ReLU activation functions
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = F.relu(self.fc3(t))
        t = self.out(t)

        """t = self.relu1(self.conv1(t))
        t = self.relu2(self.conv2(t))
        t = self.relu3(self.conv3(t))
        t = self.relu4(self.conv4(t))
        t = t.flatten(start_dim = 1)
        t = self.out(t)"""

        """t = self.relu1(self.conv1(t))
        t = self.relu2(self.conv2(t))
        t = self.pool(t)
        t = self.out(t)"""
        
        return t

#we store experiences in a namedtuple class called experience. This makes it easier to keep track of everything
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))
#to see how the Experience class works uncomment the following code
#e = Experience(2,3,1,4)
#print (e)

#define a class for the replay memory
class ReplayMemory():
    #the replay memory takes a capacity input defining its size
    def __init__(self,capacity):
        self.capacity = capacity
        #define the memory structure where experiences will be stored as a list
        self.memory = []
        #define push count attribute to keep track of the total number of experiences added to memory
        self.push_count = 0

    #define the push function which will store experiences in memory
    def push(self, experience):
        #check if there is space in memory
        if len(self.memory) < self.capacity:
            #if so append experience to the memory list
            self.memory.append(experience)
        else:
            #if not replace the oldest experience in memory
            self.memory[self.push_count % self.capacity] = experience
        #iterate the push_count
        self.push_count += 1

    #define sample function that samples a random batch of batch_size experiences from memory
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    #define can_provide_sample function that returns a boolean denoting if there are enough experiences stored in memory to sample a batch of batch_size
    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size

#define a class for the epsilon greedy strategy
class EpsilonGreedyStrategy():
    #the class takes a starting exploration rate (epsilon), ending exploration rate, an exploration rate decay and the device (CPU/GPU) that will be used for tensor calculations
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay
        self.device = device

    #define a function which returns the exploration rate (epsilon) at the current step
    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * math.exp(-1*current_step*self.decay)

#define a class for the agent
class Agent():
    #we initialise the agent with a strategy (epsilon greedy) and number of actions the agent can take at each step. We also set the current step to 0
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    #define the function used by the agent to choose the action. This is done using the policy network (DQN) and the state
    def select_action(self, state, policy_net):
        #first the current exploration rate (epsilon) is determined
        rate = self.strategy.get_exploration_rate(self.current_step)
        #iterate current step
        self.current_step += 1

        #check if the exploration rate is greater than a randomly generated number (0-1)
        if rate > random.random():
            #if so the agent explores and takes a random action
            #return as a tensor for consitency and tensor processing later
            return torch.tensor([random.randrange(self.num_actions)], device=self.device)
        else:
            #if not the agent takes the optimal action (exploits)
            #we specify that we pass data to our policy network with gradient tracking turned off since we are currently using the model for inference not training
            #Note that during training PyTorch would need to keep track of all forward pass calculations so it can apply back propagation later
            with torch.no_grad():
                #the agent decides the optimal action using the policy network
                #we also specify that the agent uses the device initialised previously for this calculation
                return policy_net(state).argmax(dim=1).to(self.device)

#define a class for the environment       
class Environment():
    #initialise environment as black image with the shape of a reference image file
    def __init__(self, ref, num_squares, device):
        """Initialise environment with reference image file location and a device
            Only takes grayscale images for now"""

        #read the original image as a GRAYSCALE image
        original_ref = cv.imread(ref, cv.IMREAD_GRAYSCALE)
        #determine height and width of original image
        #WILL NEED TO BE UPDATED FOR COLOUR
        original_height, original_width = np.shape(original_ref)
        #determine the minimum dimension
        min_dim = min(original_height, original_width)
        #crop image to only take center square using the minimum dimension
        cropped = original_ref[int(0.5*(original_height-min_dim)):int(0.5*(original_height+min_dim)),int(0.5*(original_width-min_dim)):int(0.5*(original_width+min_dim))]
        #resize the image to the grid size
        resized = cv.resize(cropped,(IMG_SIZE,IMG_SIZE))
        
        self.ref = resized

        #self.ref = cv.imread(ref, cv.IMREAD_GRAYSCALE)
        self.img = np.zeros(np.shape(self.ref), np.uint8)
        self.action_limit = num_squares
        self.action_count = 0
        self.device = device
        self.action_space = action_space
        self.height, self.width = np.shape(self.ref)

        #initialise the grid as a square matrix of ones of size GRID_SIZE
        self.grid = 6*np.ones((GRID_SIZE,GRID_SIZE))
        #iterate through each of the rows in the grid and set each grid point to the number of unique actions possible from that point
        for row in range(GRID_SIZE):
            self.grid[row:,row:] = (GRID_SIZE-row)*np.ones(self.grid[row:,row:].shape)

        #initialise actions_taken list that keeps track of actions taken
        #self.actions_taken = [None]

        #initialise previous_states list that keeps track of previous states
        self.previous_states = np.zeros((1,self.height,self.width))

    #define reset function to reset environment to its initial state
    def reset(self):
        self.action_count = 0
        self.img = np.zeros(np.shape(self.ref), np.uint8)
        #self.actions_taken = [None]
        self.previous_states = np.zeros((1,self.height,self.width))
        return

    #define a function that takes the index of an action and outputs the row, column and size of the corresponding square
    def determine_square(self,action):
        #iterate through each of the rows in the grid
        for row in range(GRID_SIZE):
            #check if the action belongs to the current row
            if action < np.sum(self.grid[row]):
                #iterate through each of the columns in the current row
                for column in range(GRID_SIZE):
                    #determine value of the point at the current row and column
                    point = self.grid[row,column]
                    #check if the square starts at this point
                    if action < point:
                        #if so return the row, column and the square size as a tuple
                        return int(row), int(column), int(action + 1)
                    #if the square does not belong to this point subtract the value of the point from the action
                    else:
                        action -= point
            #if the square does not correspond to this row subtract the sum of the column from the action
            else:
                action -= np.sum(self.grid[row])
        return False

    #define take_action function which takes the index of an action and then updates the environment and outputs the reward
    def take_action(self, action):
        #iterate the action count
        self.action_count += 1

        #determine 1 - MSE before taking action
        #prev_MSE = 1-(np.sum((self.ref-self.img)**2)/self.img.size)
        
        #determine square size and corresponding row and column from the action index
        #size = int(action%GRID_SIZE + 1)
        #grid_point = int(action/GRID_SIZE)
        #row = int(grid_point/GRID_SIZE)
        #column = grid_point%GRID_SIZE

        #determine the row, column and size of the square from the action index
        row, column, size = self.determine_square(int(action))

        #from row and column determine top left and bottom right corner of the to-be-drawn square
        square_top_left = np.array([column*SCALE,row*SCALE]) #in format (x,y) for cv.rectangle()
        square_bottom_right = square_top_left+size*SCALE - 1 #subtract 1 because cv.rectangle() includes last index
        #the intensity (for grayscale images) is taken as the average intensity of the pixels in the same square in the reference image
        #we add 1 to the square_bottom_right values to compensate for the 1 being previously subtracted
        square_intensity = np.mean(self.ref[square_top_left[1]:square_bottom_right[1]+1,
                                            square_top_left[0]:square_bottom_right[0]+1])
        #the square is drawn onto the image, note the weird syntax for cv.rectangle() --> see above comments
        cv.rectangle(self.img, tuple(square_top_left), tuple(square_bottom_right),
                     int(square_intensity),-1)

        state = self.ref.astype(np.float32) - self.img.astype(np.float32)
        
        #define the reward function
        #THIS MAY NEED SOME SERIOUS TUNING
        #reward = (self.img.size*255-np.sum(abs(state)))/(self.img.size*255)
        #Mean Square Error reward function
        #!Convert to arrays to numpy arrays to floats to prevent negatives rounding around from 256
        reward = -(np.sum(state**2)/self.img.size)

        #reward function that takes change in 1 - MSE error before and after an action as the reward
        #new_MSE = 1-(np.sum((self.ref-self.img)**2)/self.img.size)
        #reward = new_MSE - prev_MSE

        #check if the action taken is the same as the previous action
        #if self.actions_taken[-1] == action:
            #if so subtract 1000 from the reward
            #reward -= 1000
        
        #check if the reward received is the same as the previous reward
        #MAY NEED TO UPDATE FOR RGB
        if (state == self.previous_states).all((1,2)).any():
            #if so subtract the absolute value of the reward from the reward
            reward -= np.abs(reward)

        #append the action to actions_taken list
        #self.actions_taken.append(action)

        #expand the 0th dimension of the states matrix for concatentaion
        state = np.expand_dims(state,0)
        #append the state to the previous_states list
        self.previous_states = np.concatenate((self.previous_states,state),0)

        #return the reward in a tensor using the device. This is for tensor processing later
        #SEEMS SLOW BUT LEAVING FOR NOW MAY NEED TO CONSIDER NOT CONVERTING TO TENSOR
        #MIGHT JUST BE SLOW FOR START UP
        return torch.tensor([reward], device=self.device)

    #define a function to render the environment
    def render(self):
        #render both the environment image and the reference image in adjacent subplots
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.imshow(self.img.astype(np.uint8), cmap='gray')
        ax1.set_title("Environment")
        ax2.imshow(self.ref, cmap='gray')
        ax2.set_title("Reference")

        #this combintation allows for the plot to be shown without pausing Python
        #the plots are drawn in the pause
        plt.pause(0.001)
        #plt.ion() does the non-blocking magic I think
        plt.ion()
        plt.show()
        return

    #define a function to close all environment windows
    def close(self):
        plt.ioff()
        plt.close()
        return

    #define a function that gets the state that will be passed through to the DQN
    def get_state(self):
        #take tensor of difference between reference and environment image as state we also add another batch dimension

        """#check if the agent is finished and the episode is done
        if self.done():
            #set final state to np array the same shape as the original image but with all elements set to -5000. This is to allow the final state to be easily distinguishable by the QValues get_next static method
            final_state = np.ones_like(self.img)*-5000
            #convert np array to float tensor and add an extra batch dimension
            #FOR COLOUR IMAGES THE NP ARRAY WILL HAVE TO BE TRANSPOSED USING (self.ref - self.img).transpose((2,0,1))
            #MAKE SURE RGB not BGR!!
            return torch.from_numpy(final_state).type(torch.FloatTensor).unsqueeze(0).to(self.device)"""

        #if it is not the final state we return the difference array as a float tensor and add an extra batch dimension
        #FOR COLOUR IMAGES THE NP ARRAY WILL HAVE TO BE TRANSPOSED USING (self.ref - self.img).transpose((2,0,1))
        #MAKE SURE RGB not BGR!!
        #!Convert to arrays to numpy arrays to floats to prevent negatives rounding around from 256
        return torch.from_numpy(self.ref.astype(np.float32) - self.img.astype(np.float32)).type(torch.FloatTensor).unsqueeze(0).to(self.device)

    #define a function that returns True if the agent has reached it's action limit and False otherwise
    def done(self):
        if self.action_count >= self.action_limit:
            return True
        else:
            return False

#define a class for the Q-Values
class QValues():
    #this class contains 2 static methods meaning that they can be called without first creating an instance of the class.
    #since we will not be creating an instance of this class the device must be redefined
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #we define a static function that returns a tensor of the predicted q-values from the policy network for the state action-pair that was passed in
    @staticmethod
    def get_current(policy_net, states, actions):
        #remember the output of the policy network for a given state is the q-values of each action
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

    """#we define a second static function to determine the predicted q-value for the next states while making sure to set the q-value for any actions which will result in the final state to 0
    @staticmethod
    def get_next(target_net, next_states):
        #determine indices of the final states by determining which states are tensors where the max value is equal to -5000
        final_state_locations = next_states.max(dim=1)[0].eq(0).type(torch.bool)
        #FOR COLOUR IMAGES WE NEED TO FLATTEN THE CHANNELS DIMENSION (DIMENSION 1 SINCE IMAGE BATCH TENSORS IN FORMAT BCHW) OF THE next_states TENSOR, SEE BELOW
        #final_state_locations = next_states.flatten(start_dim=1)max(dim=1)[0].eq(0).type(torch.bool)
        #determine the indices of the states of the next_states that are not final states
        non_final_states_locations = (final_state_locations == False)
        #extract the value of the non final states into its own tensor
        non_final_states = next_states[non_final_states_locations]
        #take the batch size from the shape of the image batch shape (zero-th dimension)
        batch_size = next_states.shape[0]
        #initialise values tensor which will contain the q-values of the next states as tensor of zeros
        values = torch.zeros(batch_size).to(QValues.device)
        #set the values of the non-final states using the target network. The q-values of the final states remain 0 because the agent is unable to recieve any reward once the episode has ended
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0]
        return values"""

    #alternative solution for get_next() that uses the fact that if the agent is 1 action away from the limit all following actions will result in the episode ending
    #WILL NOT WORK IF THE AGENT IS ALLOWED TO FINISH EARLY (i.e THE REFERENCE IMAGE AND THE ENVIRONMENT IMAGE ARE IDENTICAL) -> DON'T THINK THIS IS WORTH IMPLEMENTING
    @staticmethod
    def get_next(target_net, next_states, action_count, action_limit):
        if action_count >= (action_limit - 1):
            #determine value of the rewards of the final states using mean square error
            #WILL NEED TO BE UPDATED FOR RGB
            final_rewards = -torch.sum(next_states**2,(1,2))/(torch.prod(torch.tensor(next_states[0].size()),0))
            #print (final_rewards)
            return final_rewards
        return target_net(next_states).max(dim=1)[0]
        

#define a function that takes a batch of experiences and extracts 4 separate tensors containing the states, actions, next_states and rewards, respectively
def extract_tensors(experiences):
    batch = Experience(*zip(*experiences))

    #the first tensor  contains the states
    t1 = torch.cat(batch.state)
    #the second tensor contains the actions
    t2 = torch.cat(batch.action)
    #the third tensor contains the rewards
    t3 = torch.cat(batch.reward)
    #the fourth tensor contains the next states
    t4 = torch.cat(batch.next_state)

    #return a tuple containing each of the tensors
    return (t1, t2, t3, t4)

    

#the following utility functions have been taken from https://deeplizard.com/learn/video/jkdXDinWfo8
def plot(values, moving_avg_period):
    plt.figure(2)
    plt.clf()        
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Difference')
    plt.plot(values)

    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(get_moving_average(moving_avg_period, values))
    plt.pause(0.001)
    print(f"Episode {len(values)} \n {moving_avg_period} episode moving average {moving_avg[-1]}")
    #special case for iPython inlinse display
    if is_ipython: display.clear_output(wait=True)

def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1) \
            .mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()

"""----------------------------------------------------------------------Code-----------------------------------------------------------"""

#Hyperparameters 

#batch size is how many samples are propagated through the network in one forward/backward pass
#apparently a batch size of 32 is good for image recognition
#batch_size = 256
batch_size = 64
#discount rate used in Bellman equation
#gamma = 0.999
gamma = 1 - 1/NUM_SQUARES
#starting exploration rate
eps_start = 1
#final exploration rate
eps_end = 0.01
#exploration rate decay 
eps_decay = 0.001
#eps_decay = 0.0001
#number of episodes between each update of the target network
target_update = 10
#experience capacity of the memory 
memory_size = 100000
#the learning rate
lr = 0.0001
#lr = 0.001
#number of episodes we use to train the network
num_episodes = 3000

#set the device to be the GPU if available and otherwise the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#"""
#initialise the environment, strategy and memory
env = Environment("TestImages/TestImage12.png", NUM_SQUARES, device)
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
agent = Agent(strategy, env.action_space, device)
memory = ReplayMemory(memory_size)

#initialise the policy network and the target network
policy_net = DQN(env.height,env.width).to(device)


#set the optimizer to the Adam optimizer (Adam Algorithm) which accepts our policy network parameters and learning rate
optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

#create list of final image differences in order to plot them with the plot() utility function
final_differences = []

#initialise loss as 0
loss = 0

#TRAINING LOOP
#iterate through num_episodes episodes
for episode in range(num_episodes):

    #reload the networks every 500 episodes
    if episode%500 == 0:
        #save model checkpoint
        torch.save({'model_state_dict': policy_net.state_dict(),
                    'optimizer_state_dict' : optimizer.state_dict(),
                    'loss': loss}, "DQN/policy_netV2.pt")
        #load checkpoint
        checkpoint = torch.load("DQN/policy_netV2.pt")

        #clear GPU cache
        torch.cuda.empty_cache()
        
        #load model state_dict
        policy_net.load_state_dict(checkpoint["model_state_dict"])
        #load optimizer_state_dict
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        #load loss
        loss = checkpoint["loss"]
    
        #set policy_net to training mode
        policy_net.train()
        
        #initialise the target network
        target_net = DQN(env.height,env.width).to(device)
        #set the weights and biases of the target network to equal those of the policy network using load_state_dict()
        target_net.load_state_dict(policy_net.state_dict())
        #put the target network in eval mode which tells PyTorch that the network is not in training mode and will only be used for inference
        target_net.eval()

    #reset the environment
    env.reset()
    #get starting state
    state = env.get_state()

    #iterate through timesteps (count() will keep counting integers until the loop is broken)
    for timestep in count():
        #determine action taken by agent (returns action index)
        action = agent.select_action(state, policy_net)
        #determine reward based on environment
        reward = env.take_action(action)
        #determine new state after taking action
        next_state = env.get_state()
        #push the experience based on the starting state, action, the new state and the reward to memory
        memory.push(Experience(state, action, next_state, reward))
        #set the state to the new state
        state = next_state

        #check if there are enough experiences stored in memory that a sample of size batch_size can be taken
        if memory.can_provide_sample(batch_size):
            #if so take a sample of experiences from memory
            experiences = memory.sample(batch_size)
            #extract separate tensors of the states, actions, rewards and next_states of the experiences in the batch using the previously defined extract_tensors() function
            states, actions, rewards, next_states = extract_tensors(experiences)

            #determine current q-values using the get_current static method in the QValues class
            current_q_values = QValues.get_current(policy_net, states, actions)
            #determine next q-values using the get_next static method in the QValues class
            #next_q_values = QValues.get_next(target_net, next_states)
            #alternative method
            next_q_values = QValues.get_next(target_net, next_states, env.action_count, env.action_limit)
            #determine target q-values using the formula q_*(s,a) = E[R_{t+1} + gamma*max_{a'}q_*(s',a')]
            target_q_values = ((next_q_values * gamma) + rewards).type(torch.cuda.FloatTensor)

            #determine the loss between the target q-values and current q-values using mean square error function
            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            #we set the optimizer gradient to zero since PyTorch accumulates the gradients when it does backwards propagation. If we didn't do this we'd be accumulating gradients each time backwards propagation is run
            optimizer.zero_grad()
            #we compute the gradient of the loss function with respect to all the policy network weights and biases
            loss.backward()
            #with .step() we update the weights and biases in the policy network using the gradients calculated with loss.backward()
            optimizer.step()

        #check if the agent is finished
        if env.done():
            #if so append the final difference between the reference and environment image to the final_differences list
            final_differences.append(np.sum(abs(env.ref-env.img)))
            #plot the moving average over 100 episodes using the plot() utility function
            plot(final_differences, 100)
            #break out of the loop so the next episode can start
            break

    #check if enough episodes have passed that the target network should be updated
    if episode % target_update == 0:
        #update the target network using the weights and biases of the policy network
        target_net.load_state_dict(policy_net.state_dict())


#reset the environment after the policy network has been trained
env.reset()
#"""
env = Environment("TestImages/TestImage12.png", NUM_SQUARES, device)
policy_net = DQN(env.height,env.width).to(device)
checkpoint = torch.load("DQN/policy_netV2.pt")
policy_net.load_state_dict(checkpoint["model_state_dict"])
policy_net.eval()

#create a new agent with a greedy strategy
greedy_strategy = EpsilonGreedyStrategy(0,0,0)
greedy_agent = Agent(greedy_strategy, NUM_SQUARES, device)

#run the greedy agent
while not env.done():
  state = env.get_state()
  action = greedy_agent.select_action(state, policy_net)
  env.take_action(action)

#render the result of the greedy agent's actions
env.render()     
    
