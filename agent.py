from snakeMachineLearning import SnakeGame, Point, Direction, snakeBlock
import random
import torch
from model import Linear_QNet, QTrainer
from collections import deque
import numpy as np
from snakeMachineLearning import SnakeGame
from helper import plot

#number of training examples used in one iteration 
# higher = faster but takes more memory
BATCH_SIZE = 1000

#the maximum number of game states that are stored in memory for training
#prevents from consuming too much energy
MAX_MEMORY = 100_000

#learning rate of the algorithm, which controls how much the weights of 
#the model are updated in response to each batch of training examples
# higher eaquals faster learning but less stable/accurate results
# lower = slower leaning but more accurate and stable seults
LR = 0.001

class Agent:
    def __init__(self):
        #number of games played in the current training session
        self.n_game = 0

        #controls how often the program will take a random action instead
        # of following it's learned policy
        #higher equals more random
        self.epsilon = 0

        #discount factor used in the Q-learning algorithm
        # higher equals prioritizes more long term rewards
        # lower equals prioritizses more immediate rewards
        self.gamma = 0.9

        # will get rid of older data once the memory is exceeded 
        # and replace it with the older data
        self.memory = deque(maxlen = MAX_MEMORY)

        #sets up the neural network that uses deep q learning
        self.model = Linear_QNet(11, 256, 3)

        # come back to this
        self.trainer = QTrainer(self.model, lr=LR, gamma =self.gamma)
    
    #gets the current state of the snake game. returns an array of 
    #boolean values describing the state with 1 as True and 0 as false
    def get_state(self,game):
        head = game.snake[0]
        point_left = Point(head.x - snakeBlock, head.y)
        point_right = Point(head.x + snakeBlock, head.y)
        point_up = Point(head.x , head.y - snakeBlock)
        point_down = Point(head.x , head.y + snakeBlock)

        direction_left = game.direction == Direction.LEFT
        direction_right = game.direction == Direction.RIGHT
        direction_up = game.direction == Direction.UP
        direction_down = game.direction == Direction.DOWN

        state = [
            #danger in front of the snake
            (direction_up and game.is_collision(point_up)) or
            (direction_down and game.is_collision(point_down)) or
            (direction_right and game.is_collision(point_right)) or
            (direction_left and game.is_collision(point_left)),

            #danger right of the snake
            (direction_up and game.is_collision(point_right)) or
            (direction_down and game.is_collision(point_left)) or
            (direction_right and game.is_collision(point_down)) or
            (direction_left and game.is_collision(point_up)),

            #danger left of the snake
            (direction_up and game.is_collision(point_left)) or
            (direction_down and game.is_collision(point_right)) or
            (direction_right and game.is_collision(point_up)) or
            (direction_left and game.is_collision(point_down)),

            direction_right,
            direction_left,
            direction_up,
            direction_down,

            game.food.x < game.head.x, # food is to the left
            game.food.x > game.head.x, # food is to the right
            game.food.y < game.head.y, # food is up
            game.food.y > game.head.y # food is down

        ]
        return np.array(state, dtype = int)

    #Gets next move to make from the model based on 
    #the current state of the snake
    def get_action_from_model(self, state):
        # actions become less random the more games the ai plays
        self.epsilon = 80 - self.n_game

     #decides what move to make based on the output nodes
        final_move = [0, 0, 0]

        #this creates random behaviour from the ai. if a random number is 
        #smaller than the learning rate then a random action will be taken.
        #As the learning rate grows less random actions are taken
        if(random.randint(0, 200) < self.epsilon):
            move = random.randint(0, 2)
            final_move[move] = 1
        
        # if not use the learning policy of the ai
        else:
            state0 = torch.tensor(state, dtype=torch.float).cuda()
            prediction = self.model(state0).cuda()
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move

    # This method performs a single training step
    # updates the model's weights based on the current experience.
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    #This method performs a batch update to the model's weights
    def train_long_memory(self):

        #if a certain amount of steps have been played in a single game take a sample
        if (len(self.memory) > BATCH_SIZE):
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        
        #if not take all steps to update the model weights
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    #add the step played to memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

def train():
    # variables for plotting the learning
    plot_scores = []
    plot_mean_score = []

    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame()

    while True:

        #play a step 
        state_old = agent.get_state(game)
        final_move = agent.get_action_from_model(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        #if the game is done
        if done:
            game.reset()
            agent.n_game += 1
            #update the model with all the steps of game if under batch size
            agent.train_long_memory()
            if score > record:
                record = score
                agent.model.save()
            print('Game:',agent.n_game,'Score:',score,'Record:',record)
            
            #plotting
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_game
            plot_mean_score.append(mean_score)
            plot(plot_scores, plot_mean_score)

train()
