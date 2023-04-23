import pygame as game
import time
import random
import torch.nn as neuralNetwork
import torch
import os
import torch.optim as optim
import numpy as np
from enum import Enum
from collections import namedtuple
game.init()

#colour values
black = (0,0,0)
white = (255,255,255)
blue = (0,0,255)
red = (255,0,0)
pink = (255,192,203)
blue2 = (0,100,255)



# tick rate of the snake game
snakeSpeed = 40

#size of individual snake block
snakeBlock = 20

font =  game.font.SysFont("None", 25)

Point = namedtuple('Point','x , y')

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

class SnakeGame:
    def __init__(self, width = 640, height = 480):
        self.width = width
        self.height = height
        self.disp=game.display.set_mode((self.width,self.height))
        game.display.set_caption("Andre\'s snake game")
        self.clock = game.time.Clock()
        self.reset()
    
    # resets game values
    def reset(self):
        self.head  = Point(self.width/2, self.height/2)
        self.snake = [self.head, Point(self.head.x-snakeBlock,self.head.y),Point(self.head.x-(2*snakeBlock),self.head.y)]
        self.direction = Direction.RIGHT
        self.score = 0
        self.food = None
        self.place_food()
        self.frame_iteration = 0
        
    #place food for snake in a random location
    def place_food(self):
        foodx = round(random.randrange(0, self.width - snakeBlock) / snakeBlock) * snakeBlock
        foody = round(random.randrange(0, self.height - snakeBlock) /snakeBlock) * snakeBlock
        self.food = Point(foodx, foody)
        if(self.food in self.snake):
            self.place_food()

    #plays based on what the agent tells it to
    def play_step(self, action):
        self.frame_iteration += 1
        for event in game.event.get() :
            if event.type==game.QUIT:
                game.quit()
                quit()
        
        self.move(action)

        #add snake block to keep the snake looking like it's moving
        self.snake.insert(0,self.head)

        reward = 0
        game_over = False

        #punish the snake if it dies
        if(self.is_collision() or self.frame_iteration > 100 * len(self.snake)):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        #give reward if the snake eats the food
        if self.head == self.food:
            self.score +=1
            reward = 10
            self.place_food()
        else:
            self.snake.pop()
        
        self.update_UI()
        self.clock.tick(snakeSpeed)

        return reward, game_over, self.score
    
    #move based on the neural network output
    def move(self, action):
        facing_direction = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        index = facing_direction.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_direction = facing_direction[index]
        elif np.array_equal(action, [0,1,0]):
            new_direction = facing_direction[(index + 1) % 4]
        else:
            new_direction = facing_direction[(index - 1) % 4]
        self.direction = new_direction

    #setup coordinates for next snake block to place
        x = self.head.x
        y = self.head.y

        if new_direction == Direction.LEFT:
            x1_change = -snakeBlock
            y1_change = 0
                
        if new_direction == Direction.RIGHT:
            x1_change = snakeBlock
            y1_change = 0
                
        if new_direction == Direction.UP:
            y1_change = -snakeBlock
            x1_change = 0

        if new_direction == Direction.DOWN:
            y1_change = snakeBlock
            x1_change = 0
        
        x = x + x1_change
        y = y + y1_change

        self.head = Point(x, y)
    
    # check if the snake hits itslef or the wall
    def is_collision(self, point=None):
        if point is None:
            point = self.head
        if point.x >= self.width or point.x < 0 or point.y >= self.height or point.y < 0:
            return True
        if point in self.snake[1:]:
            return True
        return False
    
    def update_UI(self):
        self.disp.fill(black)
        for pt in self.snake:
            game.draw.rect(self.disp,blue,game.Rect(pt.x,pt.y,snakeBlock,snakeBlock))
            game.draw.rect(self.disp,pink,game.Rect(pt.x+4,pt.y+4,12,12))
        game.draw.rect(self.disp,red,game.Rect(self.food.x,self.food.y,snakeBlock,snakeBlock))
        text = font.render("Score: "+str(self.score),True,white)
        self.disp.blit(text,[0,0])
        game.display.flip()
    