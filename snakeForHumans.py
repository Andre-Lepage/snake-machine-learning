import pygame
import time
import random
pygame.init()


pygame.display.set_caption("Andre\'s snake game")

black = (0,0,0)
white = (255,255,255)
blue = (0,0,255)
red = (255,0,0)



disp_width = 800
disp_height = 600

disp=pygame.display.set_mode((disp_width,disp_height))

snakeSpeed = 15
snakeBlock = 20

clock = pygame.time.Clock()

font_style = pygame.font.SysFont("None", 50)



def message(msg, colour):
    mesg = font_style.render(msg, True, colour)
    disp.blit(mesg, [disp_width/10, disp_height/3])

def displayScore(score, colour):
    mesg = font_style.render(str(score), True, colour)
    disp.blit(mesg, [10, 10])   

def drawSnake( snakeList):
    for block in snakeList:
        pygame.draw.rect(disp,blue, [block[0],block[1],snakeBlock,snakeBlock])


def gameLoop():
    
    game_close = False
    game_over = False

    score = 0

    x1 = disp_width/2
    y1 = disp_height/2
    
    snakeLength = 1

    snakeList = []

    moving = ""

    x1_change = 0
    y1_change = 0

    foodx = round(random.randrange(0, disp_width - snakeBlock) / snakeBlock) * snakeBlock
    foody = round(random.randrange(0, disp_height - snakeBlock) /snakeBlock) * snakeBlock

    while not game_close:


        while game_over == True:
            disp.fill(white)
            message("You Lost: click q to quit, click c to continue", red)
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        game_close = True
                        game_over = False
                    if event.key == pygame.K_c:
                        gameLoop()
        
        

        for event in pygame.event.get() :
            if event.type==pygame.QUIT:
                    game_close=True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    if moving != "right":
                        x1_change = -snakeBlock
                        y1_change = 0
                    moving = "left"
                if event.key == pygame.K_RIGHT:
                    if moving != "left":
                        x1_change = snakeBlock
                        y1_change = 0
                    moving = "right"
                if event.key == pygame.K_UP:
                    if moving != "down":
                        y1_change = -snakeBlock
                        x1_change = 0
                    moving = "up"
                if event.key == pygame.K_DOWN:
                    if moving != "up":
                        y1_change = snakeBlock
                        x1_change = 0
                    moving = "down"
        if x1 >= disp_width or x1 < 0 or y1 >= disp_height or y1 < 0 :
            game_over = True

        x1 += x1_change
        y1 += y1_change

        disp.fill(white)
        pygame.draw.rect(disp,red,[foodx,foody,snakeBlock,snakeBlock])

        snakeHead = [x1, y1]
        snakeList.append(snakeHead)

        if len(snakeList) > snakeLength:
            del snakeList[0]
 
        for block in snakeList[:-1]:
            if block == snakeHead:
                game_over = True
        
        displayScore(score, red)
        drawSnake(snakeList)
        pygame.display.update()
        
        if x1 == foodx and y1 == foody:
            foodx = round(random.randrange(0, disp_width - snakeBlock) / snakeBlock) * snakeBlock
            foody = round(random.randrange(0, disp_height - snakeBlock) /snakeBlock) * snakeBlock
            snakeLength +=1
            score += 1

        clock.tick(snakeSpeed)

    pygame.quit()
    quit()


gameLoop()