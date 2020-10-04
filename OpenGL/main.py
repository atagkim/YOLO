import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

#필요없음
colors = ((1,0,0),
          (0,1,0),
          (0,0,1),
          (0,0,0))

#필요없음
surfaces = ((0,1,2,3),
            (3,2,7,6),
            (6,7,5,4),
            (4,5,1,0),
            (1,5,7,2),
            (4,0,3,6))

vertices = ((1,-1,-1),(1,1,-1),
            (-1,1,-1),(-1,-1,-1),
            (1,-1,1),(1,1,1),
            (-1,-1,1),(-1,1,1))

edges = ((0,1),(0,3),(0,4),
         (2,1),(2,3),(2,7),
         (6,3),(6,4),(6,7),
         (5,1),(5,4),(5,7))

def drawCube():
    #필요없음
    glBegin(GL_QUADS)
    for surface in surfaces:
        x=0
        for vertex in surface:
            glColor3fv(colors[x])
            glVertex3fv(vertices[vertex])
            x+=1
    glEnd()

    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()

def myOpenGL():
    ###
    test = True
    pitch = 0.0
    yaw = 0.0
    roll = 0.0
    val = -10
    ###

    pygame.init()
    display = (800,600)
    win = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    gluPerspective(45, (display[0]/display[1]),0.1,50.0)
    glTranslatef(0.0, 0.0, val)
    #값 수정

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        ###
        k = pygame.key.get_pressed()

        if k[pygame.K_LEFT]:
            yaw = -0.5
            pitch = 0
            roll = 0
            test = True
        elif k[pygame.K_RIGHT]:
            yaw = 0.5
            pitch = 0
            roll = 0
            test = True
        elif k[pygame.K_UP]:
            pitch = -0.5
            yaw = 0
            roll = 0
            test = True
        elif k[pygame.K_DOWN]:
            pitch = 0.5
            yaw = 0
            roll = 0
            test = True
        elif k[pygame.K_q]:
            roll = -0.5
            yaw = 0
            pitch = 0
            test = True
        elif k[pygame.K_w]:
            roll = 0.5
            yaw = 0
            pitch = 0
            test = True
        ###

        if(test):
            glRotatef(pitch,1.0,0,0)
            glRotatef(yaw, 0, 1.0, 0)
            glRotatef(roll, 0, 0, 1.0)
            test = False
        #원래는 1,3,1,1
        ###

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        drawCube()

        ###
        if k[pygame.K_a]:
            pygame.image.save(win,'test.png')

        pygame.display.flip()
        pygame.time.wait(10)

myOpenGL()