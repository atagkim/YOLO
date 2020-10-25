import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

colors = ((252/256,210/256,113/256),
          (247/256,141/256,63/256),
          (43/256,187/256,216/256),
          (16/256,46/256,55/256),
          (232/256,237/256,224/256))
#cube용도
surfaces = ((0,1,2,3),
            (3,2,7,6),
            (6,7,5,4),
            (4,5,1,0),
            (1,5,7,2),
            (4,0,3,6))
#cube용도
vertices = ((1,-1,-1),(1,1,-1),
            (-1,1,-1),(-1,-1,-1),
            (1,-1,1),(1,1,1),
            (-1,-1,1),(-1,1,1))
#cube용도
edges = ((0,1),(0,3),(0,4),
         (2,1),(2,3),(2,7),
         (6,3),(6,4),(6,7),
         (5,1),(5,4),(5,7))

win=None
chk_end=False
chk_shape=None
chk_first=False

def drawCube():
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

def drawPyramid():
    glBegin(GL_TRIANGLES)
    #glEdgeFlagv(GL_TRUE)
    glColor3fv(colors[0])
    glVertex3f(0.0, 1.0, 0.0)
    glVertex3f(-1.0, -1.0, 1.0)
    glVertex3f(1.0, -1.0, 1.0)
    glColor3fv(colors[1])
    glVertex3f(0.0, 1.0, 0.0)
    glVertex3f(1.0, -1.0, 1.0)
    glVertex3f(1.0, -1.0, -1.0)
    glColor3fv(colors[2])
    glVertex3f(0.0, 1.0, 0.0)
    glVertex3f(1.0, -1.0, -1.0)
    glVertex3f(-1.0, -1.0, -1.0)
    glColor3fv(colors[3])
    glVertex3f(0.0, 1.0, 0.0)
    glVertex3f(-1.0, -1.0, -1.0)
    glVertex3f(-1.0, -1.0, 1.0)
    glEnd()

    glBegin(GL_QUADS)
    glColor3fv(colors[4])
    for vertex in surfaces[5]:
        glVertex3fv(vertices[vertex])
    glEnd()

def reset():
    pygame.init()
    display = (500, 500)
    global win
    win = pygame.display.set_mode(display)
    pygame.display.set_caption("Draw 3D")
    background = pygame.image.load("images/tmp.jpg")
    win.blit(background,(0,0))

def rotate():
    flag = False
    pitch = 0.0
    yaw = 0.0
    roll = 0.0
    k = pygame.key.get_pressed()

    if k[pygame.K_LEFT]:
        yaw = -0.5
        pitch = 0
        roll = 0
        flag = True
    elif k[pygame.K_RIGHT]:
        yaw = 0.5
        pitch = 0
        roll = 0
        flag = True
    elif k[pygame.K_UP]:
        pitch = -0.5
        yaw = 0
        roll = 0
        flag = True
    elif k[pygame.K_DOWN]:
        pitch = 0.5
        yaw = 0
        roll = 0
        flag = True
    elif k[pygame.K_q]:
        roll = -0.5
        yaw = 0
        pitch = 0
        flag = True
    elif k[pygame.K_w]:
        roll = 0.5
        yaw = 0
        pitch = 0
        flag = True

    if (flag):
        glRotatef(pitch, 1.0, 0, 0)
        glRotatef(yaw, 0, 1.0, 0)
        glRotatef(roll, 0, 0, 1.0)

def key_input():
    k = pygame.key.get_pressed()
    global chk_shape
    if k[pygame.K_1]:
        chk_shape = 'Cube'
    elif k[pygame.K_2]:
        chk_shape = 'Pyramid'

def setting():
    display = (500,500)
    global win
    win = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -10)

def myOpenGL():
    reset()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        key_input()
        global chk_first
        if chk_first==True:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            rotate()
        if chk_shape=='Cube':
            if chk_first==False:
                setting()
            chk_first=True
            drawCube()

        elif chk_shape=='Pyramid':
            if chk_first==False:
                setting()
            chk_first=True
            drawPyramid()

        k = pygame.key.get_pressed()
        if k[pygame.K_s]:
            chk_first=False
            pygame.image.save(win, 'images/3D.png')
            pygame.quit()
            break

        if k[pygame.K_ESCAPE]:
            pygame.quit()
            break

        pygame.display.flip()
        pygame.time.wait(10)

def main():
    myOpenGL()

if __name__ == "__main__":
    main()