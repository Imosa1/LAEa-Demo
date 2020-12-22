import pygame
import random
import numpy as np
import scipy as sp
from scipy.spatial.transform import Rotation as R
from pygame.locals import (RLEACCEL, K_ESCAPE, KEYDOWN, QUIT)
from pygame.locals import (K_UP, K_DOWN, K_LEFT, K_RIGHT)
from pygame.locals import (K_q, K_w, K_e, K_a, K_s, K_d, K_z, K_x, K_c)
from pygame.locals import (K_u, K_i, K_o, K_j, K_k, K_l, K_COMMA)
import math as ma
import sys

from SpaceFunc import *


# Define a Player object by extending pygame.sprite.Sprite
# The surface drawn on the screen is now an attribute of 'player'
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super(Player, self).__init__()
        self.location = np.array([0, 0, 0])
        self.orientation = R.from_rotvec(-2 / 3 * np.pi * normV(np.array([1, 1, 1])))

    # Move the sprite based on user keypresses
    def update(self, pressed_keys):
        if pressed_keys[K_w]:
            self.location = self.location + self.orientation.apply(np.array([0, 0, 1]),True)
        if pressed_keys[K_s]:
            self.location = self.location - self.orientation.apply(np.array([0, 0, 1]),True)
        if pressed_keys[K_a]:
            self.location = self.location - self.orientation.apply(np.array([1, 0, 0]),True)
        if pressed_keys[K_d]:
            self.location = self.location + self.orientation.apply(np.array([1, 0, 0]),True)
        if pressed_keys[K_q]:
            self.location = self.location + self.orientation.apply(np.array([0, 1, 0]),True)
        if pressed_keys[K_e]:
            self.location = self.location - self.orientation.apply(np.array([0, 1, 0]),True)
        if pressed_keys[K_z]:
            self.location -= np.array([1, 0, 0])
        if pressed_keys[K_x]:
            self.location = np.array([0, 0, 0])
        if pressed_keys[K_c]:
            self.location += np.array([1, 0, 0])

        if pressed_keys[K_i]:
            self.orientation = R.from_euler('x', 10, degrees=True) * self.orientation
        if pressed_keys[K_k]:
            self.orientation = R.from_euler('x', -10, degrees=True) * self.orientation
        if pressed_keys[K_j]:
            self.orientation = R.from_euler('y', 10, degrees=True) * self.orientation
        if pressed_keys[K_l]:
            self.orientation = R.from_euler('y', -10, degrees=True) * self.orientation
        if pressed_keys[K_u]:
            self.orientation = R.from_euler('z', 10, degrees=True) * self.orientation
        if pressed_keys[K_o]:
            self.orientation = R.from_euler('z', -10, degrees=True) * self.orientation
        if pressed_keys[K_COMMA]:
            self.orientation = R.from_rotvec(-2 / 3 * np.pi * normV(np.array([1, 1, 1])))


if __name__ == "__main__":
    print("Hello World!")
    pygame.init()
    print(pygame)

    # completed number munching
    # setting up position
    O = np.array([0, 0, 0])
    # setting up some things to look at
    fList, cList = octahedron(np.array([0, 0, 0]), 30,R.from_rotvec(np.array([2 * np.pi, 0, 0])))
    nf, nc = octahedron(np.array([6, 7, 8]),5,R.from_rotvec(np.array([10, 15, 5])),highColor=8)
    fList, cList = newFace(nf, nc, fList, cList)
    fList, cList = newFace(np.array([[[8, 0, 0], [8, 1, 0], [8, 0, 1]]]),np.array([[9, 0, 0]]), fList, cList)
    fList, cList = newFace(np.array([[[0, 8, 0], [0, 8, 1], [1, 8, 0]]]),np.array([[0, 9, 0]]), fList, cList)
    fList, cList = newFace(np.array([[[0, 0, 8], [1, 0, 8], [0, 1, 8]]]),np.array([[0, 0, 9]]), fList, cList)
    fList, cList = newFace(np.array([[[-8, 0, 0], [-8, 1, 0], [-8, 0, 1]]]),np.array([[0, 9, 9]]), fList, cList)
    fList, cList = newFace(np.array([[[0, -8, 0], [0, -8, 1], [1, -8, 0]]]),np.array([[9, 0, 9]]), fList, cList)
    fList, cList = newFace(np.array([[[0, 0, -8], [1, 0, -8], [0, 1, -8]]]),np.array([[9, 9, 0]]), fList, cList)
    fList, cList = newFace(np.array([[[16, 0, 0], [16, 1, 0], [16, 0, 1]]]),np.array([[9, 0, 0]]), fList, cList)
    fList, cList = newFace(np.array([[[0, 16, 0], [0, 16, 1], [1, 16, 0]]]),np.array([[0, 9, 0]]), fList, cList)
    fList, cList = newFace(np.array([[[0, 0, 16], [1, 0, 16], [0, 1, 16]]]),np.array([[0, 0, 9]]), fList, cList)
    cList = cList / 9 * 255
    nFaces = fList.shape[0]
    # making some sight vectors
    pixelRadius = 28
    pixelCoords = pixelCircle(pixelRadius, False)
    perspCoords = pixelCircle(pixelRadius, True) / pixelRadius
    sightVectors = persp2eyeVec(perspCoords)
    # ray tracing
    print(perspCoords.shape)
    print(sightVectors.shape)


    clock = pygame.time.Clock()
    SCREEN_DIMENSION = (1000, 650)
    sCx = int(SCREEN_DIMENSION[0] * 2 / 3)
    sCy = int(SCREEN_DIMENSION[1] / 2)
    sRadius = min(sCx, sCy)
    screenCoords = np.multiply(perspCoords * sRadius,np.broadcast_to(np.array([1, -1]), (perspCoords.shape))) + np.array([sCx, sCy]).astype(int)

    fontCrime = pygame.font.SysFont(None, 24) # Need to test this module in a smaller environemnt.
    screen = pygame.display.set_mode(SCREEN_DIMENSION)

    # Draw the reticle
    crosshair = pygame.Surface((sRadius*2,sRadius*2))
    reticleC = (180, 180, 180) 
    pygame.draw.circle(crosshair, reticleC, (sRadius, sRadius), sRadius, width=2)
    pygame.draw.circle(crosshair, reticleC, (sRadius, sRadius), sRadius * 0.707, width=2)
    pygame.draw.line(crosshair, reticleC, (0, sRadius), (sRadius*2, sRadius), width=2)
    pygame.draw.line(crosshair, reticleC, (sRadius, 0), (sRadius, sRadius*2), width=2)
    crosshair.set_colorkey((0, 0, 0), RLEACCEL)

    # Perspective Surface
    perspSurf = pygame.Surface((pixelRadius*2,pixelRadius*2))
    perspSurf.set_colorkey((0, 0, 0), RLEACCEL)

    player = Player()

    # Run until the user asks to quit
    running = True
    while running:
        # Did the user click the window close button?
        for event in pygame.event.get():
            # Did the user hit a key?
            if event.type == KEYDOWN:
                # Was it the Escape key? If so, stop the loop.
                if event.key == K_ESCAPE:
                    running = False
            # Check for QUIT event. If QUIT, then set running to false.
            elif event.type == pygame.QUIT:
                running = False

        # Get all the keys currently pressed
        pressed_keys = pygame.key.get_pressed()
        # Update the player based on user keypresses
        player.update(pressed_keys)

        fListFlat = fList.reshape(nFaces *3, 3)
        fListP = player.orientation.apply(fListFlat-player.location).reshape(nFaces, 3, 3)

        # Fill the background with white
        # screen.fill((255, 255, 255))

        # sightVectors = R.random(500).apply(pzh)
        nearIndexs = rayNumbersV(sightVectors, (0,0,0), fListP)
        nearColors = cList[nearIndexs]
        # outArnd=np.transpose(np.array(outAroundInv(sightVectors)))

        for i in range(0, screenCoords.shape[0]):
            # draw_pixel(screen, nearColors[i], screenCoords[i], 4)
            perspSurf.set_at(pixelCoords[i]+np.array([pixelRadius,pixelRadius]),nearColors[i])

        perspSurfBig = pygame.transform.flip(pygame.transform.scale(perspSurf, (int(sRadius*2), int(sRadius*2))),False,True)
        screen.blit(perspSurfBig, (sCx-sRadius,sCy-sRadius))
        screen.blit(crosshair, (sCx-sRadius,sCy-sRadius))

        if pygame.font.get_init():
            locLabel = "Loc: [%.2f,%.2f,%.2f]"%(player.location[0],player.location[1],player.location[2])
            ori = player.orientation.as_euler('zyx', degrees=True)
            oriLabel = "Ori: [%.1f,%.1f,%.1f]"%(ori[0],ori[1],ori[2])
            screen.blit(fontCrime.render(locLabel,False,(255,255,255),(50,50,50)), (0,0))
            screen.blit(fontCrime.render(oriLabel,False,(255,255,255),(50,50,50)), (0,16))

        # Flip the display
        pygame.display.flip()

        clock.tick(5)

    # Done! Time to quit.
    pygame.display.quit()
