import pygame
import random
import numpy as np
import scipy as sp
import math as ma
from scipy.spatial.transform import Rotation as R


'''
Ray Tracing
'''


# Uses a vector and origin to see which item (from flist) is hit.
def rayNumbers(v, O, rst):
    # The order here is basically determined by wanting BxC to match the face vector &
    # wanting both vectors to start at t (although any point would do).
    B = rst[:, 1] - rst[:, 2]
    C = rst[:, 0] - rst[:, 2]
    l = O - rst[:, 2]
    U = np.stack((B, C, np.broadcast_to(-v, B.shape)), axis=1)
    # print(U)
    try:
        N = np.linalg.inv(U)
    except:
        print('Singular Matrix')
        N = np.broadcast_to(np.eye(3, 3), U.shape)
    u = np.matmul(l[:, None, :],N)[:,0, :]  # Not sure why matmul has to go l,N instead of N,l
    return u


def rayNumbersV(V, O, rst):
    indexes = []
    for v in V:
        #print(v)
        rN = rayNumbers(v, O, rst)
        rNBool = np.any(np.vstack((0 > rN[:, 0], 0 > rN[:, 1], 1 < rN[:, 0] + rN[:, 1], 0 >= rN[:, 2])), axis=0)
        rN[rNBool, 2] = np.inf
        indexes.append(np.argmin(rN[:, 2]))
    return indexes


'''
Conversion Functions
'''


def outAround2Vector(out, around):
    '''
    Takes 0,0,1 rotates it round the x axis (down) by the out angle and then rotates it around the z axis by the around angle
    only works with one pair of angles at the momen
    '''
    return R.from_euler('xz', [out, around],degrees=True).apply(np.array([0, 0, 1]))


def outAroundInv(v):
    '''
    Inverts outAround
    '''
    if len(v.shape) == 1:
        return np.arccos(v[2]) * 180 / np.pi, (np.arctan2(v[1], v[0]) * 180 / np.pi) + 90
    elif len(v.shape) == 2:
        return np.arccos(v[:, 2]) * 180 / np.pi, (np.arctan2(v[:, 1], v[:, 0]) * 180 / np.pi) + 90
    else:
        raise Exception("This function can only handle lists of vectors")


def outAround2Screen(s, oA):
    SCREEN_DIMENSION = s.get_size()
    sCx = SCREEN_DIMENSION[0] / 2
    sCy = SCREEN_DIMENSION[1] / 2
    sRadius = min(sCx, sCy)
    oRadius = oA[0] * sRadius / 180
    x = oRadius * ma.sin(ma.radians(oA[1]))
    y = oRadius * ma.cos(ma.radians(oA[1]))
    return sCx + x, sCy + y


def persp2eyeVec(coords):
    '''
    Perspective should refer to the normalized version of what goes onto the screen. Screen should refer to what gets put onto the screen
    Takes perspective coordintes and returns the eye vector corresponding to that perspective coordinate
    '''
    # print(coords)
    coords = np.reshape(coords, (int(coords.size / 2), 2))
    outDist = np.linalg.norm(coords, axis=1)
    outAngle = np.arcsin(outDist) * 2
    aroundAngle = np.arctan2(coords[:, 1], coords[:, 0]) + np.pi / 2
    # print(coords.T)
    # print(outDist)
    # next we need to convert the
    OACoords = np.array([outAngle, aroundAngle]).T
    # print(np.degrees(OACoords).T)
    vector = R.from_euler('xz', OACoords,
                          degrees=False).apply(np.array([0, 0, 1]))
    return vector


'''
Drawing Functions
'''


def draw_dot(s, c, l, radius=1):
    if isinstance(c, int):
        c = (c, c, c)
    pygame.draw.circle(s, c, l, radius)


def draw_pixel(s, c, l, size=1):
    if isinstance(c, int):
        c = (c, c, c)
    pygame.draw.rect(s, c, pygame.Rect(l, (size, size)))


'''
Object Creation Functions
'''
def pixelCircle(r, centers=False, normalized=False):
    '''
    Returns pixels a distance r from center is at coordiate point 0,0 and pixels are assumed to be 1 unit wide.
    The closest pixel to the center has the center (0.5,0.5) It is assumed that this level of detail will matter some day.
    Not designed for efficincy 
    @param r: the radius of the circle in pixels
    @param centers=False: if the returned pixels should use the center of the pixels (0.5 adjustment) or should return the bottom left corner.
    @param normalized=False: if the returned coordinates should be divided by r. This will always return the the center of the pixels regardless of what center is set to. 
    '''
    r2 = r * r
    pixels = np.array([[r * ma.cos(ma.pi / 4) - 0.5, r * ma.sin(ma.pi / 4) - 0.5]],
        dtype=int)
    y = pixels[0, 1]
    x = pixels[0, 0]
    cenStack = np.stack((np.arange(0, x), np.arange(0, y))).T
    # print(x,y)
    while (y >= 0 and x < r):
        # print(x+1,y,ma.pow(x+1+0.5,2)+ma.pow(y+0.5,2))
        if ma.pow(x + 1 + 0.5, 2) + ma.pow(y + 0.5, 2) < r2:
            x += 1
        else:
            y -= 1
        newStack = np.stack((np.arange(x - y, x + 1), np.arange(0, y + 1))).T
        pixels = np.concatenate((pixels, newStack), axis=0)
    # print(np.flip(pixels[1:,:],axis=1))
    pixels = np.concatenate((pixels, np.flip(pixels[1:, :], axis=1)), axis=0)
    pixels = np.concatenate((pixels, cenStack), axis=0)
    # rotating the quarter circle
    pixels = pixels + 0.5
    pixelX = np.multiply(pixels,np.broadcast_to(np.array([-1, 1]), (pixels.shape)))
    pixels = np.concatenate((pixels, pixelX), axis=0)
    pixelY = np.multiply(pixels,np.broadcast_to(np.array([1, -1]), (pixels.shape)))
    pixels = np.concatenate((pixels, pixelY), axis=0)
    if centers or normalized:
        if normalized:
            pixels = pixels / r
        return pixels
    else:
        pixels = pixels - 0.5
        return (pixels.astype(int))

'''
Helps ensure that the lists of face and color coordinates remain synched. Adds new face and color coordinates to lists of face and color coordinates, then returns updated lists of face and color coordinates.  
Not designed to be efficient
'''
def newFace(f, c, fL, cL):
    fL = np.concatenate((fL, f), axis=0)
    cL = np.concatenate((cL, c), axis=0)
    return fL, cL

'''
Creates an octahedron with a specified size, rotation, and color distribution. 
Not designed to be efficient
@param center: an arraylike with 3 elements specifiying the center of the octahedron
@param r: a number specifying the radius of the octahedron
@param rot: A scipy.Rotation object, specifying the rotation of the octahedron
@param lowColor: the darkest color value (0-9) to be used in coloring the octahedron
@param highColor: the brightest color value (0-9) to be used in coloring the octahedron 
@return sidesR: a numpy array of shape (8,3,3). A list of 8 faces, each consisting of 3 coordinates, each consisting of 3 values.
@return colors: a numpy array of shape (8,3). A list of 8 colors, each consisting of 3 color values (0-9). 
'''
def octahedron(center, r, rot, lowColor=3, highColor=7):
    #center=np.array([0,0,0])
    #r=3
    #rot=R.from_rotvec(np.array([.1, .1, .1]))
    offset = np.transpose((np.stack((np.eye(3), np.eye(3) * -1),
                                    axis=2)).reshape(3, 6))
    corners = r * offset + center
    #print(offset)
    sideSel = np.array([[1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 0, 1], [1, 0, 0, 1, 1, 0],[0, 1, 1, 0, 1, 0], [0, 1, 1, 0, 0, 1], [1, 0, 0, 1, 0, 1],[0, 1, 0, 1, 1, 0], [0, 1, 0, 1, 0, 1]],dtype=bool)
    sides = np.zeros((1, 3, 3))
    for i in sideSel:
        sides = np.append(sides, offset[None, i], axis=0)
    sides = sides[1:]
    #print(sides)
    #print(rot.apply(sides.reshape(24,3)).reshape(8,3,3))
    sidesR = rot.apply(sides.reshape(24, 3)).reshape(8, 3, 3) * r + center
    lC, hC = lowColor, highColor  #low and high color
    colors = (np.sum(sides, axis=2) + 1) * (hC - lC) / 2 + lC
    #print(colors.shape)
    return sidesR, colors

'''
Creates a small arrow pointing to a location with a color, orientation, and scale
Not designed to be efficient
@param loc: an arraylike with 3 elements specifiying the location of the tickmark
@param color: a color for the tickmark
@param orientation=0: the orientation for the tickmark
@param scale=1: the scale of the tickmark
@return 
'''
def tickMark(loc, color, orientation=0, scale=1):
    side = np.array([[0, 0, 0], [scale, scale/2, 0], [scale/2, scale, 0]])
    if orientation != 0:
        side = orientation.apply(side)
    if isinstance(color,int):
      color = np.array([color,color,color])
    return side, color