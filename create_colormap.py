#!/usr/bin/env python
# -*- coding: utf-8 -*

import numpy as np
import sys
import cv2

def lut():
    colormap = [(0,0,0),(0.5,0,0),(0,0.5,0),(0.5,0.5,0),(0,0,0.5),(0.5,0,0.5),(0,0.5,0.5), 
            (0.5,0.5,0.5),(0.25,0,0),(0.75,0,0),(0.25,0.5,0),(0.75,0.5,0),(0.25,0,0.5), 
            (0.75,0,0.5),(0.25,0.5,0.5),(0.75,0.5,0.5),(0,0.25,0),(0.5,0.25,0),(0,0.75,0), 
            (0.5,0.75,0),(0,0.25,0.5)] + [(0,0,0)] * 235
    colormap = [colormap]
    colormap = np.array(colormap) * 255
    colormap = colormap.astype(np.uint8)

    return colormap

if __name__=="__main__":

    im_color = lut()
    cv2.imwrite('deeplab_colormap.png', im_color)

