#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import pandas as pd
import numpy as np

__author__  = ['Gianluca Carlini', 'Nico Curti']
__email__ = ['gianluca.carlini3@unibo.it', 'nico.curti2@unibo.it']

__all__ = ['imfill', 'LargestConnectedComponentsWithStats']

def imfill (img : np.ndarray) -> np.ndarray:
  '''
  Fill the holes of a single image.

  Parameters
  ----------
  img : array-like
   Image to fill

  Returns
  -------
  filled : array-like
   filled image
  '''
  # Copy the thresholded image.
  img = np.pad(img.astype('uint8'), pad_width=((2, 2), (2, 2)), mode='constant', constant_values=(0., 0.))
  im_floodfill = img.copy()
  # Floodfill from point (0, 0)
  cv2.floodFill(im_floodfill, mask=None, seedPoint=(0, 0), newVal=255)
  # Invert floodfilled image
  im_floodfill_inv = cv2.bitwise_not(im_floodfill)
  # Combine the two images to get the foreground.
  im_floodfill = img | im_floodfill_inv
  return im_floodfill[2:-2, 2:-2]

def LargestConnectedComponentsWithStats (src : np.ndarray) -> np.ndarray:
  '''
  Get the Largest connected component.

  Get the largest connected component in the binary image.
  The function extracts the full list of connected components using
  the connectedComponentsWithStats openCV function and then re-apply
  the input image (considered as mask) to filter out the background
  component. The resulting indexes are sorted and only the index related
  to the larger component is stored in the 'index' variable.

  Parameters
  ----------
  src : array-like
    Binary input image to analyze.

  Returns
  -------
  labels : array-like
    Destination labeled image with only the largest connected component highlighted.

  coords : tuple
    Coordinates of the largest connected component.
  '''

  _ , labels, stats, _ = cv2.connectedComponentsWithStats(src)
  labels = cv2.bitwise_and(labels, labels, mask=src)

  # get the unique values different from zero (background)
  index = np.unique(labels[labels != 0])
  stats = pd.DataFrame(stats, columns=['LEFT', 'TOP', 'WIDTH', 'HEIGHT', 'AREA'])
  stats = stats.iloc[index]
  stats.sort_values('AREA', ascending=False, inplace=True)

  labels = np.where(labels != stats.index[0], 0, 255).astype('uint8')

  left, top, w, h, _ = stats.iloc[0]

  return labels, (left, top, w, h)
  