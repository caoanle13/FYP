"""
    Module for useful image helper functions such as conversions.
"""

from PIL import Image, ImageOps
import numpy as np
from paths import *
import os
from skimage import io
import webcolors
import cv2




def array_to_pil(arr):
    """ Helper function to convert numpy arrays into PIL images.
    
    Arguments:
        -arr {nd array}: input numpy array.
    Returns:
        {PIL Image}: output PIL image.
    """

    image = Image.fromarray(arr.astype('uint8'))
    return image



def boolean_to_pil(data):
    """ Helper function to convert a boolean np array to a PIL image.
    
    Arguments:
        data {nd array} -- Boolean numpy array (only contains True or False).
    Returns:
        {PIL Image} -- output PIL Image.
    """
    size = data.shape[::-1]
    databytes = np.packbits(data, axis=1)
    image = Image.frombytes(mode='1', size=size, data=databytes)
    image = image.convert(mode='RGB')
    return image



def superimpose(background, overlay):
    """ Helper function to merge two images together.
    
    Arguments:
        background {PIL Image}: first image.
        overlay {PIL Image}: second image. 
    Returns:
        {PIL Image}: output image.
    """
    background = background.convert("RGBA")
    overlay = overlay.convert("RGBA")
    merged_image = Image.blend(background, overlay, alpha=0.5).convert("RGB")
    return merged_image


def equalize(image):
    """ Helper function to perform histogram equalization.
    
    Arguments:
        - image {PIL Image}: Input image to be equalized.
    Returns:
        {PIL Image}: Output equalized image.
    """
    equalized_image = ImageOps.equalize(image)
    return equalized_image



def combine_masks(filenames, target):
    """ Function to combine different single coloured mask into one multiple coloured mask
    
    Arguments:
        filenames {list} -- List of numpy file names.
        target {int} -- 0 for content, 1 for style.
    """

    MASK_DIR = STYLE_MASK_PATH if target else CONTENT_MASK_PATH
    SAVE_NAME = ("style" if target else "content") + "_combined_mask.jpg"

    h = np.load(os.path.join(MASK_DIR, filenames[0])).shape[0]
    w = np.load(os.path.join(MASK_DIR, filenames[0])).shape[1]
    
    mask = np.zeros([h,w],dtype=bool)

    for filename in filenames:
        path = os.path.join(MASK_DIR, filename)
        temp = np.load(path)
        # Coloured case
        if len(temp.shape) == 3:
            mask += (temp != 0)[:,:,0]
        # Boolean case
        else:
            mask += temp
    
    mask = boolean_to_pil(mask)
    SAVE_PATH = os.path.join(MASK_DIR, SAVE_NAME)
    mask.save(SAVE_PATH)


def closest_colour(rgb):
    min_colours = {}
    for key, name in webcolors.css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - rgb[0]) ** 2
        gd = (g_c - rgb[1]) ** 2
        bd = (b_c - rgb[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]


def get_colour_name(requested_colour):
    """ Function that returns a colour name given an rgb value.
    
    Arguments:
        requested_colour {iterator} -- RGB value of the colour.
    
    Returns:
        tuple -- The name of the identified colour.
    """
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name