"""
    Module for useful image helper functions such as conversions.
"""

from PIL import Image, ImageOps
import numpy as np
from paths import *
import os




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
    """Function to combine different binary masks from the same image into one single binary mask.
    
    Arguments:
        - filenames {list}: list of filenames.
        - target {int}: 0 for content, 1 for style.
    """
   
    MASK_DIR = STYLE_MASK_PATH if target else CONTENT_MASK_PATH
    
    h = skimage.io.imread(os.path.join(MASK_DIR, filenames[0])).shape[0]
    w = skimage.io.imread(os.path.join(MASK_DIR, filenames[0])).shape[1]
    
    mask = np.zeros([h,w,3],dtype=np.uint8)

    for filename in filenames:

        temp = skimage.io.imread(os.path.join(CONTENT_MASK_PATH, filename))
        mask = mask | temp
    
    SAVE_PATH = os.path.join(MASK_DIR, "combined_mask.jpg")
    skimage.io.imsave(SAVE_PATH, mask)