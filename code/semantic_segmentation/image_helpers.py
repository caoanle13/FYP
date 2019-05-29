"""
    Module for useful image helper functions such as conversions.
"""

from PIL import Image
import numpy as np




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
    """ Helper function to merge two images together
    
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



def combine_masks(filenames):
    """
    function to combine different binary masks from the same image into one single binary mask
    
    - filenames: list of strings of the binarys masks to combine
    - output: combined mask

    """
    
    h = skimage.io.imread(os.path.join(MASK_DIR, filenames[0])).shape[0]
    w = skimage.io.imread(os.path.join(MASK_DIR, filenames[0])).shape[1]
    
    mask = np.zeros([h,w,3],dtype=np.uint8)

    for filename in filenames:

        temp = skimage.io.imread(os.path.join(MASK_DIR, filename))
        mask = mask | temp
    
    skimage.io.imsave(OUTPUT_DIR+"final_mask.jpg", mask)