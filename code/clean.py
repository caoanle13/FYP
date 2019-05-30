import os
from shutil import rmtree
from paths import *


def cleanup():
    
    if os.path.isdir(IMAGE_PATH):
       rmtree(IMAGE_PATH)
    os.mkdir(IMAGE_PATH)

    if os.path.isdir(MASK_PATH):
       rmtree(MASK_PATH)
    os.mkdir(MASK_PATH)

    if os.path.isdir(OUTPUT_PATH):
       rmtree(OUTPUT_PATH)
    os.mkdir(OUTPUT_PATH)

    os.mkdir(CONTENT_MASK_PATH)
    os.mkdir(STYLE_MASK_PATH)