import os

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC_PATH = os.path.join(APP_ROOT, "static/")
JS_PATH = os.path.join(STATIC_PATH, "js/")
IMAGE_PATH = os.path.join(STATIC_PATH, "images/")
MASK_PATH = os.path.join(STATIC_PATH, "masks/")
OUTPUT_PATH = os.path.join(STATIC_PATH, "output/")
CONTENT_MASK_PATH = os.path.join(MASK_PATH, "content/")
STYLE_MASK_PATH = os.path.join(MASK_PATH, "style/")
CONTENT_IMAGE_PATH = os.path.join(IMAGE_PATH, "content_image.jpg")
STYLE_IMAGE_PATH = os.path.join(IMAGE_PATH, "style_image.jpg")