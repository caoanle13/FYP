from flask import Flask, render_template, request
import os
import shutil
#from src.mask import detect_objects, combine_masks
from semantic_segmentation.segmentation import SegmentationModel
from src.style_transfer import transfer_style

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(APP_ROOT, "static/images/")
MASK_PATH = os.path.join(APP_ROOT, "static/masks/")

seg_model = SegmentationModel()

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/masks", methods=["POST"])
def masks():

    if os.path.isdir(IMAGE_PATH):
        shutil.rmtree(IMAGE_PATH)
    os.mkdir(IMAGE_PATH)

    if os.path.isdir(MASK_PATH):
        shutil.rmtree(MASK_PATH)
    os.mkdir(MASK_PATH)
    
    content_image = request.files.getlist("content_image")[0]
    content_image.save("/".join([IMAGE_PATH, "content_image.jpg"]))

    style_image = request.files.getlist("style_image")[0]    
    style_image.save("/".join([IMAGE_PATH, "style_image.jpg"]))

    #detect_objects("content_image.jpg")
    seg_model.infer("static/images/content_image.jpg")

    return "hi"

    #mask_files = [file for file in os.listdir('static/masks/') if not file.startswith(".")]

    #return render_template("masks.html", image="maskrcnn.jpg", masks=mask_files)


@app.route("/style", methods=["POST"])
def style():
    form = request.form.to_dict(flat=False)
    selected = list(form)
    combine_masks(selected)

    transfer_style("content_image.jpg", "final_mask.jpg", "style_image.jpg")

    return render_template("style.html", image="output.png")



    
if __name__ == "__main__":
    app.run(host="localhost", port=8000, debug=False)
