from flask import Flask, render_template, request
import os
import shutil
from segmentation.models import SemanticModel, ThresholdModel
#from src.style_transfer import transfer_style
from paths import *
from clean import cleanup
from stylisation.style_transfer_model import TransferModel

app = Flask(__name__)



@app.route('/')
def index():
    return render_template("index.html")

@app.route("/masks", methods=["POST"])
def masks():

    content_image = request.files["content_image"]        
    content_image.save(CONTENT_IMAGE_PATH)

    style_image = request.files["style_image"]
    style_image.save(STYLE_IMAGE_PATH)

    transfer_option = request.form["transfer-option"]
    if transfer_option == "full":
        return render_template("no_masks.html")

    elif transfer_option == "semantic":
        content_model = SemanticModel(0)
        content_model.segment(path=CONTENT_IMAGE_PATH)
        masks = [file for file in os.listdir(CONTENT_MASK_PATH) if file.startswith("mask_")]
        return render_template("semantic_masks.html", masks=masks)

    elif transfer_option == "threshold":
        content_model = ThresholdModel(0)
        content_model.segment(path=CONTENT_IMAGE_PATH)
        style_model = ThresholdModel(1)
        style_model.segment(path=STYLE_IMAGE_PATH)

        return render_template("threshold_masks.html")



@app.route("/semantic_transfer", methods=["POST"])
def style():
    form = request.form.to_dict(flat=False)
    selected = list(form)
    combine_masks(selected)

    transfer_style("content_image.jpg", "final_mask.jpg", "style_image.jpg")

    return render_template("style.html", image="output.jpg")



@app.route("/treshold_transfer", methods=["POST"])
def treshold_transfer():
    c_mask = os.path.join(CONTENT_MASK_PATH, "threshold_mask.jpg")
    s_mask = os.path.join(STYLE_MASK_PATH, "threshold_mask.jpg")
    model = TransferModel(2, False, c_mask, s_mask)
    model.apply_transfer()
    return render_template("output.html", image="output.jpg")



    
if __name__ == "__main__":
    cleanup()
    app.run(host="localhost", port=8000, debug=False)
