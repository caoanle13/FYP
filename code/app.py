from flask import Flask, render_template, request, redirect, url_for
import os
import shutil
from segmentation.models import SemanticModel, ThresholdModel
from segmentation.image_helpers import combine_masks
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
        return redirect(url_for('full_transfer'))

    elif transfer_option == "semantic":
        content_segmentor = SemanticModel(0)
        content_segmentor.segment(path=CONTENT_IMAGE_PATH)
        masks = [file for file in os.listdir(CONTENT_MASK_PATH) if file.startswith("mask_")]
        return render_template("semantic_masks.html", masks=masks)

    elif transfer_option == "threshold":
        content_segmentor = ThresholdModel(0)
        content_segmentor.segment(path=CONTENT_IMAGE_PATH)
        style_segmentor = ThresholdModel(1)
        style_segmentor.segment(path=STYLE_IMAGE_PATH)

        return render_template("threshold_masks.html")


@app.route("/full_transfer")
def full_transfer():
    c_mask = None
    s_mask = None
    model = TransferModel(1, False, c_mask, s_mask)
    model.apply_transfer()
    return render_template("output.html", image="output.jpg")


@app.route("/semantic_transfer", methods=["POST"])
def semantic_transfer():
    form = request.form.to_dict(flat=False)
    selected_masks = list(form)
    combine_masks(selected_masks, target=0)

    c_mask = os.path.join(CONTENT_MASK_PATH, "combined_mask.jpg")
    s_mask = None
    model = TransferModel(2, False, c_mask, s_mask)
    model.apply_transfer()
    return render_template("output.html", image="output.jpg")

    transfer_style("content_image.jpg", "final_mask.jpg", "style_image.jpg")

    return render_template("output.html", image="output.jpg")



@app.route("/treshold_transfer", methods=["POST"])
def threshold_transfer():
    c_mask = os.path.join(CONTENT_MASK_PATH, "threshold_mask.jpg")
    s_mask = os.path.join(STYLE_MASK_PATH, "threshold_mask.jpg")
    model = TransferModel(2, False, c_mask, s_mask)
    model.apply_transfer()
    return render_template("output.html", image="output.jpg")




    
if __name__ == "__main__":
    cleanup()
    app.run(host="localhost", port=8000, debug=False)
