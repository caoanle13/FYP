from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import shutil
from segmentation.models import SemanticModel, ThresholdModel, ColourModel
from segmentation.image_helpers import combine_masks
from paths import *
from structure import cleanup, add_summary_files, write_to_summary_file, produce_zip
from stylisation.style_transfer_model import TransferModel

app = Flask(__name__)



@app.route('/')
def index():
    cleanup()
    return render_template("index.html")

@app.route("/masks", methods=["POST"])
def masks():

    content_image = request.files["content_image"] 
    content_image.save(CONTENT_IMAGE_PATH)

    style_image = request.files["style_image"]
    style_image.save(STYLE_IMAGE_PATH)

    transfer_option = request.form["transfer-option"]

    write_to_summary_file("CONTENT: " + content_image.filename)
    write_to_summary_file("STYLE: " + style_image.filename) 
    write_to_summary_file("OPTION: " + transfer_option)
    add_summary_files([CONTENT_IMAGE_PATH, STYLE_IMAGE_PATH])

    if transfer_option == "full":
        return redirect(url_for('full_transfer'))

    elif transfer_option == "semantic":
        content_segmentor = SemanticModel(0)
        content_segmentor.segment(path=CONTENT_IMAGE_PATH)
        masks = [file for file in os.listdir(CONTENT_MASK_PATH) if file.startswith("mask_")]

        masks_path = [os.path.join(CONTENT_MASK_PATH, mask) for mask in masks]
        add_summary_files(masks_path)

        return render_template("semantic_masks.html", masks=masks)

    elif transfer_option == "threshold":
        n_threshold = int(request.form["n_threshold"])
        write_to_summary_file("N THRESHOLD: " + str(n_threshold))
        
        content_segmentor = ThresholdModel(0, n_threshold)
        content_segmentor.segment(path=CONTENT_IMAGE_PATH)
        style_segmentor = ThresholdModel(1, n_threshold)
        style_segmentor.segment(path=STYLE_IMAGE_PATH)

        masks_path = [
            os.path.join(CONTENT_MASK_PATH, "threshold_mask.jpg"),
            os.path.join(STYLE_MASK_PATH, "threshold_mask.jpg")
            ]

        add_summary_files(masks_path, ["content_threshold_mask.jpg", "style_threshold_mask.jpg"])
        

        return render_template("threshold_masks.html", n=n_threshold)


    elif transfer_option == "colour":

        n_colours = int(request.form["n_colours"])
        write_to_summary_file("N COLOUR: " + str(n_colours))

        base = int(request.form["base"])
        write_to_summary_file("BASE: " + "style" if base else "content")
        
        colour_segmentor = ColourModel(base=base, n_colours=n_colours)
        colour_segmentor.segment(target=0)
        colour_segmentor.segment(target=1)

        masks_path = [
            os.path.join(CONTENT_MASK_PATH, "colour_mask.jpg"),
            os.path.join(STYLE_MASK_PATH, "colour_mask.jpg")
            ]

        add_summary_files(masks_path, ["content_colour_mask.jpg", "style_colour_mask.jpg"])
        
        return render_template("colour_masks.html", n=n_colours)


@app.route("/style_transfer", methods =["POST"])
def style_transfer():
    transfer_option = request.form["type"]
    
    if transfer_option == "semantic":
        form = request.form.to_dict(flat=False)
        selected_masks = [x for x in list(form) if x != "type"]
        combine_masks(selected_masks, target=0)

        c_mask = os.path.join(CONTENT_MASK_PATH, "combined_mask.jpg")
        s_mask = None
        n_colors = 1

        write_to_summary_file("SELECTED REGIONS: " + str(selected_masks))
        add_summary_files([c_mask])
    
    elif transfer_option == "threshold":
        n_colors = int(request.form["n_colors"])
        c_mask = os.path.join(CONTENT_MASK_PATH, "threshold_mask.jpg")
        s_mask = os.path.join(STYLE_MASK_PATH, "threshold_mask.jpg")
    
    elif transfer_option == "colour":
        n_colors = int(request.form["n_colors"])
        c_mask = os.path.join(CONTENT_MASK_PATH, "colour_mask.jpg")
        s_mask = os.path.join(STYLE_MASK_PATH, "colour_mask.jpg")

    hard_width = False

    model = TransferModel(n_colors, hard_width, c_mask, s_mask)
    model.apply_transfer()

    add_summary_files([OUTPUT_IMAGE_PATH])
    
    return render_template("output.html", image="output.jpg")



@app.route("/full_transfer")
def full_transfer():
    c_mask = None
    s_mask = None
    model = TransferModel(1, False, c_mask, s_mask)
    model.apply_transfer()
    add_summary_files([OUTPUT_IMAGE_PATH])
    return render_template("output.html", image="output.jpg")



@app.route('/download_results')
def download_results():
    produce_zip(SUMMARY_PATH)

    return send_file(
        filename_or_fp="./static/summary.zip",
        mimetype='application/zip',
        as_attachment=True,
        attachment_filename='summary.zip'
    )



    
if __name__ == "__main__":
    app.run(host="localhost", port=8000, debug=False)
