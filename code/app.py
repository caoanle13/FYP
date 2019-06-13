from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import shutil
from PIL import Image
from segmentation.models import SemanticModel, ThresholdModel, ColourModel
from segmentation.image_helpers import combine_masks
from paths import *
from structure import cleanup, log_files, log_text, produce_zip
from stylisation.style_transfer_model import TransferModel
from constants import numbers
from preprocessing.preprocessing import match_histogram


app = Flask(__name__)


# Index Route
@app.route('/')
def index():
    cleanup() # Performs the necessary clean up of project directory tree.
    return render_template("index.html")




# Masks Route
@app.route("/masks", methods=["POST"])
def masks():

    # Get form parameters
    content_image = request.files["content_image"] 
    content_image.save(CONTENT_IMAGE_PATH)

    style_image = request.files["style_image"]
    style_image.save(STYLE_IMAGE_PATH)

    transfer_option = request.form["transfer-option"]
    user_defined_regions = request.form.get('region_toggle') == "on"
    preserve_content_palette = request.form.get('preserve_content_palette') == "on"

    # Log form input to summary directory
    log_text("CONTENT: " + content_image.filename)
    log_text("STYLE: " + style_image.filename) 
    log_text("PRESERVE CONTENT PALETTE: " + str(preserve_content_palette))
    log_text("OPTION: " + transfer_option)
    log_text("USER DEFINED REGION: " + str(user_defined_regions))
    log_files([CONTENT_IMAGE_PATH, STYLE_IMAGE_PATH])

    if preserve_content_palette:
        matched_image = match_histogram(STYLE_IMAGE_PATH, CONTENT_IMAGE_PATH)
        matched_image.save(STYLE_IMAGE_PATH)
        log_files([STYLE_IMAGE_PATH], ["matched_style_image.jpg"])
        

    # Full style transfer
    if transfer_option == "full":
        return redirect(url_for('full_transfer'))

    # Semantic segmentation
    elif transfer_option == "semantic":
        content_segmentor = SemanticModel(0)
        content_segmentor.segment(path=CONTENT_IMAGE_PATH)
        masks = [file for file in os.listdir(CONTENT_MASK_PATH) if file.startswith("mask_")]

        masks_path = [os.path.join(CONTENT_MASK_PATH, mask) for mask in masks]
        log_files(masks_path)

        return render_template("semantic_masks.html", masks=masks)


    # Threshold segmentation
    elif transfer_option == "threshold":

        # Get number of thresholds and log it to summary file
        n_threshold = int(request.form["n_threshold"])
        log_text("N THRESHOLD: " + str(n_threshold))

        # Paths to segmented content and style images
        masks_path = [
            os.path.join(CONTENT_MASK_PATH, "content_threshold_mask.jpg"),
            os.path.join(STYLE_MASK_PATH, "style_threshold_mask.jpg")
        ]
        
        # Perform threhsold segmentation on content and style images
        content_segmentor = ThresholdModel(
            target=0,
            n_threshold=n_threshold,
            user_defined=user_defined_regions
            )

        style_segmentor = ThresholdModel(
            target=1,
            n_threshold=n_threshold,
            user_defined=user_defined_regions
            )

        content_segmentor.segment(path=CONTENT_IMAGE_PATH)
        style_segmentor.segment(path=STYLE_IMAGE_PATH)

        # Automatic region pairing
        if not user_defined_regions:

            # Add files to summary directory
            log_files(masks_path)
        
            return render_template(
                "threshold_masks.html",
                n=n_threshold,
                user_defined=str(user_defined_regions)
                )

        # User-defined regions
        else:

            # Get paths for all the individual masks
            content_masks = [file for file in os.listdir(CONTENT_MASK_PATH) if file.startswith("content_mask_")]
            style_masks = [file for file in os.listdir(STYLE_MASK_PATH) if file.startswith("style_mask_")]
            content_masks_path = [os.path.join(CONTENT_MASK_PATH, mask) for mask in content_masks]
            style_masks_path = [os.path.join(STYLE_MASK_PATH, mask) for mask in style_masks]
            masks_path += (content_masks_path + style_masks_path)

            # Add files to summary directory
            log_files(masks_path)

            return render_template(
                "threshold_masks.html",
                n=n_threshold,
                user_defined=str(user_defined_regions),
                content_masks=sorted(content_masks),
                style_masks=sorted(style_masks),
                n_masks=numbers[n_threshold+1]
                )


    # Colour segmentation
    elif transfer_option == "colour":

        # Get number of colours and log it to summary file
        n_colours = int(request.form["n_colours"])
        log_text("N COLOUR: " + str(n_colours))

        masks_path = [
                os.path.join(CONTENT_MASK_PATH, "content_colour_mask.jpg"),
                os.path.join(STYLE_MASK_PATH, "style_colour_mask.jpg")
                ]

        # Automatic region pairing
        if not user_defined_regions:

            # Log chosen base image to summary file
            base = int(request.form["base"])
            log_text("BASE: " + ("style" if base else "content"))

            # Create single segmentor with given base and segment content and style images
            segmentor = ColourModel(
                base=base,
                n_colours=n_colours,
                user_defined=user_defined_regions
            )
            
            segmentor.segment(target=0)
            segmentor.segment(target=1)

            # Log segmented images to summary directory
            log_files(masks_path)
        
            return render_template(
                "colour_masks.html",
                n=n_colours,
                user_defined=str(user_defined_regions)
                )

        # User-defined regions
        else:
            
            # Create 2 distinct segmentors and segment content and style images
            content_segmentor = ColourModel(
                base=0,
                n_colours=n_colours,
                user_defined=user_defined_regions
                )

            content_segmentor.segment(target=0)

            style_segmentor = ColourModel(
                base=1,
                n_colours=n_colours,
                user_defined=user_defined_regions
            )
            
            style_segmentor.segment(target=1)

            # Get paths for all the individual masks
            content_masks = [file for file in os.listdir(CONTENT_MASK_PATH) if file.startswith("content_mask_")]
            style_masks = [file for file in os.listdir(STYLE_MASK_PATH) if file.startswith("style_mask_")]
            content_masks_path = [os.path.join(CONTENT_MASK_PATH, mask) for mask in content_masks]
            style_masks_path = [os.path.join(STYLE_MASK_PATH, mask) for mask in style_masks]
            masks_path += (content_masks_path + style_masks_path)

            # Add files to summary directory
            log_files(masks_path)

            return render_template(
                "colour_masks.html",
                n=n_colours,
                user_defined=str(user_defined_regions),
                content_masks=sorted(content_masks),
                style_masks=sorted(style_masks),
                n_masks=numbers[n_colours]
                )



# Style Transfer Route
@app.route("/style_transfer", methods =["POST"])
def style_transfer():
    transfer_option = request.form["type"]
    
    # Semantic style transfer
    if transfer_option == "semantic":
        # Get selected regions and form final transfer mask
        form = request.form.to_dict(flat=False)
        selected_masks = [x for x in list(form) if x != "type"]
        combine_coloured_masks(selected_masks, target=0)

        # Parameters of Style Transfer Model
        c_mask = os.path.join(CONTENT_MASK_PATH, "combined_mask.jpg") # newly formed mask
        s_mask = None # no style mask
        n_colors = 1 # style components transfered to white region of content mask

        # Log transfer parameters
        log_text("CONTENT MASK: " + str(selected_masks))
        log_text("STYLE MASK: none")
        log_files([c_mask], ["content_combined_mask.jpg"])
    

    # Threshold Style Transfer
    elif transfer_option == "threshold":
        # Get selected masks for content and style
        content_masks = request.form.getlist("content_masks")
        style_masks = request.form.getlist("style_masks")
        n_colors = int(request.form["n_colors"])
        # Automatic pairing -> n_colors = n_colors
        if len(content_masks)==0 and len(style_masks)==0:
            c_mask = os.path.join(CONTENT_MASK_PATH, "content_threshold_mask.jpg")
            s_mask = os.path.join(STYLE_MASK_PATH, "style_threshold_mask.jpg")
            log_text("CONTENT MASK: content_threshold_mask.jpg")
            log_text("STYLE MASK: style_threshold_mask.jpg")
        # User-defined regions -> n_colors = 1
        else:
            combine_masks(content_masks, target=0)
            combine_masks(style_masks, target=1)

            # Style Transfer Parameters
            c_mask = os.path.join(CONTENT_MASK_PATH, "combined_mask.jpg")
            s_mask = os.path.join(STYLE_MASK_PATH, "combined_mask.jpg")

            # Log transfer parameters
            log_text("CONTENT MASK: " + str(content_masks))
            log_text("STYLE MASK: " + str(style_masks))

            log_files([c_mask], ["content_combined_mask.jpg"])
            log_files([s_mask], ["style_combined_mask.jpg"])
    

    # Colour Style Transfer
    elif transfer_option == "colour":
        # Get selected masks for content and style
        content_masks = request.form.getlist("content_masks")
        style_masks = request.form.getlist("style_masks")
        n_colors = int(request.form["n_colors"])
        # Automatic pairing -> n_colors = n_colors
        if len(content_masks)==0 and len(style_masks)==0:
            c_mask = os.path.join(CONTENT_MASK_PATH, "content_colour_mask.jpg")
            s_mask = os.path.join(STYLE_MASK_PATH, "style_colour_mask.jpg")
            log_text("CONTENT MASK: content_colour_mask.jpg")
            log_text("STYLE MASK: style_colour_mask.jpg")
        # User defined regions -> n_colors = 1
        else:
            combine_coloured_masks(content_masks, target=0)
            combine_coloured_masks(style_masks, target=1)

            c_mask = os.path.join(CONTENT_MASK_PATH, "combined_mask.jpg")
            s_mask = os.path.join(STYLE_MASK_PATH, "combined_mask.jpg")

            log_text("CONTENT MASK: " + str(content_masks))
            log_text("STYLE MASK: " + str(style_masks))

            log_files([c_mask], ["content_combined_mask.jpg"])
            log_files([s_mask], ["style_combined_mask.jpg"])

    # Parameter to limit input images width in the style transfer
    hard_width = False

    # Create Style Transfer Model and do the style transfer
    model = TransferModel(n_colors, hard_width, c_mask, s_mask)
    model.apply_transfer()

    # Log output image to summary directory
    log_files([OUTPUT_IMAGE_PATH])
    log_text("OUTPUT: output.jpg")
    
    return render_template("output.html", image="output.jpg")


# Full Transfer Route
@app.route("/full_transfer")
def full_transfer():
    c_mask = None
    s_mask = None
    model = TransferModel(1, False, c_mask, s_mask)
    model.apply_transfer()
    log_files([OUTPUT_IMAGE_PATH])
    return render_template("output.html", image="output.jpg")


# Download Results Route
@app.route('/download_results')
def download_results():

    # Create zip file from the summary directory
    shutil.make_archive(
        base_name="static/summary",
        format="zip",
        root_dir=SUMMARY_PATH
        )

    # Return it to user as downloadable file
    return send_file(
        filename_or_fp="static/summary.zip",
        mimetype='application/zip',
        as_attachment=True,
        attachment_filename='summary.zip'
        )



    
if __name__ == "__main__":
    app.run(host="localhost", port=8000, debug=False)
