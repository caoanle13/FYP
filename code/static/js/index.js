document.getElementById("n_colours_label").innerText = "4";
document.getElementById("n_threshold_label").innerText = "1";


// Event handler for form validation
let form = document.getElementById("image-form");
form.addEventListener("submit", function () {

    console.log("form validation!");

    let content = this.elements[0];
    let style = this.elements[1];
    let selectMenu = this.elements[2];
    let options = selectMenu.options;
    let selected = options[options.selectedIndex].value;
    let loader = document.getElementById("segmentation_loader");

    // Check content image field
    if (content.value === "") {
        alert("Please upload a content image.");
        return false;
    }
    // Check style image field
    else if (style.value === "") {
        alert("Please upload a style image as well.");
        return false;
    }
    // Ask for confirmation on full style transfer
    else if (selected === "full") {
        if (!confirm("Perform style transfer without any guidance mask?")) {
            return false;
        } else {
            // Activate loader for full style transfer
            loader.innerText = 'Performing full style transfer...';
            return true;
        }
    }
    // If everything correct, create and show loader
    loader.innerText = 'Performing ' + selected + ' segmentation...';
    $('.ui.basic.modal').modal('show');
    return true;

});


// Event handler on input images to display them
function previewImage(type) {
    let preview = document.querySelector(`img#${type}-image`);
    let file = document.querySelector(`input#hidden-${type}-image`).files[0];
    let reader = new FileReader();

    reader.addEventListener("load", () => {
        preview.style.display = 'block';
        preview.src = reader.result;
    }, false);

    if (file) {
        reader.readAsDataURL(file);
    }
}
let contentImage = document.getElementById("hidden-content-image");
let styleImage = document.getElementById("hidden-style-image");
contentImage.addEventListener("change", () => { previewImage("content");});
styleImage.addEventListener("change", () => { previewImage("style");});


// Event handler to display extra options depending on the menu selection
let selectMenu = document.getElementById("transfer_select");
selectMenu.addEventListener("change", function () {

    document.getElementById("threshold_option").style.display = "none";
    document.getElementById("colour_option").style.display = "none";
    document.getElementById("region_option").style.display = "none";

;
    let selected = this.options[this.selectedIndex].value;
    if (selected === "threshold" || selected === "colour"){
        document.getElementById(`${selected}_option`).style.display = "inline";
        document.getElementById("region_option").style.display = "inline";
    }

});

// Event handler for slider changes
function sliderHandler(type) {
    let slider = document.getElementById(`n_${type}`)
    let label = document.getElementById(`n_${type}_label`);
    label.innerText = slider.value;
}
let colourSlider = document.getElementById("n_colours");
let thresholdSlider = document.getElementById("n_threshold");
colourSlider.addEventListener("change", () => { sliderHandler("colours")});
thresholdSlider.addEventListener("change", () => { sliderHandler("threshold")});