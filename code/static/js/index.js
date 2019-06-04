function init(){
    document.getElementById("n_colours_label").innerText = "4";
    document.getElementById("n_threshold_label").innerText = "1";
}

function validate(form) {
    var content = form.elements[0];
    var style = form.elements[1];
    var select = form.elements[2];
    var options = select.options;

    if (content.value === "") {
        alert("Please upload a content image.");
        return false;
    }
    else if (style.value === "") {
        alert("Please upload a style image as well.");
        return false;
    }
    else if (options.selectedIndex === 0) {
        if (!confirm("Perform style transfer without any guidance mask?")){
            return false;
        }
    }
    activateLoader();
}


function dispExtraOptions() {
    var e = document.getElementById("transfer_select");
    var selected = e.options[e.selectedIndex].value;
    var thresholdOption = document.getElementById("threshold_option");
    var colourOption = document.getElementById("colour_option");
    if (selected === "threshold") {
        thresholdOption.style.display="inline";
        colourOption.style.display="none";
    } else if (selected === "colour") {
        thresholdOption.style.display="none";
        colourOption.style.display="inline";
    } else {
        thresholdOption.style.display = "none";
        colourOption.style.display = "none";
    }

}


function colourSliderCallback(slider) {
    document.getElementById("n_colours_label").innerText = slider.value;
}

function thresholdSliderCallback(slider) {
    document.getElementById("n_threshold_label").innerText = slider.value;
}


function previewContentImage() {
    var preview = document.querySelector('img#content-image');
    var file = document.querySelector('input#hidden-content-image').files[0];
    var reader = new FileReader();

    reader.addEventListener("load", function () {
        preview.style.display = 'block';
        preview.src = reader.result;
    }, false);

    if (file) {
        reader.readAsDataURL(file);
    }
}

function previewStyleImage() {
    var preview = document.querySelector('img#style-image');
    var file = document.querySelector('input#hidden-style-image').files[0];
    var reader = new FileReader();


    reader.addEventListener("load", function () {
        preview.style.display = 'block';
        preview.src = reader.result;
    }, false);

    if (file) {
        reader.readAsDataURL(file);
    }
}

function activateLoader() {
    
    var loader = document.getElementById("segmentation_loader");

    var e = document.getElementById("transfer_select");
    var selected = e.options[e.selectedIndex].value;

    if (selected !== 'full') {
        loader.innerText = 'Performing ' + selected + ' segmentation...';
    } else {
        loader.innerText = 'Performing full style transfer...';
    }

    $('.ui.basic.modal').modal('show');
}



init();
