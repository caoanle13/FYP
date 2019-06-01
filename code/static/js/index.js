function init(){
    document.getElementById("n_threshold_label").innerText = "1";
}

function validate(form) {
    var content = form.elements[0];
    var style = form.elements[1];
    var select = form.elements[2];
    var options = select.options;

    if (content.value === "") {
        alert("Please upload a content image.");
        content.focus();
        return false;
    }
    else if (style.value === "") {
        alert("Please upload a style image as well.");
        style.focus();
        return false;
    }
    else if (options.selectedIndex === 0) {
        if (!confirm("Perform style transfer without any guidance mask?")){
            return false;
        }
    }
}


function dispExtraOptions() {
    var e = document.getElementById("transfer_select");
    var selected = e.options[e.selectedIndex].value;
    var slider = document.getElementById("slider_option");
    if (selected === "threshold") {
        slider.style.display="inline";
    }
    else {
        slider.style.display="none";
    }
}


function sliderCallBack(slider) {
    document.getElementById("n_threshold_label").innerText = slider.value;
}


init();
