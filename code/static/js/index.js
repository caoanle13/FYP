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