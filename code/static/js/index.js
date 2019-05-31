function validate(form) {
    var content = form.elements[0];
    var style = form.elements[1];
    var select = form.elements[2];
    var options = select.options;
  
    if (content.value === "") {
        alert("Please upload a content image.");
        content.focus();
    }
    else if (style.value === "") {
        alert("Please upload a style image as well.");
        style.focus();
    }
    else if (options.selectedIndex = 0) {
        confirm("Perform style transfer without any guidance mask?");
    }
    else {
        return true;
    }
    return false;
  }