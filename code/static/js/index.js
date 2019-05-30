function validate(form) {
    var content = form.elements[0];
    var style = form.elements[1];
  
    if (content.value === "") {
        alert("Please upload a content image.");
        content.focus();
    }
    else if (style.value === "") {
        alert("Please upload a style image as well.");
        style.focus();
    }
    else {
        return true;
    }
    return false;
  }