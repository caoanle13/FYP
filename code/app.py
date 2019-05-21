from flask import Flask, render_template, request
import os
import shutil
from src.mask import create_masks

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    IMAGE_PATH = os.path.join(APP_ROOT, "images/")

    if os.path.isdir(IMAGE_PATH):
        shutil.rmtree(IMAGE_PATH)
    os.mkdir(IMAGE_PATH)
    #CONTENT_PATH = os.path.join(IMAGE_PATH, "content")
    #STYLE_PATH = os.path.join(IMAGE_PATH, "style")
    #os.mkdir(CONTENT_PATH)
    #os.mkdir(STYLE_PATH)
    
    content_image = request.files.getlist("content_image")[0] #<class 'werkzeug.datastructures.FileStorage'>    
    content_image.save("/".join([IMAGE_PATH, content_image.filename]))
    print("FILE NAME IS", content_image.filename)
    create_masks(content_image.filename)
    return render_template("complete.html", image_name=content_image.filename)
    
if __name__ == "__main__":
    app.run(host="localhost", port=8000)