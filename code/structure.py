import os
from shutil import rmtree, copy
from paths import *
import zipfile
import io
import pathlib


def cleanup():
   """ Clean up the project directory tree when starting the app.
   """
    
   if os.path.isdir(IMAGE_PATH):
      rmtree(IMAGE_PATH)
   os.mkdir(IMAGE_PATH)

   if os.path.isdir(MASK_PATH):
      rmtree(MASK_PATH)
   os.mkdir(MASK_PATH)

   if os.path.isdir(OUTPUT_PATH):
      rmtree(OUTPUT_PATH)
   os.mkdir(OUTPUT_PATH)

   os.mkdir(CONTENT_MASK_PATH)
   os.mkdir(STYLE_MASK_PATH)

   if os.path.isdir(SUMMARY_PATH):
      rmtree(SUMMARY_PATH)
   os.mkdir(SUMMARY_PATH)
   


def add_summary_files(files, names=None):
   """ Produces a summary folder with containing all the experiment details about the style transfer.
   
   Arguments:
       - files {list}: List of file paths to copy to the summary directory.
       - names {list}: List of names to replace the old ones.
   """
   if names is None:
      names = [file.split('/')[-1] for file in files]

   for file, name in zip(files, names):
      copy(file, os.path.join(SUMMARY_PATH, name))


def write_to_summary_file(text):

   with open(SUMMARY_FILE_PATH, "a+") as f:
      f.write(text +'\n')



def produce_zip(folder):
   """ Compresses a folder.
   
   Arguments:
      - folder {str}: Path to the folder to be compressed.
   
   Returns:
      - data: Compressed data.
   """
   zipf = zipfile.ZipFile("summary.zip", "w", zipfile.ZIP_DEFLATED)
   for root, dirs, files in os.walk("./static/summary"):
      for file in files:
         zipf.write(os.path.join(root, file))
   zipf.close()




