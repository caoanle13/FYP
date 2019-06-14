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
   


def log_files(files, names=None):
   """ Produces a summary folder with containing all the experiment details about the style transfer.
   
   Arguments:
       - files {list}: List of file paths to copy to the summary directory.
       - names {list}: List of names to replace the old ones.
   """
   if names is None:
      names = [file.split('/')[-1] for file in files]

   for file, name in zip(files, names):
      copy(file, os.path.join(SUMMARY_PATH, name))


def log_text(text):

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
   for root, dirs, files in os.walk(folder):
      for file in files:
         zipf.write(os.path.join(root, file))
   zipf.close()


def is_jpg_mask(filename):
   """ Checks whether a file is one of the individual mask segments to display.
   
   Arguments:
       filename {str} -- File name.
   
   Returns:
       boolean -- True or False.
   """
   if filename.startswith("content_mask_") or filename.startswith("style_mask"):
      if filename.endswith(".jpg"):
         return True
   return False



def to_npy_str(filenames):
   """ Given the list of a filenames, it replaces whatever their extension is with '.npy' 
   
   Arguments:
       filename {list} -- List of file names.
   
   Returns:
       str -- Modified file name.
   """
   npy = []
   for filename in filenames:
      root = filename.split(".")[0]
      npy.append(root + ".npy")
   return npy



def get_masks(target):
   """ Function to retrieve the list of mask names.
   
   Arguments:
       target {int} -- 0 for content, 1 for style.
   
   Returns:
       list -- List of strings of mask file names.
   """
   path = STYLE_MASK_PATH if target else CONTENT_MASK_PATH
   masks = [f for f in os.listdir(path) if is_jpg_mask(f)]
   return masks



def get_paths(dir, files):
   """Function to retrieve a list of paths.
   
   Arguments:
       dir {path} -- Path to the directory from which to create paths.
       files {list} -- List of filenames.
   
   Returns:
      list -- List of created paths.
   """
   paths = [os.path.join(dir, file) for file in files]
   return paths

