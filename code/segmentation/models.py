import os
import sys
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from shutil import rmtree
from paths import *

sys.path.append(os.path.join(APP_ROOT, "segmentation/"))

# Dict containing train ID to RGB colour mappings
from libs.constants import CITYSCAPES_LABEL_COLORS, CITYSCAPES_LABEL_NAMES
from image_helpers import array_to_pil, boolean_to_pil, superimpose, equalize





class SemanticModel():
    """ 
        Class for a Semantic Segmentation Model trained on the Cityscapes dataset and using the PSPNET network.
    """

    def __init__(self, target):

        print("Initialising the segmentation model...")
        # Root Directory of the model we downloaded
        self.MODEL_NAME = 'pspnet'

        # Full path to frozen graph for the model.
        self.PATH_TO_FROZEN_GRAPH = os.path.join(APP_ROOT, "segmentation/{}/frozen_inference_graph_opt.pb".format(self.MODEL_NAME))
        
        # We are using a model trained on Cityscapes (19 classes)
        self.NUM_CLASSES = 19

        # Input and output dimensions
        self.INPUT_SIZE = (2048, 1024)
        self.OUTPUT_SIZE=(1025,2049,3)

        self.SAVE_DIR = STYLE_MASK_PATH if target else CONTENT_MASK_PATH


        print("Loading a frozen Tensorflow model into memory...")

        self.segmentation_graph = tf.Graph()
        with self.segmentation_graph.as_default():
            segmentaion_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_FROZEN_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                segmentaion_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(segmentaion_graph_def, name='')

        print("Semantic segmentation model is ready!")




    def decode_train_ids(self, image):

        """ Function to map label IDs to colored pixels.

        Arguments:
            - image {nd array}:  of shape [1,1024,2048,1] containing label IDs.
        Returns:
            {nd array}: numpy array containing the corresponding color for each label.
        """

        image = np.squeeze(image, axis=0) # predictions will have the shape [1,1024,2048,1]
        rgb_image = np.zeros(self.OUTPUT_SIZE)
        for train_id in range(self.NUM_CLASSES):
            rgb_image[np.where((image==train_id).all(axis=2))] = CITYSCAPES_LABEL_COLORS[train_id]
        return rgb_image



    def load_image_into_numpy_array(self, image):

        """ Function to read image and resize to the known size to the model.
        
        Arguments:
            - image {PIL Image}: input image.
        Returns:
            {nd array}: output numpy array.
        """
        
        image = image.resize(self.INPUT_SIZE)
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)



    def run_inference_for_single_image(self, image, graph):

        """ Function to run inference on a single image.
        
        Arguments:
            - image {nd array}: input numpy array.
            - graph {tf Graph}: tensorflow graph to use for inference.
        Returns:
            {nd array}: numpy array containing IDs for individual labels.
        """
        with graph.as_default():
            with tf.Session() as sess:
                # Grab input and output tensors
                image_tensor = tf.get_default_graph().get_tensor_by_name('inputs:0')
                output_tensor = tf.get_default_graph().get_tensor_by_name('predictions:0')
                # Run inference
                predictions = sess.run(output_tensor,
                                    feed_dict={image_tensor: image})
        return predictions



    def produce_masks(self, image):
        """ Helper function to produce separate masks from segmentation mask with IDs.
        
        Arguments:
            - image {nd array}: numpy array containing IDs for individual labels.
        Returns:
            {list}: list of masks which are boolean numpy arrays (1 mask per class label).
        """
        
        masks = []
        image = np.squeeze(image, axis=0)
        for i in range(self.NUM_CLASSES):
            masks.append(image == i)
        return masks



    def segment(self, path):
        """ Function to perform semantic segmentation.
        It saves an RGB mask, and individual binary masks for each class
        
        Arguments:
            - path {str}: Path to the input image to run the inference on.
            - target {int}: 0 for content image, 1 for style image.
        """

        # Read input image
        image = Image.open(path)
        input_size = image.size

        image_np = self.load_image_into_numpy_array(image)

        # Actual inference result
        segmentation_mask = self.run_inference_for_single_image(image_np, self.segmentation_graph) #contains IDs

        # Produce colored mask and save it
        rgb_segmentation_mask = self.decode_train_ids(segmentation_mask) #actual colored image
        rgb_segmentation_mask = array_to_pil(rgb_segmentation_mask).resize(input_size)
        rgb_segmentation_mask.save(self.SAVE_DIR +  "segmentation_mask.jpg")
        
        # Produce individual masks for each label
        masks = self.produce_masks(segmentation_mask)

        # Save them as PIL images
        for i, mask in enumerate(masks):
            mask = mask[:,:,0]
            if np.sum(mask) * 10 > mask.size: # Making sure the region covers at least a 10th of the image area
                mask = boolean_to_pil(mask)
                mask = mask.resize(input_size)
                mask.save(self.SAVE_DIR + "mask_" + CITYSCAPES_LABEL_NAMES[i] + ".jpg")

        # Superimpose the image and its mask
        superimposed_image = superimpose(image, rgb_segmentation_mask)
        superimposed_image.save(self.SAVE_DIR + "superimposed_image.jpg")
    

    



class ThresholdModel():
    """ 
        Class for a Threshold Segmentation model.
    """

    def __init__(self, target, n_threshold):
        self.n_threshold = n_threshold
        self.SAVE_DIR = STYLE_MASK_PATH if target else CONTENT_MASK_PATH


    def create_colored_masks(self, image, n, thresholds, colors):
        """ Create "colored" (gray scale) masks given a grayscale numpy image and the number of threshold.
        
        Arguments:
            - image {nd array}: The input image in grayscale.
            - n {int}: Number of thresholds.
            - thresholds {list}: Thresholds.
            - colors {list}: Color bands.

        Returns:
            {list}: List of masks for each color band. 
        """
        masks = []
        for i in range(n+1):
            mask = np.logical_and(image > thresholds[i], image <= thresholds[i+1])
            mask = mask.astype(np.int)
            mask *= colors[i]
            masks.append(mask)
        return masks

    
    def merge_masks(self, masks, h, w):
        """ Combines individual masks with different color bands together.
        
        Arguments:
            masks {list}: List containing the individual masks as nd arrays.
            - h {int}: Height.
            - w {int}: Width.
        
        Returns:
            {nd array}: Combined mask.
        """
        combined_mask = np.zeros([h,w],dtype=np.uint8)
        for mask in masks:
            combined_mask = combined_mask | mask
        return combined_mask




    def segment(self, path):
        """ Function to perform threshold segmentation.
        It saves a binary mask, where black pixels areas with intensity lower than the threshold.
        
        Arguments:
            - path {str}: Path to the input image to be thresholded.
            - target {int}: 0 for content image, 1 for style image.
        """

        n = self.n_threshold

        # Open image and equalize its histogram.
        image = Image.open(path)
        image = equalize(image)
        image = equalize(image)
        image = equalize(image)
        image = equalize(image)

        w, h = image.size

        # Define gray scale bands and thresholds:
        colors = [0]
        thresholds = [0]
        for i in range(1, n+1):
            colors.append(round(255/n)*i)
            thresholds.append(round(255/(n+1)) * i)
        thresholds.append(255)

        # Convert to gray scale numpy array
        gray_image_np = np.array(image.convert(mode="L"))

        # Create colored mask
        masks = self.create_colored_masks(gray_image_np, self.n_threshold, thresholds, colors)

        # Combine masks together
        final_mask = self.merge_masks(masks, h, w)

        # Save output
        Image.fromarray(final_mask.astype('uint8')).save(self.SAVE_DIR + "threshold_mask.jpg")
