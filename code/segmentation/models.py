import os
import sys
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from shutil import rmtree
from paths import *
import imageio
from scipy.cluster.vq import whiten, kmeans
from scipy.spatial.distance import euclidean
from itertools import repeat
import multiprocessing
import cv2

sys.path.append(os.path.join(APP_ROOT, "segmentation/"))

# Dict containing train ID to RGB colour mappings
from city_constants import CITYSCAPES_LABEL_COLORS, CITYSCAPES_LABEL_NAMES
from image_helpers import array_to_pil, boolean_to_pil, superimpose, equalize, get_colour_name





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



    def produce_masks(self, image, origin_image):
        """ Helper function to produce separate masks from segmentation mask with IDs.
        
        Arguments:
            - image {nd array}: numpy array containing IDs for individual labels.
            - origin_image {nd array}: numpy array of the original image.
        Returns:
            {list}: list of masks which are boolean numpy arrays (1 mask per class label).
        """
        
        masks = []
        image = np.squeeze(image, axis=0)
        for i in range(self.NUM_CLASSES):
            selection = (image == i)
            mask = np.multiply(selection, origin_image)
            masks.append(mask)
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
        semantic_masks = self.produce_masks(segmentation_mask, np.array(image.resize((2049, 1025))))

        # Save them as numpy arrays and PIL images
        for i, mask in enumerate(semantic_masks):
            if mask.any(axis=-1).sum() * 15 > mask.size/3: # Making sure the region covers at least a 15th of the image area
                # For future processing
                mask = cv2.resize(mask, dsize=(input_size[0], input_size[1]))
                np.save(self.SAVE_DIR + "content_mask_" + CITYSCAPES_LABEL_NAMES[i] + ".npy", mask)
                # For display
                mask = array_to_pil(mask)
                mask = mask.resize(input_size)
                mask.save(self.SAVE_DIR + "content_mask_" + CITYSCAPES_LABEL_NAMES[i] + ".jpg")

        # Superimpose the image and its mask
        superimposed_image = superimpose(image, rgb_segmentation_mask)
        superimposed_image.save(self.SAVE_DIR + "superimposed_image.jpg")
    

    



class ThresholdModel():
    """ 
        Class for a Threshold Segmentation model.
    """

    def __init__(self, target, n_threshold, user_defined):
        self.n_threshold = n_threshold
        self.SAVE_DIR = STYLE_MASK_PATH if target else CONTENT_MASK_PATH
        self.type = "style" if target else "content"
        self.save_name = self.type + "_threshold_mask.jpg"
        self.user_defined = user_defined


    def create_colored_masks(self, image, thresholds, colors):
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
        for i in range(len(colors)):
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


    def produce_masks(self, image, colours):
        """ Helper function to produce separate masks from segmentation mask with colours.
        
        Arguments:
            - image {nd array}: numpy array containing colours for individual regions.
        Returns:
            {list}: list of masks which are boolean numpy arrays (1 mask per class label).
        """
        
        masks = []
        for colour in colours:
            masks.append(image == colour)
        return masks





    def segment(self, path):
        """ Function to perform threshold segmentation.
        It saves a "coloured" mask.
        
        Arguments:
            - path {str}: Path to the input image to be thresholded.
            - target {int}: 0 for content image, 1 for style image.
        """

        print("Doing the threhsold segmentation on {}...".format(path))

        n = self.n_threshold

        # Open image and equalize its histogram.
        image = Image.open(path)
        image = equalize(image)
       
        w, h = image.size

        # Define gray scale bands and thresholds:
        colors = [0]
        thresholds = [0]
        for i in range(1, n+1):
            colors.append(round(255/n)*i-1)
            thresholds.append(round(255/(n+1)) * i)
        thresholds.append(255)

        # Convert to gray scale numpy array
        gray_image_np = np.array(image.convert(mode="L"))

        # Create colored mask
        coloured_masks = self.create_colored_masks(gray_image_np, thresholds, colors)
            
        # Combine masks together
        output_mask = self.merge_masks(coloured_masks, h, w)

        # Save output
        img = Image.fromarray(output_mask.astype('uint8'))
        img.save(self.SAVE_DIR + self.save_name)


        if self.user_defined:
             # Produce individual masks for each label
            masks = self.produce_masks(output_mask, colors)

            # Save
            for mask, color in zip(masks, colors):
                # For future processing
                np.save(self.SAVE_DIR + self.type + "_mask_" + str(color) + ".npy", mask)
                # For display
                mask = boolean_to_pil(mask)
                mask.save(self.SAVE_DIR + self.type + "_mask_" + str(color) + ".jpg")






class ColourModel():
    """ 
        Class for a Colour-based Segmentation model.
    """

    def __init__(self, base, n_colours, user_defined):
        """ Constructor function for the colour-based segmentation model class.
        
        Arguments:
            - base {int}: 0 for content 1 for style -> choose colour palette.
            - target {int}: 0 for content 1 for style -> image on which to apply the colour segementation.
            - n_colours {int}: number of dominant colours to consider.
        """

        print("Initialising the colour-based segmentor model...")
        self.n_colours = n_colours
        self.BASE_PATH = STYLE_IMAGE_PATH if base else CONTENT_IMAGE_PATH   
        self.dom_cols = self.dominant_colours(self.BASE_PATH)
        self.user_defined = user_defined


    def dominant_colours(self, path):
        """ Function to find the dominant colours of in an image.
        
        Arguments:
            - path {str}: Path to the input image.
        
        Returns:
             {list}: List of RGB dominant colours.
        """

        print("Looking for the {} most dominant colours in {}...".format(self.n_colours, path))
        
        # Read input image and get separate channels
        img = np.array(imageio.imread(path))
        r = img[:,:,0].flatten()
        g = img[:,:,1].flatten()
        b = img[:,:,2].flatten()  

        # Standardise variables for KMeans algorithm
        scaled_r = whiten(r)
        scaled_g = whiten(g)
        scaled_b = whiten(b)

        # Run KMeans to find main clusters
        cluster_centers, distortion = kmeans(np.transpose([scaled_r, scaled_g, scaled_b]), self.n_colours)
        out_colours = []

        # Construct and return the result
        for colour in cluster_centers:
            scaled_r, scaled_g, scaled_b = colour
            out_colours.append([
                int(scaled_r * r.std()),
                int(scaled_g * g.std()),
                int(scaled_b * b.std())
            ])
        
        return out_colours

    def produce_masks(self, image, colours):
        """ Helper function to produce separate COLOURED masks from segmentation mask with colours.
        
        Arguments:
            - image {nd array}: numpy array containing colours for individual regions.
            - colours {list}: list of colours.
        Returns:
            {list}: list of masks which are boolean numpy arrays (1 mask per class label).
        """
        masks = []
        
        for colour in colours:
            selection = (image == colour)
            masks.append(np.multiply(selection, image))
        return masks



    def closest(self, point, others):
        """ Given a point and a group of other points, finds the minimum distance pair.
        
        Arguments:
            point {} -- Test point.
            others {iterator} -- Group of points.
        
        Returns:
            point -- The point among the group of points that has the minimum distance to the test point.
        """
        return min(others, key = lambda i: euclidean(point, i))



    def closest_row(self, row):
        return list( map(self.closest, row, repeat(self.dom_cols)) )



    def segment(self, target):
        """ Function to perform colour-based segmentation.
        
        Arguments:
            - target {int}: Image to be thresholded: 0 for content 1 for style.
        """

        path = STYLE_IMAGE_PATH if target else CONTENT_IMAGE_PATH

        print("Doing the colour segmentation on {}...".format(path))

        # Open image
        image = np.array(imageio.imread(path))
        output_image = np.empty(image.shape)
        
        # Replace every pixel by the nearest neighbor in the dominant colours
        with multiprocessing.Pool() as pool:
            output_image = np.array(pool.map(
                self.closest_row,
                [x[0] for x in np.split(image, image.shape[0])]
            ))

        
        # Save output
        SAVE_DIR = STYLE_MASK_PATH if target else CONTENT_MASK_PATH
        mask_type = "style" if target else "content"
        save_name = mask_type + "_colour_mask.jpg"
        print("Saving the colour segmentation output '{}_colour_mask.jpg' in {}...".format(mask_type, SAVE_DIR))
        imageio.imwrite(SAVE_DIR + save_name, output_image)

        if self.user_defined:
             # Produce individual masks for each label
            masks = self.produce_masks(output_image, self.dom_cols)

            # Save
            for mask, color in zip(masks, self.dom_cols):
                # For future processing
                np.save(SAVE_DIR + mask_type + "_mask_" + get_colour_name(color)[1] + ".npy", mask)
                # For display
                mask = array_to_pil(mask)
                mask.save(SAVE_DIR + mask_type + "_mask_" + get_colour_name(color)[1] + ".jpg")


if __name__ == "__main__":

    cm = ColourModel(0, 3)

