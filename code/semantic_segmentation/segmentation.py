import os
import sys
from PIL import Image
import numpy as np
import tensorflow as tf
from shutil import rmtree

# Dict containing train ID to RGB colour mappings
from semantic_segmentation.libs.constants import CITYSCAPES_LABEL_COLORS, CITYSCAPES_LABEL_NAMES
from semantic_segmentation.image_helpers import array_to_pil, boolean_to_pil, superimpose

# Root directory of the project as string
ROOT_DIR = os.path.abspath("./")



class SegmentationModel():

    def __init__(self):
        """ 
        Class to for a Segmentation model trained on the Cityscapes dataset and using the PSPNET network.
        """

        print("Initialising the segmentation model...")
        # Root Directory of the model we downloaded
        self.MODEL_NAME = 'pspnet'

        # Full path to frozen graph for the model.
        self.PATH_TO_FROZEN_GRAPH = os.path.join(ROOT_DIR, "semantic_segmentation/{}/frozen_inference_graph_opt.pb".format(self.MODEL_NAME))
        self.OUTPUT_DIR = os.path.join(ROOT_DIR, "static/masks")

        print(self.OUTPUT_DIR)

        # We are using a model trained on Cityscapes
        # so the output is 19 classes
        self.NUM_CLASSES = 19

        # Input and output dimensions
        self.INPUT_SIZE = (2048, 1024)
        self.OUTPUT_SIZE=(1025,2049,3)

        print("Loading a frozen Tensorflow model into memory...")

        self.segmentation_graph = tf.Graph()
        with self.segmentation_graph.as_default():
            segmentaion_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_FROZEN_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                segmentaion_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(segmentaion_graph_def, name='')

        print("Segmentation model is ready!")




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



    def infer(self, path):
        """ Function to perform semantic segmentation.
        It produces an RGB mask, and individual binary masks for each class
        
        Arguments:
            - path {str}: Path to the input image to run the inference on.
        """

        if os.path.isdir(self.OUTPUT_DIR):
            rmtree(self.OUTPUT_DIR)
        os.makedirs(self.OUTPUT_DIR)

        # Read input image
        image = Image.open(path)
        input_size = image.size

        image_np = self.load_image_into_numpy_array(image)

        # Actual inference result
        segmentation_mask = self.run_inference_for_single_image(image_np, self.segmentation_graph) #contains IDs

        # Produce colored mask and save it
        rgb_segmentation_mask = self.decode_train_ids(segmentation_mask) #actual colored image
        rgb_segmentation_mask = array_to_pil(rgb_segmentation_mask).resize(input_size)
        rgb_segmentation_mask.save(self.OUTPUT_DIR + "/segmentation_mask.jpg")
        
        # Produce individual masks for each label
        masks = self.produce_masks(segmentation_mask)

        # Save them as PIL images
        for i, mask in enumerate(masks):
            mask = mask[:,:,0]
            if np.sum(mask) * 10 > mask.size: # Making sure the region covers at least a 10th of the image area
                mask = boolean_to_pil(mask)
                mask = mask.resize(input_size)
                mask.save(self.OUTPUT_DIR + "/mask_" + CITYSCAPES_LABEL_NAMES[i] + ".jpg")

        # Superimpose the image and its mask
        superimposed_image = superimpose(image, rgb_segmentation_mask)
        superimposed_image.save(self.OUTPUT_DIR + "/superimposed_image.jpg")
    

    


if __name__ == "__main__":
    model = SegmentationModel()
    model.infer("house.jpg")

