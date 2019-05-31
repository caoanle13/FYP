import os
import sys
import tensorflow as tf
import numpy as np
import scipy.misc

from stylisation import Model
from stylisation import Parser
from stylisation.helpers import *
from paths import *
import args


class TransferModel():

    def __init__(self, mask_n_colors, hard_width=False, content_mask=None, style_mask=None):

        self.mask_n_colors = mask_n_colors
        self.hard_width = hard_width
        self.content_mask = content_mask
        self.style_mask = style_mask
        self.args = args


        # 1. read images and preprocess
        self.content_img = read_image(CONTENT_IMAGE_PATH, hard_width) 
        self.style_img = read_image(STYLE_IMAGE_PATH, hard_width) 

        # 2. get stacked 0./1. masks
        if mask_n_colors > 1: # colorful
            self.target_masks_origin, self.style_masks_origin = read_colorful_mask(
                                                                                content_mask,
                                                                                style_mask, 
                                                                                hard_width,
                                                                                mask_n_colors
                                                                                )   
        else: # single mask (mask_n_colors=1) or no mask (mask_n_colors=0)
            if content_mask is None:
                self.target_masks_origin = np.ones(self.content_img.shape[0:3]).astype(np.float32)
            else: 
                self.target_masks_origin = read_single_mask(content_mask, hard_width)
            if style_mask is None:
                self.style_masks_origin = np.ones(self.style_img.shape[0:3]).astype(np.float32)
            else:
                self.style_masks_origin = read_single_mask(style_mask, hard_width)
        

        # 3. Get initial image & target shape
        self.target_shape = self.content_img.shape
        self.init_img = get_init_image(self.content_img, args.init_noise_ratio)
        

        # 4. Check shape & number of masks
        if self.content_img.shape[1:3] != self.target_masks_origin.shape[1:3]:
            print('content and mask have different shape')
            sys.exit(0)
        if self.style_img.shape[1:3] != self.style_masks_origin.shape[1:3]:
            print('style and mask have different shape')
            sys.exit(0)
        if self.target_masks_origin.shape[0] != self.style_masks_origin.shape[0]: #0th index is number of colors
            print('content and style have different masks')
            sys.exit(0)




    def build_features_and_network(self):
        '''
        Compute features & build network
        '''
        # 1. Prepare model weights
        vgg_weights = Model.prepare_model(args.model_path)

        # 2. Compute feature maps of specific layers
        self.content_features = compute_features(vgg_weights,
                                            args.feature_pooling_type,
                                            self.content_img,
                                            args.content_layers)  

        self.style_features = compute_features(  vgg_weights,
                                            args.feature_pooling_type,
                                            self.style_img,
                                            args.style_layers)

        # 3. Compute masks of specific layers
        # ================================ CHECK THIS!!!!!!!!!!!!!!!!!!!!!!!!!! ====================================
        self.target_masks = compute_layer_masks( self.target_masks_origin,
                                            args.style_layers,
                                            args.mask_downsample_type
                                            ) 
        self.style_masks = compute_layer_masks(self.style_masks_origin,
                                            args.style_layers,
                                            args.mask_downsample_type
                                            )

        # 4. Build the network
        self.target_net = build_target_net(vgg_weights, args.feature_pooling_type, self.target_shape)


    def compute_losses(self):
        '''
        Losses
        '''    
        content_loss = sum_content_loss(self.target_net,
                                        self.content_features, 
                                        args.content_layers,
                                        args.content_layers_weights,
                                        args.content_loss_normalization
                                        )
    
        style_masked_loss = sum_masked_style_loss(  self.target_net,
                                                    self.style_features, 
                                                    self.target_masks,
                                                    self.style_masks, 
                                                    args.style_layers,
                                                    args.style_layers_weights, 
                                                    args.mask_normalization_type
                                                )

        if args.tv_weight != 0.:
            tv_loss = sum_total_variation_loss(self.target_net['input'], self.target_shape)
        else:
            tv_loss = 0.

        self.total_loss = args.content_weight * content_loss + \
                    args.style_weight * style_masked_loss + \
                    args.tv_weight * tv_loss


    def train(self):
        '''
        train 
        '''
        if not os.path.exists(OUTPUT_PATH):
            os.mkdir(OUTPUT_PATH)

        if args.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(args.learning_rate)
            train_op = optimizer.minimize(self.total_loss)

            #init
            init_op = tf.global_variables_initializer() # must! Adam has some varibales to init
            self.sess = tf.Session()
            self.sess.run(init_op)
            self.sess.run( self.target_net['input'].assign(self.init_img) )

            #train
            for i in range(args.iteration):
                self.sess.run(train_op)
                if i % args.log_iteration == 0:
                    print('Iteration %d: loss = %f' % (i+1, self.sess.run(self.total_loss)))
                    result = self.sess.run(self.target_net['input'])
                    output_path = os.path.join(OUTPUT_PATH, 'result_%s.png' % (str(i).zfill(4)))
                    write_image(output_path, result)
        
        elif args.optimizer == 'lbfgs':
            optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                self.total_loss, method='L-BFGS-B',
                options={'maxiter': args.iteration,
                        'disp': args.log_iteration})  

            # init  
            init_op = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init_op)
            self.sess.run( self.target_net['input'].assign(self.init_img) )

            # train
            optimizer.minimize(self.sess) 



    def save_result(self):
        '''
        out
        '''
        print('Iteration %d: loss = %f' % (args.iteration, self.sess.run(self.total_loss)))
        result = self.sess.run(self.target_net['input'])
        output_path = os.path.join(OUTPUT_PATH, 'output.jpg')
        write_image(output_path, result)



    def apply_transfer(self):
        """ Main function to perform the appropriate style transfer given the parameters.
        
        Arguments:
            mask_n_colors {int} -- Number of colors in mask
        
        Keyword Arguments:
            content_mask {str} -- Path to the content mask (default: {None})
            style_mask {str} -- Path to the style mask (default: {None})
        """
        # transfer can either be    "full" -> no content mask, no style mask, n_colors = 1
        #                           "threshold" -> content mask "/masks/content/threshold_mask.jpg", style mask "/masks/style/threshold_mask.jpg", n_colors=2
        #                           "semantic" -> content mask "/masks/content/semantic_mask.jpg", style mask "/masks/style/semantic_mask", n_colors = x
        
        print('Building features and network...\n')
        self.build_features_and_network()

        print('Computing losses...\n')
        self.compute_losses()

        print('Training the network')
        self.train()

        print('Saving the style transfer output')
        self.save_result()