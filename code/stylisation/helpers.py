""" Helper functions for the Style Transfer Model """


import scipy.misc
import imageio
import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans
from stylisation import Model


def read_image(path, hard_width):   # read and preprocess
    """ Reads and preprocess single image.
    
    Returns:
        {nd array} -- the preprocessed image as a numpy array.
    """
    img = imageio.imread(path)      
    if hard_width:
        img = scipy.misc.imresize(img, float(hard_width) / img.shape[1])
    img = img.astype(np.float32)
    img = img[np.newaxis, :, :, :]
    img = img - [123.68, 116.779, 103.939]
    return img



def read_single_mask(path, hard_width): 
    """ Reads a single spatial mask.
    
    Arguments:
        path {str}: Path to the mask.
        hard_width {int}: Size limit.
    
    Returns:
        {np stack}: stacked mask.
    """         
    rawmask = imageio.imread(path)
    if hard_width:
        rawmask = scipy.misc.imresize(rawmask, float(hard_width) / rawmask.shape[1], interp='nearest')    
    rawmask = rawmask / 255 # integer division, only pure white pixels become 1
    rawmask = rawmask.astype(np.float32)   
    single = (rawmask.transpose([2, 0, 1]))[0]
    return np.stack([single])



# colorful, run K-Means to get rid of possible intermediate colors
def read_colorful_mask(target_path, style_path, hard_width, n_colors):
    """ Reads a colorful mask (n_mask_color > 2) and runs K-Means to get rid of possible intermediate colors.
    
    Arguments:
        - target_path {str}: Path to content mask.
        - style_path {str}: Path to style mask.
        - hard_width {int}: Size limit.
        - n_colors {int}: Number of colors in the colorful mask.
    
    Raises:
        - AttributeError: when this function is called with empty mask.
    
    Returns:
        {np stack}: stacked masks.
    """

    # Verify that content and style masks exist
    if target_path is None or style_path is None:
        raise AttributeError("mask path can't be empty when n_colors > 1 ")

    # Read the masks
    target_mask = imageio.imread(target_path)
    style_mask = imageio.imread(style_path)

    # Adjust size if necessary
    if hard_width: # use 'nearest' to avoid more intermediate colors
        target_mask = scipy.misc.imresize(target_mask, float(hard_width) / target_mask.shape[1], 
            interp='nearest') 
        style_mask = scipy.misc.imresize(style_mask, float(hard_width) / style_mask.shape[1], 
            interp='nearest')
    
    # Flatten masks
    target_shape = target_mask.shape[0:2]
    target_mask = target_mask.reshape([target_shape[0]*target_shape[1], -1])
    style_shape = style_mask.shape[0:2]
    style_mask = style_mask.reshape([style_shape[0]*style_shape[1], -1])

    # Compute KMeans clustering on the style mask with n_colors as the number of clusters.
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(style_mask)

    # Predict labels
    target_labels = kmeans.predict(target_mask.astype(np.float32))
    target_labels = target_labels.reshape([target_shape[0], target_shape[1]])
    style_labels = kmeans.predict(style_mask.astype(np.float32))
    style_labels = style_labels.reshape([style_shape[0], style_shape[1]])

    # Create lists of masks (1 per color) and stack them
    target_masks = []
    style_masks = []
    for i in range(n_colors):
        target_masks.append( (target_labels == i).astype(np.float32) )
        style_masks.append( (style_labels == i).astype(np.float32) )
    return np.stack(target_masks), np.stack(style_masks)





def write_image(path, img):   # postprocess and write
    img = img + [123.68, 116.779, 103.939]
    img = img[0]
    img = np.clip(img, 0, 255).astype('uint8')
    imageio.imwrite(path, img)



def get_init_image(content_img, init_noise_ratio):
    """ Produces the initial image which is a linear combination between the content image and noise.
    
    Arguments:
        - content_img {nd array}: Content image as a numpy array.
        - init_noise_ratio {float}: Strength of noise.
    
    Returns:
        {nd array}: A linear combination between the content image and nosie.
    """
    noise_img = np.random.uniform(-20., 20., content_img.shape).astype(np.float32)
    init_img = init_noise_ratio * noise_img + (1. - init_noise_ratio) * content_img
    return init_img



'''
    compute features & masks 
    build net
'''

def compute_features(vgg_weights, pooling_type, input_img, layers):
    """ Computes the features of an image at specific layers of the network.
    
    Arguments:
        - vgg_weights {unknown}: Weights of the network.
        - pooling_type {str}: Can be either 'avg' or 'max' (default: 'avg')
        - input_img {nd array}: Input image.
        - layers {list(str)}: List containing the layers to consider.
    
    Returns:   
        {dict}: Features at each of the layers.
    """
    input = tf.placeholder(tf.float32, shape=input_img.shape)
    net = Model.build_image_net(input, vgg_weights, pooling_type)
    features = {}
    with tf.Session() as sess:
        for layer in layers:
            features[layer] = sess.run(net[layer], feed_dict={input: input_img})
    return features



def compute_layer_masks(masks, layers, ds_type):
    masks_tf = masks.transpose([1,2,0]) # [numberOfMasks, h, w] -> [h, w, masks]
    masks_tf = masks_tf[np.newaxis, :, :, :] # -> [1, h, w, masks]

    input = tf.placeholder(tf.float32, shape=masks_tf.shape)
    net = Model.build_mask_net(input, ds_type) # only do pooling, so no intervention between masks
    layer_masks = {}
    with tf.Session() as sess:
        for layer in layers:
            out = sess.run(net[layer], feed_dict={input: masks_tf})
            layer_masks[layer] = out[0].transpose([2,0,1])
    return layer_masks



def build_target_net(vgg_weights, pooling_type, target_shape):
    input = tf.Variable( np.zeros(target_shape).astype('float32') )
    net = Model.build_image_net(input, vgg_weights, pooling_type)
    net['input'] = input
    return net


'''
    loss
'''

def content_layer_loss(p, x, loss_norm):
    _, h, w, d = p.shape
    M = h * w
    N = d
    if loss_norm  == 1:
        K = 1. / (N * M)
    elif loss_norm == 2:
        K = 1. / (2. * N**0.5 * M**0.5)
    loss = K * tf.reduce_sum( tf.pow((x - p), 2) )
    return loss 



def sum_content_loss(target_net, content_features, layers, layers_weights, loss_norm):
    content_loss = 0.
    for layer, weight in zip(layers, layers_weights):
        p = content_features[layer]
        x = target_net[layer]
        content_loss += content_layer_loss(p, x, loss_norm) * weight
    content_loss /= float(sum(layers_weights))
    return content_loss



def masked_gram(x, mx, mask_norm, N):
    R = mx.shape[0]
    M = mx.shape[1] * mx.shape[2]

    # TODO: use local variable?
    mx = mx.reshape([R, M])
    x = tf.reshape(x, [M, N])
    x = tf.transpose(x) # N * M
    masked_gram = []
    for i in range(R):
        mask = mx[i]
        masked_x = x * mask
        if mask_norm == 'square_sum':
            K = 1. / np.sum(mask**2)
        elif mask_norm == 'sum':
            K = 1. / np.sum(mask)
        gram = K * tf.matmul(masked_x, tf.transpose(masked_x))
        masked_gram.append(gram)
    return tf.stack(masked_gram)



def masked_style_layer_loss(a, ma, x, mx, mask_norm):
    N = a.shape[3]
    R = ma.shape[0]
    K = 1. / (4. * N**2 * R)
    A = masked_gram(a, ma, mask_norm, N)
    G = masked_gram(x, mx, mask_norm, N)
    loss = K * tf.reduce_sum( tf.pow((G - A), 2) )
    return loss



def sum_masked_style_loss(target_net, style_features, target_masks, style_masks, layers, layers_weights, mask_norm):
    style_loss = 0.
    for layer, weight in zip(layers, layers_weights):
        a = style_features[layer]
        ma = style_masks[layer]
        x = target_net[layer]
        mx = target_masks[layer]
        style_loss += masked_style_layer_loss(a, ma, x, mx, mask_norm) * weight
    style_loss /= float(sum(layers_weights))
    return style_loss



def gram_matrix(x): 
    _, h, w, d = x.get_shape() # x is a tensor
    M = h.value * w.value
    N = d.value    
    F = tf.reshape(x, (M, N))
    G = tf.matmul(tf.transpose(F), F)
    return (1./M) * G



def style_layer_loss(a, x):
    N = a.shape[3]
    A = gram_matrix(tf.convert_to_tensor(a))
    G = gram_matrix(x)
    loss = (1./(4 * N**2 )) * tf.reduce_sum(tf.pow((G - A), 2))
    return loss



def sum_style_loss(target_net, style_features, layers, layers_weights): # for testing  
    style_loss = 0.
    for layer, weight in zip(layers, layers_weights):
        a = style_features[layer]
        x = target_net[layer]
        style_loss += style_layer_loss(a, x) * weight
    style_loss /= float(sum(layers_weights))
    return style_loss  



def sum_total_variation_loss(input, shape):
    b, h, w, d = shape
    x = input
    tv_y_size = b * (h-1) * w * d
    tv_x_size = b * h * (w-1) * d
    loss_y = tf.nn.l2_loss(x[:,1:,:,:] - x[:,:-1,:,:])  # l2_loss() use 1/2 factor
    loss_y /= tv_y_size
    loss_x = tf.nn.l2_loss(x[:,:,1:,:] - x[:,:,:-1,:]) 
    loss_x /= tv_x_size
    loss = 2 * (loss_y + loss_x)
    loss = tf.cast(loss, tf.float32) # ?
    return loss