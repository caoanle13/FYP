import numpy as np
import cv2
from PIL import Image

def match_histogram(source_img_path, reference_image):

    source_image = cv2.imread(source_img_path)
    reference_image = cv2.imread(reference_image)

    source_B, source_G, source_R = np.split(source_image, 3, axis=2)
    source_R = source_R[:,:,0]
    source_G = source_G[:,:,0]
    source_B = source_B[:,:,0]

    reference_B, reference_G, reference_R = np.split(reference_image, 3, axis=2)
    reference_R = reference_R[:,:,0]
    reference_G = reference_G[:,:,0]
    reference_B = reference_B[:,:,0]

    #  save original shape
    orig_shape = source_R.shape

    # Unravel into 1d array
    source_R = source_R.ravel()
    source_G = source_G.ravel()
    source_B = source_B.ravel()

    reference_R = reference_R.ravel()
    reference_G = reference_G.ravel()
    reference_B = reference_B.ravel()


    # Get the set of unique pixel values and their corresponding indices and counts
    # for source
    s_values_R, s_idx_R, s_counts_R = np.unique(
        source_R, return_inverse=True, return_counts=True)

    s_values_G, s_idx_G, s_counts_G = np.unique(
        source_G, return_inverse=True, return_counts=True)

    s_values_B, s_idx_B, s_counts_B = np.unique(
        source_B, return_inverse=True, return_counts=True)

    # Same for reference (without r_idx)
    r_values_R, r_counts_R = np.unique(reference_R, return_counts=True)
    r_values_G, r_counts_G = np.unique(reference_G, return_counts=True)
    r_values_B, r_counts_B = np.unique(reference_B, return_counts=True)

    # Now we need to calculate the empirical cumulative distribuition, scaled 0 to
    # 1. Each quantiles tells us, for each unique value, what proportion of the data
    # fall at or below that value.
    s_quantiles_R = np.cumsum(s_counts_R).astype(np.float64) / source_R.size
    s_quantiles_G = np.cumsum(s_counts_G).astype(np.float64) / source_G.size
    s_quantiles_B = np.cumsum(s_counts_B).astype(np.float64) / source_B.size

    r_quantiles_R = np.cumsum(r_counts_R).astype(np.float64) / reference_R.size
    r_quantiles_G = np.cumsum(r_counts_G).astype(np.float64) / reference_G.size
    r_quantiles_B = np.cumsum(r_counts_B).astype(np.float64) / reference_B.size

    # The value of the source CDF at each unique input is mapped to the reference CDF 
    # The reference cdf value is then used to look up the corresponding reference
    # value (interpolated between values if the match isn't exact).
    interp_r_values_R = np.interp(s_quantiles_R, r_quantiles_R, r_values_R)
    interp_r_values_G = np.interp(s_quantiles_G, r_quantiles_G, r_values_G)
    interp_r_values_B = np.interp(s_quantiles_B, r_quantiles_B, r_values_B)

    #Now we can take our s_idx to recreate an array of the same size as the source
    #but with new values
    target_R = interp_r_values_R[s_idx_R].reshape(orig_shape)
    target_G = interp_r_values_G[s_idx_G].reshape(orig_shape)
    target_B = interp_r_values_B[s_idx_B].reshape(orig_shape)

    target_image = np.dstack((target_R, target_G, target_B))


    pil_image = Image.fromarray(target_image.astype('uint8'))

    return pil_image