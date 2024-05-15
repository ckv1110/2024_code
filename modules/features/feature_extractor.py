import histomicstk as htk
import tifffile as tiff
import skimage.io
import skimage.measure
import skimage.color
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
from stardist.models import StarDist2D
from csbdeep.utils import normalize
import gc
import os
from skimage.color import rgb2hed, hed2rgb, separate_stains, combine_stains

def load_image(image_path, as_tf=True):
    """
    Load a TIFF image using tifffile and convert it to a tensorflow array.

    Parameters:
    - image_path: Path to the TIFF image file.
    - as_tf: If True, the image will be converted to a tensorflow array. If False, the image will be returned as a numpy array.

    Returns:
    A tensforflow array representing the image.
    """
    with tiff.TiffFile(image_path) as tif:
        image_array = tif.asarray()

    if as_tf:
        image_array_tf = tf.convert_to_tensor(image_array, dtype=tf.float32)
        return image_array_tf
    else:
        return image_array

def calculate_centroids_tf_labels(image_tensor):
    """
    Calculates the centroids of unique labels in an image tensor.

    Args:
        image_tensor (tf.Tensor): The input image tensor.

    Returns:
        List: A list of tuples containing the label, centroid x-coordinate, and centroid y-coordinate.
            The list is sorted by label.

    """
    unique_labels = tf.unique(tf.reshape(image_tensor, [-1])).y
    unique_labels = unique_labels[unique_labels != 0]
    centroids = []
    for label in tqdm(unique_labels, desc='Calculating centroids with labels attached'):
        positions = tf.where(tf.equal(image_tensor, label))
        centroid_x = tf.reduce_mean(tf.cast(positions[:, 1], tf.float32))
        centroid_y = tf.reduce_mean(tf.cast(positions[:, 0], tf.float32))
        # Attach label to centroid for later sorting
        centroids.append((label.numpy(), centroid_x.numpy(), centroid_y.numpy()))
    # Sort centroids by label
    centroids_sorted = sorted(centroids, key=lambda x: x[0])
    return centroids_sorted

def get_deconvolved_stains(im_input, stain_colour_map):
    """
    Perform color deconvolution on an input image using stain colour map.

    Parameters:
    - im_input (numpy.ndarray): The input image to be deconvolved.
    - stain_colour_map (dict): A dictionary containing stain colour values. Format is as follows: {'hematoxylin': [R, G, B], 'eosin': [R, G, B], 'null': [R, G, B]}.
        - Recommend QuPath to estimate stain vectors and use adjust the colour values accordingly.
    Returns:
    - im_nuclei_stain (numpy.ndarray): The deconvolved image for nuclei stain.
    - im_ecm_stain (numpy.ndarray): The deconvolved image for extracellular matrix (ECM) stain.
    """

    # Create stain matrix
    W = np.array([stain_colour_map['hematoxylin'],
                  stain_colour_map['eosin'],
                  stain_colour_map['null']]).T
    
    print('Performing color deconvolution...')
    # Perform standard color deconvolution
    im_stains = htk.preprocessing.color_deconvolution.color_deconvolution(im_input, W).Stains  
    im_nuclei_stain = im_stains[:, :, 0]
    im_ecm_stain = im_stains[:, :, 1]
    print('Completed color deconvolution.')

    return im_nuclei_stain, im_ecm_stain

def deconvolve_scikit(im_input, stain_colour_map):
    # im_hed = rgb2hed(im_input)
    W = np.array([stain_colour_map['hematoxylin'],
                  stain_colour_map['eosin'],
                  stain_colour_map['null']])

    im_hed = separate_stains(im_input, W)
    null = np.zeros_like(im_hed[:, :, 0])
    print('Performing color deconvolution...')
    im_h = hed2rgb(np.stack((im_hed[:, :, 0], null, null), axis=-1))
    im_e = hed2rgb(np.stack((null, im_hed[:, :, 1], null), axis=-1))
    im_d = hed2rgb(np.stack((null, null, im_hed[:, :, 2]), axis=-1))
    print('Completed color deconvolution.')
    return im_h, im_e, im_d

def get_nuclei_features(nuclei_stain_im, nuclei_seg_mask, verbose=False, features=None):
    """
    Extracts features from nuclei stain image and nuclei segmentation mask and saves them as .csv files.

    Args:
        nuclei_stain_im (numpy.ndarray): Nuclei stain image.
        nuclei_seg_mask (numpy.ndarray): Nuclei segmentation mask.
        save_path (str): Path to save the extracted features.
        file_name (str, optional): Prefix for the saved feature files. Defaults to 'Experiment'.
        verbose (bool, optional): If True, prints the number of nuclei detected. Defaults to False.

    Returns:
        The extracted features as a pandas DataFrame.
    """

    # Get nuclei count
    nuclei_count = skimage.measure.regionprops(nuclei_seg_mask)
    if verbose:
        print(f'Number of nuclei detected: {len(nuclei_count)}')
    
    # Compute nuclei features
    if features == 'all':
        print('Computing all features...')
        fsd_f = htk.features.compute_fsd_features(nuclei_seg_mask)
        grad_f = htk.features.compute_gradient_features(nuclei_seg_mask, nuclei_stain_im)
        morph_f = htk.features.compute_morphometry_features(nuclei_seg_mask)
        hara_f = htk.features.compute_haralick_features(nuclei_seg_mask, nuclei_stain_im)
        # Concatenate the dataframes
        print('Completed computing features')
        features = pd.concat([hara_f, fsd_f, grad_f, morph_f], axis=1)
        # os.makedirs(save_path, exist_ok=True)
        # features.insert(0, 'Nuclei_ID', range(1, 1 + len(features)))
        # features.to_csv(f"{save_path}/{file_name}-features.csv", index=False)
        return features
    elif features == 'fsd':
        print('Computing Fourier shape descriptors...')
        fsd_f = htk.features.compute_fsd_features(nuclei_seg_mask)
        return fsd_f
    elif features == 'grad':
        print('Computing gradient features...')
        grad_f = htk.features.compute_gradient_features(nuclei_seg_mask, nuclei_stain_im)
        return grad_f
    elif features == 'morph':
        print('Computing morphometry features...')
        morph_f = htk.features.compute_morphometry_features(nuclei_seg_mask)
        return morph_f
    elif features == 'hara':
        print('Computing Haralick features...')
        hara_f = htk.features.compute_haralick_features(nuclei_seg_mask, nuclei_stain_im)
        return hara_f

def segment_nuclei(NAME,
                   model_type, im_array, 
                   axes='YXC', 
                   block_size=2000, min_overlap=300, context=300, 
                   prob_thresh=0.5, nms_thresh=0.8):
    """
    Segment nuclei in an image using a pretrained StarDist2D model.

    Parameters:
    - model_type (str): The type of pretrained model to use.
    - im_array (numpy.ndarray): The input image array.
    - axes (str): The order of axes in the input image array. Default is 'YXC'.
    - block_size (int): The size of the blocks used for prediction. Default is 2000.
    - min_overlap (int): The minimum overlap between blocks. Default is 300.
    - context (int): The context size used for prediction. Default is 300.
    - prob_thresh (float): The probability threshold for classifying pixels as nuclei. Default is 0.5.
    - nms_thresh (float): The non-maximum suppression threshold. Default is 0.8.

    Returns:\n
    (Saves the segmented nuclei labels as a TIFF file, and also returns the labels.)
    - labels (numpy.ndarray): The segmented nuclei labels.
    """

    model = StarDist2D.from_pretrained(model_type)
    labels, _  = model.predict_instances_big(
        normalize(im_array,0.1,99.8), 
        axes=axes, 
        block_size=block_size, min_overlap=min_overlap, context=context, 
        prob_thresh=prob_thresh, nms_thresh=nms_thresh
        )
    del _
    gc.collect()
    os.makedirs(f'./{NAME}/processed/', exist_ok=True)
    tiff.imwrite(f'./{NAME}/processed/{NAME}_StarDist2D_prediction.tif', labels.astype('uint16'))
    return labels