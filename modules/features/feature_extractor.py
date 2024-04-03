import histomicstk as htk
import tifffile as tiff
import glob
import skimage.io
import skimage.measure
import skimage.color
import numpy as np
import pandas as pd
import os
from multiprocessing import Pool

# img_path = glob.glob('./example_img/*HnE*.tiff')[0]
# img = tiff.TiffReader(img_path)
# img_arr = img.asarray()

# plt.imshow(img_arr)
# _ = plt.title('Input Image', fontsize=16)

def get_centroids(nuclei_seg_mask):
    print('Computing nuclei centroids...')
    n_colours = np.max(nuclei_seg_mask)

    centroids = {tuple(np.mean(coords) for coords in np.where(nuclei_seg_mask.T == i)) 
                 for i in range(1, n_colours + 1)
                 }

    print('Completed computing nuclei centroids.')
    return centroids

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

def get_nuclei_features(nuclei_stain_im, nuclei_seg_mask, save_path, file_name='Experiment', verbose=False):
    """
    Extracts features from nuclei stain image and nuclei segmentation mask and saves them as .csv files.

    Args:
        nuclei_stain_im (numpy.ndarray): Nuclei stain image.
        nuclei_seg_mask (numpy.ndarray): Nuclei segmentation mask.
        save_path (str): Path to save the extracted features.
        file_name (str, optional): Prefix for the saved feature files. Defaults to 'Experiment'.
        verbose (bool, optional): If True, prints the number of nuclei detected. Defaults to False.

    Returns:
        None -> Saves the extracted features as a .csv file.
    """

    # Get nuclei count
    nuclei_count = skimage.measure.regionprops(nuclei_seg_mask)
    if verbose:
        print(f'Number of nuclei detected: {len(nuclei_count)}')
    
    # Compute nuclei features
    print('Computing features...')
    fsd_f = htk.features.compute_fsd_features(nuclei_seg_mask)
    grad_f = htk.features.compute_gradient_features(nuclei_seg_mask, nuclei_stain_im)
    morph_f = htk.features.compute_morphometry_features(nuclei_seg_mask)
    hara_f = htk.features.compute_haralick_features(nuclei_seg_mask, nuclei_stain_im)

    # fsd_f.to_csv(save_path + file_name + "-fourier_shape_descriptors.csv")
    # grad_f.to_csv(save_path + file_name + "-gradient.csv")
    # morph_f.to_csv(save_path + file_name + "-morphometry.csv")
    # hara_f.to_csv(save_path + file_name + "-haralick.csv")

    # Concatenate the dataframes
    print('Completed computing features.\nSaving features...')
    features = pd.concat([fsd_f, grad_f, morph_f, hara_f], axis=1)
    os.makedirs(save_path, exist_ok=True)
    features.insert(0, 'Nuclei_ID', range(1, 1 + len(features)))
    features.to_csv(f"{save_path}/{file_name}-features.csv", index=False)
    return features