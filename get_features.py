import glob
from modules.features import load_image, get_deconvolved_stains, get_nuclei_features, calculate_centroids_tf_labels
import os
import gc
import pandas as pd
import tensorflow as tf

if __name__ == '__main__':
    exp = 'T19'
    save_path = f'./example_img/{exp}_feature_set'
    nuc_seg_path = glob.glob(f'./example_img/*{exp}*prediction*.tif')[0]
    os.makedirs(save_path, exist_ok=True)

    nuclei_seg_tf = load_image(nuc_seg_path)
    with tf.device('/GPU:0'):  # This specifies the first GPU
        # Operations here will be executed on the GPU if available
        centroids = calculate_centroids_tf_labels(nuclei_seg_tf)
    centroids_df = pd.DataFrame(centroids, columns=['Nuclei_ID','X_centroid','Y_centroid'])
    del nuclei_seg_tf
    centroids_df.to_csv(f"{save_path}/{exp}_centroids.csv", index=False)
    del centroids_df
    gc.collect()

    
    im_path = glob.glob(f'./example_img/*{exp}*HnE*.tiff')[0]
    hne_im = load_image(im_path, as_tf=False)

    stain_colour_map = {'hematoxylin': [0.52, 0.74, 0.42],
                        'eosin':       [0.27, 0.71, 0.65],
                        'dab':         [0.27, 0.57, 0.78],
                        'null':        [0.00, 0.00, 0.00],
                        }

    nuc_stain_im, _ = get_deconvolved_stains(hne_im, stain_colour_map)
    del hne_im
    gc.collect()

    nuclei_seg_im = load_image(nuc_seg_path, as_tf=False)
    features_df = get_nuclei_features(nuc_stain_im, nuclei_seg_im, verbose=True)
    features_df.to_csv(f"{save_path}/{exp}_features.csv", index=False)

    # final_df = pd.concat([centroids_df, features_df], axis=1)
    # os.makedirs(save_path, exist_ok=True)
    # final_df.to_csv(f"{save_path}/{exp}_features.csv", index=False)