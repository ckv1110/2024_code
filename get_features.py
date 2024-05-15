import glob
from modules.features import load_image, get_deconvolved_stains, get_nuclei_features, calculate_centroids_tf_labels
import os
import gc
import pandas as pd
import tensorflow as tf
import time

if __name__ == '__main__':
    NAME = "T19"
    PATH = os.getcwd()
    FEATURE = "hara"

    start_time = time.time()
    nuc_seg_path = glob.glob(f"{PATH}/{NAME}/processed/*{NAME}*prediction.tif*")[0]
    # nuc_stain_path = glob.glob(f"{PATH}/{NAME}/processed/*{NAME}*Hematoxylin*.tif*")[0]

    if FEATURE=='fsd' or FEATURE=='morph':
        nuclei_seg_im = load_image(nuc_seg_path, as_tf=False)
        nuc_stain_im = None
        os.makedirs(f"{PATH}/{NAME}/results", exist_ok=True)
        features_df = get_nuclei_features(nuc_stain_im, nuclei_seg_im, verbose=True, features=FEATURE)
        features_df.to_csv(f"{PATH}/{NAME}/results/{NAME}_{FEATURE}_features.csv", index=False)
    else:
        nuclei_seg_im = load_image(nuc_seg_path, as_tf=False)
        print('Loading HnE image...')
        im_path = glob.glob(f'{PATH}/{NAME}/raw/*{NAME}*HnE*.tif*')[0]
        hne_im = load_image(im_path, as_tf=False)

        stain_colour_map = {'hematoxylin': [0.52, 0.74, 0.42],
                            'eosin':       [0.27, 0.71, 0.65],
                            'dab':         [0.27, 0.57, 0.78],
                            'null':        [0.00, 0.00, 0.00],
                            }

        nuc_stain_im, _ = get_deconvolved_stains(hne_im, stain_colour_map)
        del hne_im, _
        gc.collect()
        # nuc_stain_im = load_image(nuc_stain_path, as_tf=False)

        os.makedirs(f"{PATH}/{NAME}/results", exist_ok=True)
        features_df = get_nuclei_features(nuc_stain_im, nuclei_seg_im, verbose=True, features=FEATURE)
        features_df.to_csv(f"{PATH}/{NAME}/results/{NAME}_{FEATURE}_features.csv", index=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    # final_df = pd.concat([centroids_df, features_df], axis=1)
    # os.makedirs(save_path, exist_ok=True)
    # final_df.to_csv(f"{save_path}/{exp}_features.csv", index=False)