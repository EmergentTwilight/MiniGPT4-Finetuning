import pandas as pd
import json
import os
from tqdm import tqdm

def prepare_flickr30k_annotations(flickr30k_path='flickr30k'):
    """
    Prepares the Flickr30k annotations for the GroundedDetailDataset.

    This function reads the original flickr_annotations_30k.csv, processes it,
    and saves a JSON file in the format expected by the GroundedDetailDataset.
    The script assumes the dataset is downloaded in the 'flickr30k' directory
    in the project root.

    The output is a JSON file named 'flickr30k_grounded_detail.json' saved
    in the same directory.
    """
    csv_path = os.path.join(flickr30k_path, 'flickr_annotations_30k.csv')
    output_path = os.path.join(flickr30k_path, 'flickr30k_grounded_detail.json')
    images_path = os.path.join(flickr30k_path, 'flickr30k-images')

    print(f"Reading annotations from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        print("Please make sure you have downloaded the Flickr30k dataset and it is located in the 'flickr30k' directory.")
        return

    annotations = []
    print("Processing annotations...")
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        filename = row['filename']
        # Check if the image file exists
        if not os.path.exists(os.path.join(images_path, filename)):
            continue

        captions = json.loads(row['raw'])
        img_id = filename.split('.')[0]
        
        # We use all captions concatenated as a single detailed description.
        grounded_caption = " ".join(captions)

        annotations.append({
            'image_id': img_id,
            'grounded_caption': grounded_caption,
        })

    print(f"Saving processed annotations to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(annotations, f, indent=4)
    
    print("Done.")
    print(f"A total of {len(annotations)} annotations have been processed.")
    print(f"You can now use '{output_path}' for training.")


if __name__ == '__main__':
    # Assuming the script is run from the root of the MiniGPT4-Finetuning project,
    # and the 'flickr30k' directory is at the same level.
    prepare_flickr30k_annotations('flickr30k') 