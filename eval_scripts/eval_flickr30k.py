import os
import json
import argparse
import re
import random
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import tempfile
import nltk
from pycocoevalcap.tokenizer import ptbtokenizer

from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser
from minigpt4.conversation.conversation import CONV_VISION_minigptv2
from minigpt4.common.config import Config


class Flickr30kEvalDataset(Dataset):
    def __init__(self, annotations, vis_processor, img_path):
        self.annotations = annotations
        self.img_path = img_path
        self.vis_processor = vis_processor
        self.prompt = "Describe the content of this image in detail."

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        item = self.annotations[idx]
        image_id_str = item['image_id']
        image_file = image_id_str + '.jpg'
        image_path = os.path.join(self.img_path, image_file)
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        
        return {
            "image": image,
            "image_id": int(image_id_str),
            "prompt": self.prompt,
        }

class PythonTokenizer:
    """
    Wrapper for NLTK's TreebankWordTokenizer.
    """
    def __init__(self):
        self.tokenizer = nltk.tokenize.TreebankWordTokenizer()

    def tokenize(self, captions_for_image):
        tokenized_captions = {}
        for image_id, captions in captions_for_image.items():
            tokenized_captions[image_id] = [
                ' '.join(self.tokenizer.tokenize(c['caption'].lower()))
                for c in captions
            ]
        return tokenized_captions


def convert_to_coco_format(annotations):
    coco_images = []
    coco_annotations = []
    ann_id = 0
    
    for ann in annotations:
        image_id_str = ann['image_id']
        image_id_int = int(image_id_str)
        
        # Add image info
        coco_images.append({
            "id": image_id_int,
            "file_name": image_id_str + '.jpg'
        })
        
        # Split grounded captions into individual sentences
        captions = [s.strip() for s in ann['grounded_caption'].split('.') if s.strip()]
        
        for caption in captions:
            coco_annotations.append({
                "image_id": image_id_int,
                "id": ann_id,
                "caption": caption + '.' # Add period back
            })
            ann_id += 1
            
    return {
        "images": coco_images,
        "annotations": coco_annotations,
        "info": {},
        "licenses": [],
    }

def post_handle(captions):
    for i in range(len(captions)):
        caption = captions[i]
        if '### Assistant:' in caption:
            caption = caption.split('### Assistant:')[-1].strip()
        
        if '[INST] <Img>' in caption:
            caption = caption.split('[INST] <Img>')[-1].strip()
        
        if '[INST] [Img]' in caption:
            caption = caption.split('[INST] [Img]')[-1].strip()

        if '[INST] [Image]' in caption:
            caption = caption.split('[INST] [Image]')[-1].strip()

        if '[INST] <p>' in caption:
            caption = caption.split('[INST] <p>')[-1].strip()

        if '<DESC>' in caption:
            caption = caption.split('<DESC>')[-1].strip()
        
        if '[INST]' in caption:
            caption = caption.split('[INST]')[-1].strip()

        captions[i] = caption.strip()
    
    return captions


def main():
    parser = eval_parser()
    parser.add_argument("--save_path", type=str, default='output/flickr30k_eval', help="path to save the evaluation results")
    args = parser.parse_args()
    cfg = Config(args)

    model, vis_processor = init_model(args)
    conv_temp = CONV_VISION_minigptv2.copy()
    conv_temp.system = ""
    model.eval()
    
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    eval_file_path = cfg.evaluation_datasets_cfg["flickr30k"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["flickr30k"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["flickr30k"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["flickr30k"]["max_new_tokens"]

    with open(eval_file_path) as f:
        annotations = json.load(f)

    random.shuffle(annotations)
    annotations = annotations[:len(annotations) // 20]

    # Convert original annotations to COCO format for ground truth
    coco_formatted_gt = convert_to_coco_format(annotations)
    
    # Save COCO formatted ground truth to a temporary file
    temp_gt_file = tempfile.NamedTemporaryFile(mode='w', delete=False, dir=save_path, suffix=".json")
    json.dump(coco_formatted_gt, temp_gt_file)
    temp_gt_file.close()

    dataset = Flickr30kEvalDataset(annotations, vis_processor, img_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    results = []
    current_sample_idx = 0
    for batch in tqdm(dataloader, desc="Evaluating Batches"):
        images = batch["image"].to(model.device)
        prompts = batch["prompt"]
        image_ids = batch["image_id"]
        
        texts = prepare_texts(prompts, conv_temp)
        
        with torch.no_grad():
            captions = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
        captions = post_handle(captions)
        
        for i in range(len(captions)):
            results.append({"image_id": image_ids[i].item(), "caption": captions[i]})
            print("Model Output: ", captions[i])
            print("Ground Truth: ", annotations[current_sample_idx]['grounded_caption'])
            print("-" * 100)
            current_sample_idx += 1

    # Save results to a temporary file
    temp_res_file = tempfile.NamedTemporaryFile(mode='w', delete=False, dir=save_path, suffix=".json")
    json.dump(results, temp_res_file)
    temp_res_file.close()

    # Create COCO object for ground truth
    coco_gt = COCO(temp_gt_file.name)
    
    # Create COCO object for results
    coco_res = coco_gt.loadRes(temp_res_file.name)
    
    # Create COCOEvalCap object
    coco_eval = COCOEvalCap(coco_gt, coco_res)

    # Use Python-based tokenizer
    coco_eval.tokenizer = PythonTokenizer()
    
    # Evaluate
    coco_eval.evaluate()
    
    # Print and save results
    eval_results = coco_eval.eval
    print("Evaluation Results:")
    for metric, score in eval_results.items():
        print(f'{metric}: {score:.4f}')

    results_file = os.path.join(save_path, "flickr30k_results.json")
    with open(results_file, 'w') as f:
        json.dump({'results': results, 'metrics': eval_results}, f, indent=4)
    print(f"Results and metrics saved to {results_file}")
    
    os.remove(temp_res_file.name)
    os.remove(temp_gt_file.name)

if __name__ == "__main__":
    main()
