import math
import os
import argparse
import json
import torch
import transformers
import numpy as np
from tqdm import tqdm
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN
from videollava.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from videollava.model.builder import load_pretrained_model
from torch.utils.data import Dataset, DataLoader
import random
import wandb
from verifier import Verifier

def get_args_parser():
    parser = argparse.ArgumentParser('Video-LLaVA reasoning generate', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--save_interval', default=100, type=int)
    parser.add_argument('--model_path', default='LanguageBind/Video-LLaVA-7B', type=str)
    parser.add_argument('--model_base', default=None, type=str)
    parser.add_argument('--reasoning_dir', default="data/STaR", type=str)
    parser.add_argument('--video_dir', default="data", type=str)
    parser.add_argument('--dataset', default="Kinetics700", type=str)
    parser.add_argument('--save_dir', default='tmp', type=str, help='save directory')
    parser.add_argument('--save_file', default='lmm_reasoning.json', type=str, help='Index of the chunk to process')
    parser.add_argument('--init_file', default='lmm_reasoning.json', type=str, help='save directory')
    parser.add_argument('--chunk_inference', default=False, action='store_true')
    parser.add_argument('--num_chunks', default=1, type=int, help='Number of videos per chunk')
    parser.add_argument('--chunk_index', default=0, type=int, help='Index of the chunk to process')
    parser.add_argument('--video_list', default='train.txt', type=str, help='name of file with list of videos to process')
    parser.add_argument('--rationalize', default=True, action='store_true', help='whether to rationalize')
    parser.add_argument('--device', default='cuda', type=str)
    return parser

def label_dict_to_string(label_dict):
    """
    Converts any label dictionary into a descriptive string for use with large language models.
    
    Args:
    label_dict (dict): A dictionary where keys represent different types of annotations and
                       values are the corresponding annotation data, which can be lists, booleans, etc.

    Returns:
    str: A descriptive string representing the label dictionary.
    """
    label_strings = []
    for key, values in label_dict.items():
        if isinstance(values, list):
            # Convert list values into a comma-separated string
            values_str = ", ".join(str(v) for v in values)
            description = f"{key}: {values_str}"
        elif isinstance(values, (str, int, float, bool)):
            # Directly use the value if it's a single primitive type
            description = f"{key}: {values}"
        else:
            # Handle other types of data structures or custom objects
            description = f"{key}: [Complex Data]"
        
        label_strings.append(description)

    # Join all descriptions into a single string
    return ". ".join(label_strings).replace('_', ' ')

class ReasoningInferenceDataset(Dataset):
    def __init__(self, train_file, video_dir, video_processor):
        self.video_dir = video_dir
        self.train_file = train_file
        self.video_processor = video_processor

    def __len__(self):
        return len(self.train_file)

    def __getitem__(self, idx):
        item = self.train_file[idx]
        video_name = os.path.join(self.video_dir, item['video_path'])
        try:
            video_tensor = self.video_processor(video_name, return_tensors='pt')['pixel_values'][0].half() 
        except:
            video_tensor = torch.tensor([0])
            item['question'] = "failed"
            
        return item, video_tensor

def reasoning_collate_fn(batch):
    # Separate items and tensors
    items = [item for item, tensor in batch]
    tensor_list = [tensor for item, tensor in batch]

    # Handle the case where the video tensor might be of different sizes
    # by using torch.nn.utils.rnn.pad_sequence if necessary
    # For now, we assume all tensors are the same size and can be stacked directly
    try:
        video_tensors = torch.stack(tensor_list)
    except RuntimeError:
        # If tensors are not of the same size, you must decide how to handle this.
        # Here we pad tensors to the maximum tensor length in the batch
        max_size = max([t.size(1) for t in tensor_list])  # Assuming time dimension is dimension 1
        padded_tensors = [torch.nn.functional.pad(t, (0, max_size - t.size(1))) for t in tensor_list]
        video_tensors = torch.stack(padded_tensors)

    # Return a dictionary or any other structure that suits your needs
    return items, video_tensors

class VideoInferenceProcessor:
    def __init__(self, save_dir, save_file, save_interval, rationale_prompt, rationalization_prompt, rationalize, device):
        self.rationale_prompt = rationale_prompt
        self.rationalization_prompt = rationalization_prompt
        self.save_dir = save_dir
        self.save_file = save_file
        self.save_interval = save_interval
        self.device = device
        self.rationalize = rationalize
        self.conv_template = conv_templates["llava_v1"].copy()
        self.verifier = Verifier()
        
    def get_model_output(self, model, tokenizer, video_tensor, qs):
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_VID_START_TOKEN + ''.join([DEFAULT_IMAGE_TOKEN]*8) + DEFAULT_VID_END_TOKEN + '\n' + qs
        else:
            qs = ''.join([DEFAULT_IMAGE_TOKEN]*8) + '\n' + qs   
            
        conv = self.conv_template.copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=[video_tensor],
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
            
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
            
        outputs = outputs.strip()
        return outputs


    def run(self, model, tokenizer, dataloader, outputs, wandb):
        for i, (annots, video_tensors) in enumerate(tqdm(dataloader)):
            try:
                for j, ann in enumerate(annots):
                    rationale = True
                    if ann['question'] == "failed":
                        print(ann['video_path'] + " has failed")
                        #import ipdb; ipdb.set_trace()
                        continue
     
                    video_tensor = video_tensors[j].to(self.device)
                    text_prompt = self.rationale_prompt(ann['question'])
                    wandb.log({"sample":i, "rationale":rationale,  "text_prompt": len(text_prompt)})
                    ann['output'] = self.get_model_output(model, tokenizer, video_tensor, text_prompt)
                    
                    correct_cls = self.verifier.verify(ann)
                    if not correct_cls and self.rationalize:
                        rationale = False
                        text_prompt = self.rationalization_prompt(ann['question'], label_dict_to_string(ann['label']))
                        wandb.log({"sample": i, "rationale": rationale, "text_prompt": len(text_prompt)})
                        ann['output'] = self.get_model_output(model, tokenizer, video_tensor, text_prompt)
                        correct_cls = self.verifier.verify(ann)
    
                    conversations = [{'from': 'human', 'value': '<video>\n' + ann['question']}, {'from': 'gpt', 'value': ann['output']}]
                    ann.update({"conversations": conversations, "rationale": rationale, "cls_correct": correct_cls})
                    outputs.append(ann)
            except:
                print("failed for some reason")

            if (i + 1) % self.save_interval == 0:
                #import ipdb; ipdb.set_trace()
                with open(os.path.join(self.save_dir, self.save_file), 'w') as file:
                    json.dump(outputs, file, indent=4)
            
        return outputs


def main(args):
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # If using multi-GPU
    
    wandb.init(project='video-star-generation', config=args)
    save_dir = os.path.join(args.reasoning_dir, args.dataset, args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    print("Generating reasoning")
    outputs = []

    ## Translation of class names to natural text class names
    #with open(os.path.join(args.reasoning_dir, args.dataset, 'star_annotations/train.json'), 'r') as file:
    train_file = os.path.join(args.reasoning_dir, args.dataset, 'star_annotations/train.json')
    print('train_file:', train_file)
    with open(train_file, 'r') as file:
        train = json.load(file)

    #import ipdb; ipdb.set_trace()
    ## Checking if continueing run from init_file
    if os.path.isfile(os.path.join(save_dir, args.init_file)):
        print("existing file found!")
        with open(os.path.join(save_dir, args.init_file), 'r') as file:
            existing_data = json.load(file)
        print("found",len(existing_data), "out of",len(train),"videos processed!")
        
        existing_videos = set([v['question_id'] for v in existing_data])
        print(f"reducting train: {len(train)} by {len(existing_videos)}")
        train = [t for t in train if t['question_id'] not in existing_videos]

    if args.chunk_inference:
        save_file=str(args.chunk_index) + "-" + args.save_file
        total_videos = len(train)
        chunk_size = total_videos//args.num_chunks
        start_index = args.chunk_index * chunk_size
        end_index = min(start_index + chunk_size, total_videos)
        selected_vids = train[start_index:end_index]
        print(start_index, end_index)
    else:
        save_file=args.save_file
        selected_vids=train
        
    print("number of videos:", len(selected_vids))
    random.shuffle(selected_vids)
    
    rationale_prompt = lambda q: f"Question: {q}\nCan you explain step-by-step how one can arrive at this conclusion?"
    rationalization_prompt = lambda q, a: f"Question: {q}\nAnswer: {a}.\nCan you explain step-by-step how one can arrive at this conclusion?"
    
    ## Loading Model
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, _ = load_pretrained_model(args.model_path, None, model_name)
    model = model.to(args.device)
    
    ## Preparing dataset
    dataset = ReasoningInferenceDataset(selected_vids,
                                        args.video_dir,
                                        processor['video'])
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=reasoning_collate_fn)
    
    ## Defining video processor
    video_processor = VideoInferenceProcessor(save_dir,
                                              save_file,
                                              args.save_interval,
                                              rationale_prompt,
                                              rationalization_prompt,
                                              args.rationalize,
                                              args.device)
    
    ## Running actual inference
    outputs = video_processor.run(model, tokenizer, dataloader, outputs, wandb)
    with open(os.path.join(save_dir, save_file), 'w') as file:
        json.dump(outputs, file, indent=4)
    wandb.finish()
    print("######### Inference completed. #########")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Video-LLaVA reasoning generate', parents=[get_args_parser()])
    args = parser.parse_args()
    print(args)
    main(args)