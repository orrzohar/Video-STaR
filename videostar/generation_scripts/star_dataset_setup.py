import os 
import cv2
import json
import random

import shutil
import imageio
import numpy as np
from tqdm import tqdm
from openai import OpenAI
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from videostar.data_gen_utils import *
from videostar.ego_exo_processors import *

from IPython.display import display, Image
from IPython.display import clear_output
import ipywidgets as widgets
import time
from PIL import Image

from decord import VideoReader
from tqdm import tqdm


api_key = ""
gpt_model = "gpt-4"

client = OpenAI(
    # This is the default and can be omitted
    api_key=api_key,
)




## Kinetics

dataset = "data/Kinetics700/anns.json"
with open(dataset, 'r') as file:
    anns = json.load(file)



task = "action recognition"
example = "what is happening in this video?"
kinetics_questioins = generate_questions(num_questions=50, num_classes=20, label_list: list, task:str, example:str)



## STAR-B

dataset = "data/STARB"
train_file = "STAR_train.json"
with open(os.path.join(dataset, train_file), 'r') as file:
    anns = json.load(file)


starb = STARB_dataset()
out = []
for a in tqdm(anns):
    out += starb.process_video(a)



with open('data/STaR/STARB/star_annotations/train.json', 'w') as f:
    json.dump(out, f, indent=4)


dataset = "data/STARB"
train_file = "STAR_val.json"
with open(os.path.join(dataset, train_file), 'r') as file:
    anns = json.load(file)
    
starb = STARB_dataset()

out = []
for a in tqdm(anns):
    out += starb.process_video(a)
    
with open('data/STaR/STARB/star_annotations/test.json', 'w') as f:
    json.dump(out, f, indent=4)



dataset = "data/FineDiving/Annotations/FineDiving_fine-grained_annotation.pkl"
with open(dataset, 'rb') as file:
    anns = pickle.load(file)

with open('data/FineDiving/train_test_split/train_split.pkl', 'rb') as f:
    data = pickle.load(f)
    
train_vid_names = set([d[0] + "_" + str(d[1]) for d in data])

with open('data/FineDiving/train_test_split/test_split.pkl', 'rb') as f:
    data = pickle.load(f)
    
test_vid_names = set([d[0] + "_" + str(d[1]) for d in data])

def dict_to_list_with_key(annotations_dict):
    annotations_list = []
    for key, annotation_data in annotations_dict.items():
        # Create a new dictionary for the current annotation
        annotation_dict = {
            "video_id": f"{key[0]}_{str(key[1])}"  # Add the original dictionary key as "video_id"
        }
        # Add the rest of the annotation data from the original dictionary value
        annotation_dict.update(annotation_data)
        # Append the new dictionary to the list
        annotations_list.append(annotation_dict)
    return annotations_list
    
anns=dict_to_list_with_key(anns)




## FineDiving

fd=FineDiving_dataset()
#fd.process_videos()

annotation_train = []
annotation_test = []
other = []


for a in anns:
    tmp = fd.run(a)
    if a['video_id'] in train_vid_names:
        annotation_train += tmp
    elif a['video_id'] in test_vid_names:
        annotation_test += tmp
    else:
        print(f"Video: { tmp['video_id']} not in train or test")
        other += tmp 


with open('data/STaR/FineDiving/star_annotations/train.json', 'w') as f:
    json.dump(annotation_train, f, indent=4)

with open('data/STaR/FineDiving/star_annotations/test.json', 'w') as f:
    json.dump(annotation_test, f, indent=4)



## MOMA

dataset = "data/MOMA/anns/anns.json"
with open(dataset, 'r') as file:
    anns = json.load(file)


moma = MOMA_dataset()
out = []
for a in tqdm(anns):
    out += moma.run(a)

out = [t for t in out if 'unsure' not in t['label'] and 'unsure' not in t['question'] and 'unclassified' not in t['question'] and 'unclassified' not in t['label']]
with open('data/STaR/MOMA/star_annotations/train.json', 'w') as f:
    json.dump(out, f, indent=4)




## Something-Something-v2

dataset = "data/something-something-v2/labels/train.json"
with open(dataset, 'r') as file:
    anns = json.load(file)

ss_d = SomethingSomething_dataset()

out = []
for a in tqdm(anns):
    out += ss_d.run(a)

with open('data/STaR/something-something-v2/star_annotations/train.json', 'w') as f:
    json.dump(out, f, indent=4)



## EgoExo-4D


import os 
import json
import numpy as np
from tqdm import tqdm

from videostar.data_gen_utils import *
from videostar.ego_exo_processors import *


dataset_dir = "data/EgoExo4D/annotations"
annot = {}
    
with open("data/EgoExo4D/takes.json", 'r') as file:
    takes = json.load(file)
takes = {t['take_uid']: t for t in takes}

with open(os.path.join(dataset_dir, 'atomic_descriptions_train.json'), 'r') as file:
    atomic_description = json.load(file)
annot['atomic_description'], _ = process_atomic_descriptions(takes, atomic_description)


with open(os.path.join(dataset_dir, 'expert_commentary_train.json'), 'r') as file:
    expert_commentary = json.load(file)
annot['expert_commentary'],_ = process_expert_commentary(takes, expert_commentary)


with open(os.path.join(dataset_dir, 'proficiency_demonstration_train.json'), 'r') as file:
    proficiency_demonstration = json.load(file)
annot['good_execution'], annot['tips_for_improvement']=process_proficiency_demonstration(takes, proficiency_demonstration)


with open(os.path.join(dataset_dir, 'proficiency_demonstrator_train.json'), 'r') as file:
    proficiency_demonstrator = json.load(file)
annot['proficiency_demonstrator']=process_proficiency_demonstrator(takes, proficiency_demonstrator, annot['atomic_description'])
## theses videos are too long. need to crop to subtimes. should probably use the existing clipped videos so to grab key steps.