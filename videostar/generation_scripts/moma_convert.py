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
from IPython.display import display, Image
from IPython.display import clear_output
import ipywidgets as widgets
import time
from PIL import Image
from decord import VideoReader
from tqdm import tqdm
import argparse

api_key = ""
gpt_model = "gpt-4"

client = OpenAI(
    # This is the default and can be omitted
    api_key=api_key,
)


def get_args_parser():
    parser = argparse.ArgumentParser('MOMA dataset convert', add_help=False)
    parser.add_argument('--chunk_inference', default=False, action='store_true')
    parser.add_argument('--num_chunks', default=1, type=int, help='Number of videos per chunk')
    parser.add_argument('--chunk_index', default=0, type=int, help='Index of the chunk to process')
    parser.add_argument('--save_dir', default='data/MOMA_star/star_annotations', type=str)
    parser.add_argument('--save_file', default='train.json', type=str)
    return parser


def main(args):
    dataset = "data/MOMA/anns/anns.json"
    with open(dataset, 'r') as file:
        anns = json.load(file)
        
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print(f"Directory {args.save_dir} created")

    if args.chunk_inference:
        save_file=str(args.chunk_index) + "-" + args.save_file
        total_videos = len(anns)
        chunk_size = total_videos//args.num_chunks
        start_index = args.chunk_index * chunk_size
        end_index = min(start_index + chunk_size, total_videos)
        selected_anns = anns[start_index:end_index]
        print(start_index, end_index)
    else:
        save_file=args.save_file
        selected_anns=anns

    print("number of videos:", len(selected_anns))

    moma = MOMA_dataset()
    out = []

    save_path = os.path.join(args.save_dir, save_file)
    for a in tqdm(selected_anns):
        out += moma.run(a)

    with open(save_path, 'w') as file:
        json.dump(out, file, indent=4)

    print("######### conversion completed. #########")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MOMA dataset convert', parents=[get_args_parser()])
    args = parser.parse_args()
    print(args)
    main(args)