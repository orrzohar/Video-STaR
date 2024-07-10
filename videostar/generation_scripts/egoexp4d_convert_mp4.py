import os 
import cv2
import json
import random
import shutil
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from videostar.generation_scripts.data_gen_utils import *
from videostar.generation_scripts.ego_exo_processors import *
from tqdm import tqdm
import argparse


file_names = [
    'Adjust a Rear Derailueur.json',
    'Los_Andes_FPC17_men.json',
    'Making Sesame-Ginger Asian Salad.json',
    'Basketball Drills - Mid-Range Jump Shooting.json',
    'Los_Andes_FPC17_women.json',
    'Making White Radish & Lettuce & Tomato & Cucumber Salad.json',
    'Basketball Drills - Mikan Layup.json',
    'Los_Andes_FPC18_men.json',
    'Minnesota_VE_Casual.json',
    'Basketball Drills - Reverse Layup.json',
    'Los_Andes_FPC1_women.json',
    'Performing the advanced choreography.json',
    'Clean and Lubricate the Chain.json',
    'Los_Andes_FPC2_women.json',
    'Performing the basic choreography.json',
    'Cooking an Omelet.json',
    'Los_Andes_FPC5_women.json',
    'Playing Guitar - Freeplaying.json',
    'Cooking Brownies.json',
    'LosAndes_Intermediate_Salsa.json',
    'Playing Guitar.json',
    'Cooking Dumplings.json',
    'Los_Andes_Super_Final_men.json',
    'Playing Guitar - Scales and Arpeggios.json',
    'Cooking Noodles.json',
    'LosAndes_V1_R1.json',
    'Playing Guitar - Suzuki Books.json',
    'Cooking Pasta.json',
    'LosAndes_V1_R2.json',
    'Playing Piano - Freeplaying.json',
    'Cooking Scrambled Eggs.json',
    'LosAndes_V2_R1.json',
    'Playing Piano.json',
    'Cooking Sushi Rolls.json',
    'LosAndes_V2_R2.json',
    'Playing Piano - Scales and Arpeggios.json',
    'Cooking Tomato & Eggs.json',
    'LosAndes_V2_R3.json',
    'Playing Piano - Suzuki Books.json',
    'Covid-19 Rapid Antigen Test.json',
    'LosAndes_V3_R1.json',
    'Playing Violin - Freeplaying.json',
    'First Aid - CPR.json',
    'LosAndes_V3_R2.json',
    'Playing Violin.json',
    'Fix a Flat Tire - Replace a Bike Tube.json',
    'LosAndes_V3_R3.json',
    'Playing Violin - Scales and Arpeggios.json',
    'Install a Wheel.json',
    'LosAndes_V3_R4.json',
    'Playing Violin - Suzuki Books.json',
    'LosAndes_Advanced_Salsa.json',
    'LosAndes_V4_R1.json',
    'Remove a Wheel.json',
    'LosAndes_Basic_Salsa.json',
    'LosAndes_V4_R2.json',
    'Rock Climbing.json',
    'Los_Andes_FPC10_men.json',
    'LosAndes_V4_R3.json',
    'Soccer Drills - Dribbling.json',
    'Los_Andes_FPC11_men.json',
    'LosAndes_V4_R4.json',
    'Soccer Drills - Inside Trap and Outside Play.json',
    'Los_Andes_FPC12_men.json',
    'LosAndes_V5_R1.json',
    'Soccer Drills - Juggling.json',
    'Los_Andes_FPC13_women.json',
    'LosAndes_V5_R2.json',
    'Soccer Drills - Outside Trap and Outside Play.json',
    'Los_Andes_FPC14_men.json',
    'LosAndes_V6_R1.json',
    'Soccer Drills - Penalty Kick.json',
    'Los_Andes_FPC14_women.json',
    'Making Chai Tea.json',
    'Soccer.json',
    'Los_Andes_FPC15_men.json',
    'Making Coffee latte.json',
    'Teaching the advanced choreography.json',
    'Los_Andes_FPC15_women.json',
    'Making Cucumber & Tomato Salad.json',
    'Teaching the basic choreography.json',
    'Los_Andes_FPC16_men.json',
    'Making Greek Salad.json',
    'Los_Andes_FPC16_women.json',
    'Making Milk Tea.json'
]

# Example usage
print(file_names)

json_files = [f"data/EgoExo4D/sub_ann/{fn}" for fn in file_names]



def get_args_parser():
    parser = argparse.ArgumentParser('EgoExo dataset convert', add_help=False)
    parser.add_argument('--chunk_inference', default=False, action='store_true')
    parser.add_argument('--num_chunks', default=1, type=int, help='Number of videos per chunk')
    parser.add_argument('--chunk_index', default=0, type=int, help='Index of the chunk to process')
    parser.add_argument('--save_dir', default='data/EgoExo4D/star_annotations', type=str)
    parser.add_argument('--save_file', default='train.json', type=str)
    return parser


def main(args):
    dataset = json_files[args.chunk_index]
    save_file = dataset.split('/')[-1]
    print('Save file dir:', save_file)

    with open(dataset, 'r') as file:
        selected_anns = json.load(file)

    print("number of videos:", len(selected_anns))

    egoexo = EgoExo4D_dataset()

    out = []

    for a in tqdm(selected_anns):
        if a['type'] =='atomic_description':
            out += egoexo.run_atomic_description(a)
        elif a['type'] =='expert_commentary':
            out += egoexo.run_expert_commentary(a)
        elif a['type'] =='good_execution':
            out += egoexo.run_good_execution(a)
        elif a['type'] =='tips_for_improvement':
            out += egoexo.run_tips_for_improvment(a)
        elif a['type'] =='proficiency_demonstrator':
            out += egoexo.run_proficiency_demonstrator(a)
        else:
            print('problem! ', a)

    
    with open(os.path.join(args.save_dir, save_file), 'w') as file:
        json.dump(out, file, indent=4)

    print("######### conversion completed. #########")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('EgoExo dataset convert', parents=[get_args_parser()])
    args = parser.parse_args()
    print(args)
    main(args)