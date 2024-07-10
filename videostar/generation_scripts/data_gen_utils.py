import os
import re
import cv2
import ast
import random
import shutil
import pickle
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from videostar.generation_scripts.templates import *

api_key = ""
gpt_model = "gpt-4"

client = OpenAI(
    # This is the default and can be omitted
    api_key=api_key,
)

def xywh_to_corners(bbox):
    x_min, y_min, width, height = bbox
    return [x_min, y_min, x_min + width, y_min + height]


def convert_webm_to_mp4(webm_path, mp4_path):
    # Capture the input video
    cap = cv2.VideoCapture(webm_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames in the video

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec used for mp4 format
    out = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))

    # Read frames from the webm file and write them to the mp4 file
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    return total_frames / fps if fps else 0 


def clean_text(text):
    # Trim leading and trailing spaces
    if isinstance(text, list):
        # Apply clean_text to each element of the list
        return [clean_text(item) for item in text]
        
    if isinstance(text, str):
        text = text.strip()
        text = text.replace('.,','.').replace('..','.').replace('?.','?')
        # Correct capitalization for the first letter, maintaining the rest as is
        if text:
            text = text[0].upper() + text[1:]
        # Ensure proper punctuation at the end; add a period if punctuation is missing
        if text and not re.search(r"[.!?]$", text):
            # Add a period if the sentence doesn't end with punctuation
            text += '.'
    return text
    

def trim_video_cv2(input_path, start_time, end_time, targetname):
    # Create a VideoCapture object
    directory = os.path.dirname(targetname)
    if not os.path.exists(directory):
        print('creating directory', directory)
        os.makedirs(directory)
        
    cap = cv2.VideoCapture(input_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        raise ValueError("Could not open the video file")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Calculate the start and end frame numbers
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    # Set video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID' if 'mp4v' is not working
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(targetname, fourcc, fps, (width, height))

    # Read and write the frames from start to end
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if start_frame <= frame_num <= end_frame:
            out.write(frame)
        elif frame_num > end_frame:
            break
        frame_num += 1

    # Release everything when job is finished
    cap.release()
    out.release()


def select_index(matches):
    # Get all indices where the value is True
    true_indices = [i for i, match in enumerate(matches) if match]
    
    if true_indices:
        # If there are True values, randomly select one of them
        return random.choice(true_indices)
    else:
        # If there are no True values, randomly select from all indices
        return random.randrange(len(matches))


def get_video_details(video_path):
    """
    Returns the duration of the video in seconds using OpenCV.

    Parameters:
    video_path (str): The path to the video file.

    Returns:
    float: Duration of the video in seconds.
    """
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Get the total number of frames in the video
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)

    # Get the frame rate of the video
    fps = video.get(cv2.CAP_PROP_FPS)
    
    # Calculate the duration of the video
    duration = frame_count / fps

    # Get the width and height of the video
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Release the video capture object
    video.release()
    
    return fps, int(frame_count), duration, width, height


def generate_questions(num_questions: int, num_classes: int, label_list: list, task:str, example:str):
    ## Inputs:
    # num_questions: number of questions to generate
    # num_classes
    # label_list: list of dicts, each dict representing a single video
    # task: task description. 
    # label: lambda function that transforms a single dict label instance into the class label.
    ## outputs:
    # list of proposed questions.
    pbar = tqdm(total=num_questions, desc="Generating questions")
    init_question_prompt = lambda c: f"I am perfoming {task} with {c} labels. How would you ask someone to perform {task} on a video without mentioning the actual task?  For example, a suitable question is '{example}'. Generate 10 unique questions."

    questions4 = []
    while len(questions4) < num_questions:
        cls_list =[t for t in random.sample(label_list, num_classes)]
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": init_question_prompt(cls_list),
                }
            ],
            model = gpt_model,
        )
        question = chat_completion.choices[0].message.content
        if "without mentioning" not in question and question not in questions4:
            questions4+=question.split('\n')
            pbar.update(len(question.split('\n'))) 
    
    pbar.close()  
    return questions4


def normalize_bbox(bbox, W, H):
    """
    Normalize bounding box coordinates to [0,1] after padding the image to be square.
    
    Parameters:
    - bboxes: List of bounding boxes in the format [[x1, y1, x2, y2], ...]
    - W: Original width of the image
    - H: Original height of the image

    Returns:
    - List of normalized bounding boxes in the format [[wa, ha, wb, hb], ...]
    """
    max_side = max(W, H)

    # Calculate padding to make the image square
    pad_w = (max_side - W) / 2
    pad_h = (max_side - H) / 2
    
    x1, y1, x2, y2 = bbox

    # Adjust coordinates for padding
    x1_padded = x1 + pad_w
    y1_padded = y1 + pad_h
    x2_padded = x2 + pad_w
    y2_padded = y2 + pad_h

    # Normalize coordinates to [0, 1] range
    wa = x1_padded / max_side
    ha = y1_padded / max_side
    wb = x2_padded / max_side
    hb = y2_padded / max_side

    norm_bbox = [wa, ha, wb, hb]
    return [round(i, random.randint(2, 4)) for i in norm_bbox]
    

class STARB_dataset(object):
    def __init__(self, source_dataset_path="data/STARB", target_dataset_path="data/STaR/STARB",video_path='Charades_v1_480', dest_video_path='star_videos'):
        self.source_dataset_path=source_dataset_path
        self.video_path = os.path.join(source_dataset_path, video_path)
        self.dest_video_path = os.path.join(target_dataset_path, dest_video_path)
        self.setup_label_translation()
        self.id=0
        os.makedirs(self.dest_video_path, exist_ok=True)

    def setup_label_translation(self):
        trans_dict = {}
        
        # Open the text file in read mode
        with open(os.path.join(self.source_dataset_path, "classes/action_classes.txt"), 'r') as file:
            # Read the contents of the file
            action_classes = file.read().splitlines()
        
        with open(os.path.join(self.source_dataset_path, "classes/object_classes.txt"), 'r') as file:
            # Read the contents of the file
            object_classes = file.read().splitlines()
        
        with open(os.path.join(self.source_dataset_path, "classes/relationship_classes.txt"), 'r') as file:
            # Read the contents of the file
            relationship_classes = file.read().splitlines()
        
        with open(os.path.join(self.source_dataset_path, "classes/verb_classes.txt"), 'r') as file:
            # Read the contents of the file
            verb_classes = file.read().splitlines()
            
        for line in action_classes:
            parts = line.split(' ', 1)
            if len(parts) == 2:
                action_code, description = parts
                trans_dict[action_code] = description
        
        for line in object_classes:
            parts = line.split(' ', 1)
            if len(parts) == 2:
                action_code, description = parts
                trans_dict[action_code] = description
        
        for line in relationship_classes:
            parts = line.split(' ', 1)    
            if len(parts) == 2:
                action_code, description = parts
                trans_dict[action_code] = description
        
        for line in relationship_classes:
            parts = line.split(' ', 1)
            if len(parts) == 2:
                action_code, description = parts
                trans_dict[action_code] = description
                
        self.translation_dict = trans_dict
        return

    def translate_label_instance(self, label_instance):
        _, frame_count, duration, width, height = get_video_details(os.path.join(self.video_path, label_instance['video_id']+".mp4"))
        if  label_instance['end']>duration:
            #print("labeled end:", label_instance['end'], "duration:",duration)
            label_instance['end']=duration
            
        label_instance['duration'] = round(duration,2)
        label_instance['start_time'] = round(label_instance['start'], 3)
        label_instance['end_time'] = round(label_instance['end'], 3)
        label_instance['start'] /= duration
        label_instance['end'] /= duration
        label_instance['start_frame'] = int(label_instance['start'] * frame_count)
        label_instance['end_frame'] = int(label_instance['end'] * frame_count)
        label_instance['frame_count'] = frame_count

        key_types = ['actions', 'bbox_labels', 'rel_labels', 'verbs']
        # Check and translate actions
        for situation in label_instance['situations'].values():
            for k in key_types:
                if k in situation:
                    situation[k] = [self.translation_dict.get(tmp, "Unknown") for tmp in situation[k]]
                    
            if 'rel_pairs' in situation:
                situation['rel_pairs'] = [[self.translation_dict.get(tmp[0], "Unknown"), self.translation_dict.get(tmp[1], "Unknown")] for tmp in situation['rel_pairs']]
                
            if 'bbox' in situation:
                situation['bbox'] = [normalize_bbox(bbox,  width, height) for bbox in situation['bbox']]
                
    def save_videos(self, annot):
        original_video_path = os.path.join(self.video_path, f"{annot['video_id']}.mp4")
        full_video_path = os.path.join(self.dest_video_path, f"{annot['video_id']}.mp4")
        trimmed_video_path = os.path.join(self.dest_video_path, f"{annot['video_id']}_T{annot['start_time']}_{annot['end_time']}.mp4")
        annot['duration_trimmed']=round(annot['end_time']-annot['start_time'],2)

        # Save Original Video
        if not os.path.exists(full_video_path):
            shutil.copyfile(original_video_path, full_video_path)
        
        # Extract frames for the trimmed version
        if not os.path.exists(trimmed_video_path):
            trim_video_cv2(original_video_path, annot['start_time'], annot['end_time'], targetname=trimmed_video_path)
        
        annot['full_video'] = full_video_path
        annot['trimmed_video'] = trimmed_video_path

    def generate_temporal_qa(self, annot):
        def qa_summarize_actions(situations):
            unique_actions = set()
            for situation in situations.values():
                unique_actions.update(situation['actions'])
            return list(unique_actions)
            
        unique_actions = qa_summarize_actions(annot['situations'])
        t1, t2 = min([int(k) for k in annot['situations'].keys()])/annot['frame_count'], max([int(k) for k in annot['situations'].keys()])/annot['frame_count']
        out = []
        out.append({
            "video_id": annot["video_id"],
            "question_id": annot["question_id"]+"_t",
            "question": random.choice(temporal_action_prompts)(round(t1, random.randint(2, 4)), round(t2, random.randint(2, 4))),
            "label": {"unique_actions": unique_actions},
            "video_path": annot['full_video'].split('/', 1)[-1],
            "duration": annot["duration"],
            "label_type": ["unique_actions"],
            "verifier_type": ["sentence"],
            "id": self.id
        })
        self.id+=1
        ###################
        out.append({
            "video_id": annot["video_id"],
            "question_id": annot["question_id"]+"_rt",
            "question": random.choice(reverse_temporal_action_prompts)(unique_actions),
            "label": {"temporal_range": [round(t1, random.randint(2, 4)), round(t2, random.randint(2, 4))]},
            "video_path": annot['full_video'].split('/', 1)[-1],
            "duration": annot["duration"],
            "label_type": ["temporal_range"],
            "verifier_type": ["temporal_range"],
            "id": self.id
        })
        self.id+=1

        out.append({
            "video_id": annot["video_id"],
            "question_id": annot["question_id"]+"_tqa",
            "question": random.choice(temporal_qa_prompts)(annot['question'], annot['answer']),
            "label": {"temporal_range": [round(t1, random.randint(2, 4)), round(t2, random.randint(2, 4))]},
            "video_path": annot['full_video'].split('/', 1)[-1],
            "duration": annot["duration"],
            "label_type": ["temporal_range"],
            "verifier_type": ["temporal_range"],
            "id": self.id
        })
        self.id+=1
        return out

    def generate_spatial_qa(self, annot):
        out = []
        valid_frames = [frame for frame, data in annot['situations'].items() if data.get('bbox_labels')]
        
        # Check if there are valid frames with bbox labels
        if not valid_frames:
            print("No valid frames with bounding box labels available.")
            return out

        frame = random.choice(valid_frames)
        ann = annot['situations'][frame]
        
        selected_index = random.randint(0, len(ann['bbox_labels'])-1)
        object_label = ann['bbox_labels'][selected_index]
        object_bbox = [round(b, random.randint(2, 4)) for b in ann['bbox'][selected_index]]
        object_time = round((int(frame)-annot['start_frame'])/(annot['end_frame']-annot['start_frame']), random.randint(2, 4))
        
        out.append({
            "video_id": annot["video_id"],
            "question_id": annot["question_id"]+"_st",
            "question": random.choice(spatiotemporal_prompts)(object_bbox, object_time),
            "label": {"identify": object_label},
            "video_path": annot['trimmed_video'].split('/', 1)[-1],
            "duration": annot["duration_trimmed"],
            "label_type": ["identify"],
            "verifier_type": ["obj_identify"],
            "id": self.id
        })
        self.id+=1

        frame = random.choice(valid_frames)
        ann = annot['situations'][frame]
        if random.random()>0.5:
            selected_index = random.randint(0, len(ann['bbox_labels'])-1)
        else:
            selected_index = select_index([a==object_label for a in ann['bbox_labels']])
            
        object_label2 = ann['bbox_labels'][selected_index]
        object_bbox2 = [round(b, random.randint(2, 4)) for b in ann['bbox'][selected_index]]
        object_time2 = round((int(frame)-annot['start_frame'])/(annot['end_frame']-annot['start_frame']), random.randint(2, 4))            
        
        same_or_not = "The objects are the same" if object_label==object_label2 else "the objects are NOT the same"
        out.append({
            "video_id": annot["video_id"],
            "question_id": annot["question_id"]+"_st2",
            "question": random.choice(spatiotemporal_comparison_prompts)(object_bbox, object_time, object_bbox2, object_time2),
            "label": {"first_object": object_label, "second_object": object_label2, "same_or_not": same_or_not},
            "video_path": annot['trimmed_video'].split('/', 1)[-1],
            "duration": annot["duration_trimmed"],
            "label_type": ["first_object", "second_object", "same_or_not"],
            "verifier_type": ["simple", "simple", "sentence"],
            "id": self.id
        })
        
        self.id+=1
        return out 

    def original_qa(self, annot):
        out = []
        out.append({
            "video_id": annot["video_id"],
            "question_id": annot["question_id"],
            "question": annot["question"],
            "label": {"answer_question": annot["answer"]},
            "video_path": annot['trimmed_video'].split('/', 1)[-1],
            "duration": annot["duration_trimmed"],
            "label_type": ["answer_question"],
            "verifier_type": ["sentence"],
            "id": self.id
        })
        self.id+=1
        return out
        
    def reversed_qa(self, annot):
        out = []
        out.append({
            "video_id": annot["video_id"],
            "question_id": annot["question_id"]+"_r",
            "question": random.choice(question_answer_reversal)(annot["answer"]),
            "label": {"question_from_answer": annot["question"]},
            "video_path": annot['trimmed_video'].split('/', 1)[-1],
            "duration": annot["duration_trimmed"],
            "label_type": ["question_from_answer"],
            "verifier_type": ["sentence"],
            "id": self.id
        })
        self.id+=1
        return out    

    def process_video(self, annot):
        out   = []
        self.translate_label_instance(annot)
        self.save_videos(annot)
        
        out  += self.original_qa(annot)
        out  += self.reversed_qa(annot)
        out  += self.generate_temporal_qa(annot)
        out  += self.generate_spatial_qa(annot)
        return out
        

class FineDiving_dataset(object):
    def __init__(self, 
                 source_path="FineDiving", 
                 target_path="STaR/FineDiving",
                 data_path="data",
                 video_frames_subdir='VideoFrames', 
                 dest_video_path='star_videos', 
                 annotation_file='Annotations/FineDiving_fine-grained_annotation.pkl'):
        
        self.source_path = source_path
        self.target_path = target_path
        self.data_path = data_path
        self.video_frames_dir = os.path.join(data_path, source_path, video_frames_subdir)
        self.dest_video_path = os.path.join(target_path, dest_video_path)
        self.annotation_file = os.path.join(data_path, source_path, annotation_file)
        self.annotations = self.load_annotations()
        self.id = 0
        # Ensure the destination path exists
        os.makedirs(self.dest_video_path, exist_ok=True)

    def load_annotations(self):
        with open(self.annotation_file, 'rb') as f:
            return pickle.load(f)

    def convert_images_to_video(self, image_folder, output_video_file, fps=12):
        frames = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')])
        if not frames:
            return

        first_frame = cv2.imread(frames[0])
        height, width, _ = first_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

        for frame_path in frames:
            frame = cv2.imread(frame_path)
            out.write(frame)
        
        out.release()

    def process_videos(self):
        for subdir, dirs, _ in os.walk(self.video_frames_dir):
            if subdir != self.video_frames_dir:
                for d in dirs:
                    image_folder = os.path.join(subdir, d)
                    output_video_file = os.path.join(self.data_path, self.dest_video_path, f"{os.path.basename(subdir)}_{d}.mp4")
                    if not os.path.exists(output_video_file):  # Check if video already exists
                        self.convert_images_to_video(image_folder, output_video_file)

    def process_annotation(self, annotation):
        video_path = os.path.join(self.dest_video_path, f"{annotation['video_id']}.mp4")
        steps_transit_frames = annotation['steps_transit_frames'].tolist()
        sub_act_ids = np.unique(annotation['frames_labels'])
        sub_actions = annotation['sub-action_types']

        processed_annotation = {
            'video_id': annotation['video_id'],
            'video_path': video_path,
            'steps_transit_frames': steps_transit_frames,
            'dive_score': annotation['dive_score'],
            'difficulty': annotation['difficulty'],
            'sub_actions': [],
            'frames_labels': [],
            'duration': len(annotation['frames_labels'])/12,
            'action_sequence': ''
        }
        try:
            processed_annotation['sub_actions'] = [sub_actions[i] for i in sub_act_ids]
            processed_annotation['frames_labels'] = [sub_actions[i] for i in annotation['frames_labels'].tolist()]
            processed_annotation['action_sequence'] = processed_annotation['sub_actions']

        except KeyError:
            processed_annotation['sub_actions'] = [sub_actions[23] if i == 22 else sub_actions[i] for i in sub_act_ids]
            processed_annotation['frames_labels'] = [sub_actions[23] if i == 22 else sub_actions[i] for i in annotation['frames_labels'].tolist()]
            processed_annotation['action_sequence'] = processed_annotation['sub_actions']
            
        return processed_annotation

    def action_tqa(self, annot):
        out = []
        question = random.choice(after_prompts)(annot['sub_actions'][0])
        label = annot['sub_actions'][1]
        out.append({
            "video_id": annot["video_id"],
            "question_id": annot["video_id"] + "_" + str(self.id),
            "question": question,
            "label": {"action_after_action": label},
            "video_path": annot['video_path'],
            "duration": annot['duration'],
            "label_type": ["action_after_action"],
            "verifier_type": ["sentence"],
            "id": self.id
        })
        self.id += 1
        
        question = random.choice(before_prompts)(annot['sub_actions'][1])
        label = annot['sub_actions'][0]
        out.append({
            "video_id": annot["video_id"],
            "question_id": annot["video_id"] + "_" + str(self.id),
            "question": question,
            "label": {"action_before_action": label},
            "video_path": annot['video_path'],
            "duration": annot['duration'],
            "label_type": ["action_before_action"],
            "verifier_type": ["sentence"],
            "id": self.id
        })
        self.id += 1
        
        return out

    def seq_action_qa(self, annot):
        out = []
        out.append({
            "video_id": annot["video_id"],
            "question_id": annot["video_id"] + "_" + str(self.id),
            "question": random.choice(subaction_sequence_prompts),
            "label": {"action_sequence": annot['action_sequence']},
            "video_path": annot['video_path'],
            "duration": annot['duration'],
            "label_type": ["action_sequence"],
            "verifier_type": ["sentence_sequence"],
            "id": self.id
        })
        self.id += 1
        return out
        
    def aqa_qa(self, annot):
        out = []
        out.append({
            "video_id": annot["video_id"],
            "question_id": annot["video_id"] + "_" + str(self.id),
            "question": random.choice(aqa_prompts),
            "label": [annot['action_sequence'], annot['difficulty'], annot['dive_score']],
            "label": {"action_sequence": annot['action_sequence'], "action_difficulty": annot['difficulty'], "final_score": annot['dive_score']},
            "video_path": annot['video_path'],
            "duration": annot['duration'],
            "label_type": ["action_sequence", "action_difficulty", "final_score"],
            "verifier_type": ["sentence_sequence", "simple", "float"],
            "id": self.id
        })
        self.id += 1
        out.append({
            "video_id": annot["video_id"],
            "question_id": annot["video_id"] + "_" + str(self.id),
            "question": random.choice(aqa_prompts),
            "label": [annot['dive_score']],
            "label": {"final_score": annot['dive_score']},
            "video_path": annot['video_path'],
            "duration": annot['duration'],
            "label_type": ["final_score"],
            "verifier_type": ["float"],
            "id": self.id
        })
        self.id += 1

        
        return out

    def run(self, annotation):
        out  = []
        annotation = self.process_annotation(annotation)
        out += self.action_tqa(annotation)
        out += self.seq_action_qa(annotation)
        out += self.aqa_qa(annotation)
        return out


class SomethingSomething_dataset(object):
    def __init__(self, 
                 source_path="something-something-v2", 
                 target_path="STaR/something-something-v2", 
                 dataset_path="data",
                 video_subdir='videos', 
                 dest_video_path='star_videos'
                 ):
        self.dataset_path = dataset_path
        self.video_dir = os.path.join(dataset_path, source_path, video_subdir)
        self.dest_video_path = os.path.join(dataset_path, target_path, dest_video_path)
        self.id = 0
        # Ensure the destination path exists
        os.makedirs(self.dest_video_path, exist_ok=True)
        
    def process_video(self, annot):
        original_video_path = os.path.join(self.video_dir, f"{annot['id']}.webm")
        converted_video_path = os.path.join(self.dest_video_path, f"{annot['id']}.mp4")
        if not os.path.exists(converted_video_path):
            annot['duration'] = convert_webm_to_mp4(original_video_path, converted_video_path)
        else:
            _, _, annot['duration'], _, _ = get_video_details(converted_video_path)
        annot['video_path'] = converted_video_path
        
    def template_qa(self, annot):
        out = []
        out.append({
            "video_id": annot["id"],
            "question_id": annot["id"] + "_" + str(self.id),
            "question": random.choice(fill_in_prompts)(annot['template']),
            "label": {"objects": annot['placeholders']},
            "video_path": annot['video_path'].split('/',1)[-1],
            "duration": annot['duration'],
            "label_type": ["objects"],
            "verifier_type": ["simple"],
            "id": self.id
        })
        self.id += 1
        return out

    def reverse_template_qa(self, annot):
        out = []
        out.append({
            "video_id": annot["id"],
            "question_id": annot["id"] + "_" + str(self.id),
            "question": random.choice(guess_full_sentence_prompts)(annot['placeholders']),
            "label": {"relationship": annot['label']}, #TODO: changed from 'answer' need to make sure this is OK. 
            "video_path": annot['video_path'],
            "duration": annot['duration'],
            "label_type": ["relationship"],
            "verifier_type": ["sentence"],
            "id": self.id
        })
        self.id += 1
        return out

    def run(self, annotation):
        out  = []
        self.process_video(annotation)
        out += self.template_qa(annotation)
        out += self.reverse_template_qa(annotation)
        return out


class MOMA_dataset(object):
    def __init__(self, source_path="data/MOMA", target_path="data/STaR/MOMA",video_path='videos/raw', dest_video_path='star_videos', max_duration=30):
        self.dataset_path=source_path
        self.video_path = os.path.join(source_path, video_path)
        self.dest_video_path = os.path.join(target_path, dest_video_path)
        self.id=0
        self.max_duration=max_duration
        os.makedirs(self.dest_video_path, exist_ok=True)
        
    def translate_label_instance(self, label_instance):
        label_instance['action_sequence'] = [a['class_name'] for a in label_instance['sub_activities']]
        for sub_activities in label_instance['sub_activities']:
            for higher_order in sub_activities['higher_order_interactions']:
                
                for actor in higher_order['actors']:
                    actor['bbox']=xywh_to_corners(actor['bbox'])
                    
                for obj in higher_order['objects']:
                    obj['bbox']=xywh_to_corners(obj['bbox'])
                    
        return label_instance

    def save_videos(self, annot):
        original_video_path = os.path.join(self.video_path, f"{annot['id']}.mp4")
        full_video_path = os.path.join(self.dest_video_path, f"{annot['id']}.mp4")
        trimmed_video_path = os.path.join(self.dest_video_path, f"{annot['id']}_T{annot['start_time']}_{annot['end_time']}.mp4")

        trimmed_sub_video_paths = [os.path.join(self.dest_video_path, f"{annot['id']}_T{a['start_time']}_{a['end_time']}.mp4") for a in annot['sub_activities']]
        
        # Save Original Video
        if not os.path.exists(full_video_path):
            shutil.copyfile(original_video_path, full_video_path)

        if not os.path.exists(trimmed_video_path):
            trim_video_cv2(original_video_path, annot['start_time'], annot['end_time'], targetname=trimmed_video_path)
            
        # Extract frames for the trimmed version
        for i in range(len(trimmed_sub_video_paths)):
            if not os.path.exists(trimmed_sub_video_paths[i]):
                trim_video_cv2(original_video_path, annot['sub_activities'][i]['start_time'], annot['sub_activities'][i]['end_time'], targetname=trimmed_sub_video_paths[i])
        
        annot['full_video'] = full_video_path
        annot['trimmed_video'] = trimmed_video_path
        annot['trimmed_sub_video_paths'] = trimmed_sub_video_paths
        return annot
        
    def generate_spatial_qa(self, annot, i, width, height):
        out = []
        subaction = annot['sub_activities'][i]
        video = annot['trimmed_sub_video_paths'][i]
        start_time = subaction['start_time']
        end_time = subaction['end_time']
        norm_time = lambda time: round((time - start_time) / (end_time - start_time), random.randint(2, 4))

        for interation in subaction['higher_order_interactions']:
            class_location_t = {}
            class_t = {}
            for i in interation['actors']:
                class_location_t[i['id']] = f"the {i['class_name']} at {normalize_bbox(i['bbox'], width, height)}"
                class_t[i['id']] = i['class_name']
            
            for i in interation['objects']:
                class_location_t[i['id']] = f"the {i['class_name']} at {normalize_bbox(i['bbox'], width, height)}"
                class_t[i['id']] = i['class_name']

            for relation in interation['relationships']:
                question = random.choice(relationship_prompts)(class_location_t[relation['source_id']], 
                                                               class_location_t[relation['target_id']], 
                                                               norm_time(interation['time']))
                
                label = relation['class_name'].replace('[src]', class_t[relation['source_id']]).replace('[trg]', class_t[relation['target_id']])
                out.append({
                    "video_id": annot["id"],
                    "question_id": annot["id"] + "_" + str(self.id),
                    "question": question,
                    "label": {"relationship_interation": label},
                    "video_path": video,
                    "duration":end_time-start_time,
                    "label_type": ["relationship_interation"],
                    "verifier_type": ["sentence"],
                    "id": self.id
                })
                self.id+=1
                
            for relation in interation['transitive_actions']:
                question = random.choice(relationship_prompts)(class_location_t[relation['source_id']], 
                                                               class_location_t[relation['target_id']], 
                                                               norm_time(interation['time']))
                
                label = relation['class_name'].replace('[src]', class_t[relation['source_id']]).replace('[trg]', class_t[relation['target_id']])
                out.append({
                    "video_id": annot["id"],
                    "question_id": annot["id"] + "_" + str(self.id),
                    "question": question,
                    "label": {"relationship_interation": label},
                    "video_path": video,
                    "duration":end_time-start_time,
                    "label_type": ["relationship_interation"],
                    "verifier_type": ["sentence"],
                    "id": self.id
                })
                self.id+=1 
                ## TODO: fill in the blank?

        if len(out)>10:
            return random.sample(out, 10)
        return out

    def generate_obj_compare_qa(self, annot, i, width, height):
        out = []
        subaction = annot['sub_activities'][i]
        video = annot['trimmed_sub_video_paths'][i]
        start_time = subaction['start_time']
        end_time = subaction['end_time']
        norm_time = lambda time: round((time - start_time) / (end_time - start_time), random.randint(2, 4))

        actors = {}
        for i, a in enumerate(subaction['higher_order_interactions']):
            for actor in a['actors']:
                if actor['id'] in actors:
                    actors[actor['id']].append(i)
                else:
                    actors[actor['id']]=[i]

        for k, v in actors.items():
            if len(v) > 2: #obj, t1, bbox1, t2:
                selected_items = random.sample(v, 2)
                subactions1 = subaction['higher_order_interactions'][selected_items[0]]
                subactions2 = subaction['higher_order_interactions'][selected_items[1]]
                
                actor1 = next((d for d in subactions1['actors'] if d.get('id') == k), None)
                actor2 = next((d for d in subactions2['actors'] if d.get('id') == k), None)
                same_or_not = "The objects are the same" if actor1['class_name']==actor2['class_name'] else "the objects are NOT the same"
                out.append({
                    "video_id": annot["id"],
                    "question_id": annot["id"] + "_" + str(self.id),
                    "question": random.choice(spatiotemporal_comparison_prompts)(normalize_bbox(actor1['bbox'], width, height), 
                                                                                 norm_time(subactions1['time']), 
                                                                                 normalize_bbox(actor2['bbox'], width, height), 
                                                                                 norm_time(subactions2['time'])),
                    "label": {"first_object": actor1['class_name'], "second_object": actor2['class_name'], "same_or_not": same_or_not},
                    "video_path": video,
                    "duration":end_time-start_time,
                    "label_type": ["first_object", "second_object", "same_or_not"],
                    "verifier_type": ["simple", "simple", "sentence"],
                    "id": self.id
                })
                self.id+=1

            subactions1 = subaction['higher_order_interactions'][random.choice(v)]
            other_actors = [a for a in list(actors.keys()) if a !=k]
            if len(other_actors)>1:
                second_actor = random.choice(other_actors)
                second_frame = random.choice(actors[second_actor])
                subactions2 = subaction['higher_order_interactions'][second_frame]
                
                actor1 = next((d for d in subactions1['actors'] if d.get('id') == k), None)
                actor2 = next((d for d in subactions2['actors'] if d.get('id') == second_actor), None)
                same_or_not = "The objects are the same" if actor1['class_name']==actor2['class_name'] else "the objects are NOT the same"
                out.append({
                    "video_id": annot["id"],
                    "question_id": annot["id"] + "_" + str(self.id),
                    "question": random.choice(spatiotemporal_comparison_prompts)(normalize_bbox(actor1['bbox'], width, height), 
                                                                                 norm_time(subactions1['time']), 
                                                                                 normalize_bbox(actor2['bbox'], width, height), 
                                                                                 norm_time(subactions2['time'])),
                    "label": {"first_object": actor1['class_name'], "second_object": actor2['class_name'], "same_or_not": same_or_not},
                    "video_path": video,
                    "duration":end_time-start_time,
                    "label_type": ["first_object", "second_object", "same_or_not"],
                    "verifier_type": ["simple", "simple", "sentence"],
                    "id": self.id
                })
                self.id+=1
                
        if len(out)>10:
            return random.sample(out, 10)
        return out
    
    def generate_loc_predict_qa(self, annot, i, width, height):
        out = []
        subaction = annot['sub_activities'][i]
        video = annot['trimmed_sub_video_paths'][i]
        start_time = subaction['start_time']
        end_time = subaction['end_time']
        norm_time = lambda time: round((time - start_time) / (end_time - start_time), random.randint(2, 4))

        actors = {}
        for i, a in enumerate(subaction['higher_order_interactions']):
            for actor in a['actors']:
                if actor['id'] in actors:
                    actors[actor['id']].append(i)
                else:
                    actors[actor['id']] = [i]
        
        for k, v in actors.items():
            if len(v) > 2: #obj, t1, bbox1, t2:
                selected_items = random.sample(v, 2)
                subactions1 = subaction['higher_order_interactions'][selected_items[0]]
                subactions2 = subaction['higher_order_interactions'][selected_items[1]]
                
                actor1 = next((d for d in subactions1['actors'] if d.get('id') == k), None)
                actor2 = next((d for d in subactions2['actors'] if d.get('id') == k), None)
                # the actor appears in multiple frames
                question =  random.choice(loc_predict_prompts)(actor1['class_name'], 
                                                               norm_time(subactions1['time']), 
                                                               normalize_bbox(actor1['bbox'], width, height), 
                                                               norm_time(subactions2['time']))
                out.append({
                    "video_id": annot["id"],
                    "question_id": annot["id"] + "_" + str(self.id),
                    "question": question,
                    "label": {"location": normalize_bbox(actor2['bbox'], width, height)},
                    "video_path": video,
                    "duration":end_time-start_time,
                    "label_type": ["location"],
                    "verifier_type": ["bbox"],
                    "id": self.id
                })
                self.id+=1
                
        if len(out)>10:
            return random.sample(out, 10)
        return out
        
    def action_sequence_qa(self, annot):
        out = []
        out.append({
            "video_id": annot["id"],
            "question_id": annot["id"] + "_" + str(self.id),
            "question": random.choice(subaction_sequence_prompts),
            "label": {"action_sequence": list(set(annot['action_sequence']))},
            "video_path": annot['trimmed_video'].split('/', 1)[-1],
            "duration":annot['end_time']-annot['start_time'],
            "label_type": ["action_sequence"],
            "verifier_type": ["sentence_sequence"],
            "id": self.id
        })
        self.id+=1
        return out    
        
    def action_tqa(self, annot):
        #TODO: can make this more dense.
        out = []
        actsequence  = list(set(annot['action_sequence']))
        if len(actsequence)>1:
            idx = [i for i in range(len(actsequence))]
            i = random.choice(idx[:-1])
            question = random.choice(after_prompts)(actsequence[i])
            out.append({
                "video_id": annot["id"],
                "question_id": annot["id"] + "_" + str(self.id),
                "question": question,
                "label": {"action_after_action": actsequence[i+1]},
                "video_path": annot['trimmed_video'],
                "duration":annot['end_time']-annot['start_time'],
                "label_type": ["action_after_action"],
                "verifier_type": ["sentence"],
                "id": self.id
            })
            self.id += 1
            
            i = random.choice(idx[1:])
            question = random.choice(before_prompts)(actsequence[i])            
            out.append({
                "video_id": annot["id"],
                "question_id": annot["id"] + "_" + str(self.id),
                "question": question,
                "label": {"action_before_action": actsequence[i-1]},
                "duration": annot['end_time']-annot['start_time'],
                "video_path": annot['trimmed_video'],
                "label_type": ["action_before_action"],
                "verifier_type": ["sentence"],
                "id": self.id
            })
            self.id += 1
        return out
        
    def run(self, annot):
        out   = []
        width, height = annot['width'], annot['height']
        annot = self.translate_label_instance(annot['activity'])
        annot = self.save_videos(annot)
        
        out  += self.action_sequence_qa(annot)
        out  += self.action_tqa(annot)
                    
        for i in range(len(annot['sub_activities'])):
            out+= self.generate_loc_predict_qa(annot, i, width, height)
            out+= self.generate_obj_compare_qa(annot, i, width, height)
            out+= self.generate_spatial_qa(annot, i, width, height)
        return out


class EgoExo4D_dataset(object):
    def __init__(self, source_path="data/EgoExo4D", target_path="data/STaR/EgoExo4D/star_videos",video_path='videos/raw', dest_video_path='star_videos', min_duration=4, max_duration=30):
        self.source_path=source_path
        self.video_path = os.path.join(source_path, video_path)
        self.target_path = target_path
        self.id=0
        self.saved_videos = {}
        self.min_duration=min_duration
        self.max_duration=max_duration

    def format_answer(self,input_string):
        return ast.literal_eval(input_string)

    def save_video(self, video_path, video_time, video_duration):
        output_videos = []
        duration = round(random.uniform(self.min_duration, self.max_duration), 1)
        if duration > video_duration:
            ## use original video
            return 0, video_duration, video_path
        
        start_time = round(max(video_time - random.uniform(0.1, duration-0.1), 0), 1)
        end_time = round(min(start_time + duration, video_duration), 1)
        if (end_time-start_time) < self.min_duration:
            start_time=round(end_time-self.min_duration, 1)
            
        trimmed_video_path =  os.path.join(self.target_path, video_path.replace('.mp4', f"_T_{start_time}_{end_time}.mp4"))
        video_path = os.path.join(self.source_path, video_path)

        if video_path in self.saved_videos:
            # Check each saved time range for this video
            for saved_start, saved_end, saved_video_path in self.saved_videos[video_path]:
                if (video_time >= saved_start and video_time <= saved_end):
                    return saved_start, saved_end, saved_video_path
            #print('adding video', trimmed_video_path)
            trim_video_cv2(video_path, start_time, end_time, targetname=trimmed_video_path)
            self.saved_videos[video_path].append([start_time, end_time, trimmed_video_path])
        else:
            #print('adding video', trimmed_video_path)
            trim_video_cv2(video_path, start_time, end_time, targetname=trimmed_video_path)
            self.saved_videos[video_path] = [[start_time, end_time, trimmed_video_path]]

        return start_time, end_time, trimmed_video_path
    
    def run_atomic_description(self, annot):
        out = []
        saved_start, saved_end, saved_video_path = self.save_video(annot['video_path'], annot['video_time'], annot['duration'])
        annot['video_time'] = round((annot['video_time'] - saved_start) / (saved_end - saved_start), random.randint(2, 4))
        
        out.append({
            "video_id": annot["id"],
            "question_id": annot["id"] + "_" + str(self.id),
            "question": clean_text(random.choice(prompts_for_normalized_time)(annot['video_time'])),
            "label": {"action_description": clean_text(annot['text'])},
            "video_path": saved_video_path,
            "duration": round(saved_end-saved_start,1),
            "label_type": ["action_description"],
            "verifier_type": ["sentence"],
            "id": self.id
        })
        self.id += 1
        
        out.append({
            "video_id": annot["id"],
            "question_id": annot["id"] + "_" + str(self.id),
            "question": random.choice(reverse_single_timestamp_prompts)(annot['text']).strip(),
            "label": {"normalized_time": annot['video_time']},
            "video_path": saved_video_path,
            "duration": round(saved_end-saved_start,1),
            "label_type": ["normalized_time"],
            "verifier_type": ["timestamp"],
            "id": self.id
        })
        self.id += 1
        return out

    def run_expert_commentary(self, annot):
        out = []
        saved_start, saved_end, saved_video_path = self.save_video(annot['video_path'], annot['video_time'], annot['duration'])
        
        out.append({
            "video_id": annot["id"],
            "question_id": annot["id"] + "_" + str(self.id),
            "question": clean_text(random.choice(prompts_for_expert_commentary)(annot['task_name'])),
            "label": {"expert_commentary": clean_text(annot['text'])},
            "video_path": saved_video_path,
            "duration": round(saved_end-saved_start,1),
            "label_type": ["expert_commentary"],
            "verifier_type": ["llm"],
            "id": self.id
        })
        self.id += 1
        return out

    def run_good_execution(self, annot):
        out = []
        saved_start, saved_end, saved_video_path = self.save_video(annot['video_path'], annot['video_time'], annot['duration'])
        annot['video_time'] = round((annot['video_time'] - saved_start) / (saved_end - saved_start), random.randint(2, 4))
        ## TODO: answer is still in list format!!
        out.append({
            "video_id": annot["id"],
            "question_id": annot["id"] + "_" + str(self.id),
            "question": clean_text(random.choice(positive_feedback_prompts)(annot['task_name'], annot['video_time'])),
            "label": {"good_execution": clean_text(ast.literal_eval(annot['text']))},
            "video_path": saved_video_path,
            "duration": round(saved_end-saved_start,1), 
            "label_type": ["good_execution"],
            "verifier_type": ["llm"],
            "id": self.id
        })
        self.id += 1
        return out

    def run_tips_for_improvment(self, annot):
        out = []
        saved_start, saved_end, saved_video_path = self.save_video(annot['video_path'], annot['video_time'], annot['duration'])
        annot['video_time'] = round((annot['video_time'] - saved_start) / (saved_end - saved_start), random.randint(2, 4))
        ## TODO: answer is still in list format!!
        out.append({
            "video_id": annot["id"],
            "question_id": annot["id"] + "_" + str(self.id),
            "question": clean_text(random.choice(general_tips_prompts)(annot['task_name'], annot['video_time'])),
            "label": {"tips_for_improvment": clean_text(ast.literal_eval(annot['text']))},
            "video_path": saved_video_path,
            "duration":saved_end-saved_start,
            "label_type": ["tips_for_improvment"],
            "verifier_type": ["llm"],
            "id": self.id
        })
        self.id += 1
        return out    
        
    def run_proficiency_demonstrator(self, annot):
        out   = []
        saved_start, saved_end, saved_video_path = self.save_video(annot['video_path'], annot['video_time'], annot['duration'])
        annot['video_time'] = round((annot['video_time'] - saved_start) / (saved_end - saved_start), random.randint(2, 4))
        ## TODO: answer is still in list format!!
        out.append({
            "video_id": annot["id"],
            "question_id": annot["id"] + "_" + str(self.id),
            "question": clean_text(random.choice(prompts_for_demonstrator)(annot)),
            "label": {"proficiency_demonstrator": clean_text(annot['proficiency'])},
            "video_path": saved_video_path,
            "duration":saved_end-saved_start,
            "label_type": ["proficiency_demonstrator"],
            "verifier_type": ["llm"],
            "id": self.id
        })
        self.id += 1
        return out