import os
import spacy
import torch
import logging
import numpy as np
from openai import OpenAI
from fuzzywuzzy import fuzz
from spacy.matcher import Matcher
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False
        
def convert_xywh_to_xyxy(box):
    """
    Convert bounding box format from (x_center, y_center, width, height) to (xmin, ymin, xmax, ymax).

    Args:
    box (list): A list of four floats representing the box in (x_center, y_center, width, height) format.

    Returns:
    list: A list of four floats representing the box in (xmin, ymin, xmax, ymax) format.
    """
    x_center, y_center, width, height = box
    xmin = x_center - (width / 2)
    ymin = y_center - (height / 2)
    xmax = x_center + (width / 2)
    ymax = y_center + (height / 2)
    return [xmin, ymin, xmax, ymax]

def convert_xyxy_to_xywh(box):
    """
    Convert bounding box format from (xmin, ymin, xmax, ymax) to (x_center, y_center, width, height).

    Args:
    box (list): A list of four floats representing the box in (xmin, ymin, xmax, ymax) format.

    Returns:
    list: A list of four floats representing the box in (x_center, y_center, width, height) format.
    """
    xmin, ymin, xmax, ymax = box
    width = xmax - xmin
    height = ymax - ymin
    x_center = xmin + (width / 2)
    y_center = ymin + (height / 2)
    return [x_center, y_center, width, height]

def box_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Args:
    box1 (list): A list of four floats [xmin, ymin, xmax, ymax] for the first box.
    box2 (list): A list of four floats [xmin, ymin, xmax, ymax] for the second box.

    Returns:
    float: The IoU of the two bounding boxes.
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No overlap

    # The area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # The area of both AABBs
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute the IoU by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the interesection area
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def is_repetitive(output_text):
    """
    Checks if significant parts of the output text are repeated.
    
    Args:
    output_text (str): The output text from the model.

    Returns:
    bool: True if there are repetitions, False otherwise.
    """
    # Clean and split the text into sentences or segments
    segments = re.split(r'\.|\?\n|\!\n', output_text)  # Split by sentence terminators.
    segments = [seg.strip() for seg in segments if seg.strip()]  # Remove empty and strip spaces
    
    seen = set()  # Set to track unique segments
    for segment in segments:
        if segment in seen:
            return True  # Repetition found
        seen.add(segment)
    
    return False


system_message = """
System Message:
You are a judge, whose task it is to decide if a video model's text prediction matches the answer. We care about both the content and ordering of the events described. 

Example 1:
Prediction: Sure, let's go through the video step-by-step. First, we see a soccer player dribbling towards the goal. Then, the players are passing the soccer ball to each other. Next, the soccer players are celebrating, which suggests that they have scored a goal or are about to score. Finally, the goalkeeper is seen saving a shot, which implies that the opposing team is trying to score against them. Based on this sequence of events, we can conclude that the video captures a soccer game in progress, with the players passing the ball, celebrating, and trying to score against the opposing team. 
Answer: the soccer player is dribbling towards the goal->the players are passing the soccer ball->the soccer players are celebrating->the goalkeeper is saving a shot
Judge's Decision: Yes

Example 2:
Prediction: the players are passing the soccer ball to each other. Next, the soccer player is dribbling towards the goal. Then, the soccer players are celebrating, which suggests that they have scored a goal or are about to score. Finally, the goalkeeper is seen saving a shot, which implies that the opposing team is trying to score against them. 
Answer: the soccer player is dribbling towards the goal->the players are passing the soccer ball->the soccer players are celebrating->the goalkeeper is saving a shot
Judge's Decision: No
"""

llm_prompt = lambda pred, answer: (
    f"You are a judge, whose task it is to decide if a video model's text prediction matches the answer.\n\n"
    f"Prediction: {pred}\n"
    f"Answer: {answer}\n"
    "Does the prediction accurately reflect the correct sequence and content of events in the answer? Enter 'Yes' or 'No'."
)

llm_prompt_simple = lambda pred, answer: (
    f"Prediction: {pred}\n"
    f"Answer: {answer}\n"
    "Is the sequence of events in the prediction accurate and in the correct order as described in the answer? Yes or No."
)

llm_verifier_configs = {
    "qwen": {"type":'qwen', "model": "Qwen/Qwen1.5-MoE-A2.7B-Chat",  "torch_dtype":"auto", "device_map":"auto"},
    "chat_gpt": {"type": "chat_gpt", "api_key":"", "gpt_model":""},
    "none": {}
    }


import re

def str_to_float(s):
    """
    Converts a string to a float if it contains exactly one valid floating-point number.
    Returns None if the string does not contain exactly one float or contains additional characters.

    Args:
    s (str or None): The string to convert.

    Returns:
    float or None: The converted float if successful, None otherwise.
    """
    if s is None:
        return None

    # Regular expression to match a floating-point number
    # This regex matches optional leading/trailing spaces, an optional sign, digits, optional decimal point, and digits
    float_regex = r'^\s*[-+]?\d*\.?\d+\s*$'

    # Check if the string matches the regex for a single floating-point number
    if re.match(float_regex, s):
        try:
            return float(s)
        except ValueError:
            # Log or handle the unlikely case where conversion fails despite regex match
            print(f"Conversion failed for: {s}")
            return None
    else:
        return None




class Verifier(object):
    def __init__(self, 
                 llm_verifier_config = llm_verifier_configs['none'],
                 llm_enc = 'all-MiniLM-L6-v2'
                ):
        self.setup_llm_verifier(llm_verifier_configs)
        # installation: pip install spacy, python -m spacy download en_core_web_sm
        self.nlp = spacy.load("en_core_web_sm") 
        self.embedding_model = SentenceTransformer(llm_enc)
        self.classifier = pipeline(model="facebook/bart-large-mnli")
        
        self.matcher = Matcher(self.nlp.vocab)
        pattern_same = [{"LEMMA": "be"}, {"LOWER": "the"}, {"LOWER": "same"}]
        pattern_not_same = [{"LEMMA": "be"}, {"LOWER": "not"}, {"LOWER": "the"}, {"LOWER": "same"}]
        self.matcher.add("ARE_THE_SAME", [pattern_same])
        self.matcher.add("ARE_NOT_THE_SAME", [pattern_not_same])

        self.pred_parse = PredictionParser()
    
    def setup_llm_verifier(self, config):
        if config is None:
            config = {}
        if config.get('type') == 'qwen':
            self.model = AutoModelForCausalLM.from_pretrained(config["model"], torch_dtype=config.get("torch_dtype", "auto"), device_map=config.get("device_map", "auto"))
            self.tokenizer = AutoTokenizer.from_pretrained(config["model"])
            self.llm = True
        elif config.get('type') == 'chat_gpt':
            self.client = OpenAI(api_key=config["api_key"])
            self.gpt_model = config["gpt_model"]
            self.llm = True
        elif config.get('type') == 'model_server':
            self.client = config["some_config"]
            self.llm = True
        else:
            self.llm = False
            logging.warning('No LLM Verifier configuration provided. LLM label verification will default to unordered sentence matching via embeddings.')
            
    def word_matching(self, data, words_cls, threshold=85):
        """
        Check if all important words from a class description are sufficiently similar
        to any words in a given dataset, based on a similarity threshold.
    
        Parameters:
        data (str): The string to match against.
        words_cls (str or any convertible type): A string or object that can be converted to a string of class descriptions.
        threshold (int): The minimum similarity percentage required for a match.
    
        Returns:
        bool: True if all non-common words in words_cls are found in data above the similarity threshold, False otherwise.
        """
        def remove_stopwords(text):
            doc = self.nlp(text)
            filtered_words = {token.text.lower() for token in doc if not token.is_stop and len(token.text)>2}
            return filtered_words

        def word_in_word_set(word_cls, words_pred, threshold):
            max_sim = max([fuzz.ratio(word_cls, word_pred) for word_pred in words_pred])
            return max_sim > threshold
            
        # Handle non-string input for words_cls
        if not isinstance(words_cls, str):
            words_cls = str(words_cls) if not hasattr(words_cls, 'item') else str(words_cls.item())
        
        words_cls = remove_stopwords(words_cls)
        words_pred = remove_stopwords(data)
        
        # Check each word in words_cls against all words in data
        for word_cls in words_cls:
            if not word_in_word_set(word_cls, words_pred, threshold):
                return False  # If no word matches for any word_cls, return False immediately
        
        return True  # All words matched successfully above the threshold

    def normalized_float(self, predicted, gt, tolerance=0.1):
        """
        Evaluates the accuracy of the predicted float against the ground truth float
        
        Parameters:
        predicted (float): The float predicted by the LLM, normalized [0, 1]
        gt (float): The ground truth, normalized [0, 1]
        tolerance (float): The acceptable difference between the predicted and GT.
    
        Returns:
        bool: True if the difference is within the tolerance, False otherwise.
        """
        if predicted is None:
            return False
            
        if not isinstance(predicted, list):
            predicted = [predicted]

        mid_difference = tolerance * 10
        for pred in predicted:
            difference = abs(pred - gt)
            mid_difference = min(mid_difference, difference)
            
        return mid_difference <= tolerance

    def abs_float(self, predicted, gt, tolerance=0.1):
        """
        Evaluates the accuracy of the predicted float against the ground truth float
        
        Parameters:
        predicted (float): The float predicted by the LLM, normalized [0, 1]
        gt (float): The ground truth, normalized [0, 1]
        tolerance (float): The acceptable difference between the predicted and GT.
    
        Returns:
        bool: True if the difference is within the tolerance, False otherwise.
        """
        if not isinstance(predicted, list):
            predicted = [predicted]

        mid_difference = tolerance * 10
        for pred in predicted:
            difference = abs(pred - gt) / gt
            mid_difference = min(mid_difference, difference)       

        return mid_difference <= tolerance
        
    def two_dim_range(self, pred, gt, threshold=0.5):
        """
        Calculate the Intersection over Union (IoU) of two-dimensional ranges.
        Useful for: calcuating the IoU of bbox prediction

        Parameters:
        - pred (tuple or list): [x,y,w,h], the predicted 2d range
        - gt (tuple or list): [X,Y,W,H], the ground truth time interval.

        where x,y are the center of the prediction and W,H the hight&width, respectively.

        Returns:
        - float: The IoU of the two ranges, or 0 if there is no overlap.
        """
        if None in pred or None in gt:
            return False
        pred = convert_xywh_to_xyxy(pred)
        gt = convert_xywh_to_xyxy(gt)
        print('two_dim_range',  box_iou(pred, gt)) ###
        return box_iou(pred, gt) > threshold

    def one_dim_range(self, pred, gt, threshold=0.5):
        """
        Calculate the Intersection over Union (IoU) of one-dimensional ranges.
        Useful for: calcuating the IoU of temporal action localization

        Parameters:
        - pred (tuple or list): [t1, t2], the predicted interval.
        - gt (tuple or list): [T1, T2], the ground truth interval.

        Returns:
        - float: The IoU of the two intervals, or 0 if there is no overlap.
        """
        if None in pred or None in gt:
            return False
        # Calculate the intersection
        intersection_start = max(pred[0], gt[0])
        intersection_end = min(pred[1], gt[1])
        intersection = max(0, intersection_end - intersection_start)

        # Calculate the union
        union_start = min(pred[0], gt[0])
        union_end = max(pred[1], gt[1])
        union = union_end - union_start
        
        print('one_dim_range', intersection, union) ###
        # Early exit if there is no intersection
        if intersection == 0:
            return 0

        # Calculate and return IoU
        return intersection / union > threshold
        
    def split_into_subsentences(self, text):
        """
        Splits text into sub-sentences using dependency parsing.
        """
        doc = self.nlp(text)
        sub_sentences = []
        for sent in doc.sents:
            sub_sentence = []
            for token in sent:
                sub_sentence.append(token.text)
                if token.dep_ in ('cc', 'punct') and sub_sentence:
                    sub_sentences.append(' '.join(sub_sentence).strip())
                    sub_sentence = []
            if sub_sentence:
                sub_sentences.append(' '.join(sub_sentence).strip())
        return [s for s in sub_sentences if len(s)>1]
    
    def split_into_sentences(self, text):
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]

    def eval_sim_matrix(self, similarity_matrix, order_matters, similarity_threshold):
        if order_matters:
        # Calculate cosine similarities and verify sequence order
            sequence_order_correct = True
            last_max_index = -1
            for similarities in similarity_matrix:
                # Calculate similarities
                max_similarity, max_index = similarities.max(dim=0)
                
                # Check if the maximum similarity index is in the correct order
                if max_index < last_max_index or max_similarity < similarity_threshold:
                    sequence_order_correct = False
                    break
                last_max_index = max_index
        else:
            
            # Verify that each correct sentence has at least one predicted sentence with a similarity above the threshold
            max_similarity_scores, _ = similarity_matrix.max(dim=1)
            sequence_order_correct = (max_similarity_scores >= similarity_threshold).all()
            
        return sequence_order_correct

    def sequence_with_embeddings(self, prediction, correct_sequence, order_matters, similarity_threshold = 0.1):
        """
        Verify that the prediction contains all elements of the correct sequence in the same semantic order (if order matters)
        or simply present (if order does not matter) using sentence embeddings.

        Parameters:
        prediction (str): The prediction text containing a sequence of actions.
        correct_sequence (list): The correct sequence of actions to verify against.
        order_matters (bool): Specifies whether the order of actions matters or not.

        Returns:
        bool: True if the sequence is correct semantically and in order (or simply present), False otherwise.
        """
        # Extract sentences from the prediction
        if not isinstance(correct_sequence, list):
            correct_sequence = self.split_into_sentences(correct_sequence)
            
        predicted_sentences = self.split_into_sentences(prediction)
        if not predicted_sentences or not correct_sequence:
            logging.warning('sequence_with_embeddings: malformed input/output. aborting.')
            return False 
            
        # Convert sentences to embeddings
        correct_embeddings = self.embedding_model.encode(correct_sequence)
        predicted_embeddings = self.embedding_model.encode(predicted_sentences)
        similarity_matrix = util.pytorch_cos_sim(correct_embeddings, predicted_embeddings)
        return self.eval_sim_matrix(similarity_matrix, order_matters, similarity_threshold)

    def sequence_with_cls(self, prediction, correct_sequence, order_matters, similarity_threshold = 0.1):
        """
        Verify that the prediction contains all elements of the correct sequence in the same semantic order (if order matters)
        or simply present (if order does not matter) using sentence embeddings.

        Parameters:
        prediction (str): The prediction text containing a sequence of actions.
        correct_sequence (list): The correct sequence of actions to verify against.
        order_matters (bool): Specifies whether the order of actions matters or not.

        Returns:
        bool: True if the sequence is correct semantically and in order (or simply present), False otherwise.
        """
        # Extract sentences from the prediction
        #predicted_sentences = self.split_into_sentences(prediction)
        if not isinstance(correct_sequence, list):
            correct_sequence = self.split_into_subsentences(correct_sequence)
            
        predicted_sentences = self.split_into_subsentences(prediction)
        if not predicted_sentences or not correct_sequence:
            logging.warning('sequence_with_embeddings: malformed input/output. aborting.')
            return False 

        pred = self.classifier(predicted_sentences, correct_sequence, multi_label=True)
        # Align the scores with the correct_sequence order
        similarity_matrix = torch.zeros((len(correct_sequence), len(predicted_sentences)))

        ## doing this because classifier shuffles the class labels
        for i, correct_label in enumerate(correct_sequence):
            for j, p in enumerate(pred):
                if correct_label in p['labels']:
                    idx = p['labels'].index(correct_label)
                    similarity_matrix[i, j] = p['scores'][idx]
        
        return self.eval_sim_matrix(similarity_matrix, order_matters, similarity_threshold)

    def same_or_not(self, pred, gt):
        """
        Determines if the last mention in the text indicates that the objects are the same or not.

        Parameters:
        text (str): The input text.

        Returns:
        bool: True if the last mention indicates the objects are the same, False otherwise.
        """
        doc = self.nlp(pred)
        matches = self.matcher(doc)
        if len(matches)==0:
            return False
        last_decision = None
        for match_id, start, end in matches:
            if self.nlp.vocab.strings[match_id] == "ARE_THE_SAME":
                last_decision = True
            elif self.nlp.vocab.strings[match_id] == "ARE_NOT_THE_SAME":
                last_decision = False

        if " not " in gt.lower():
            return not last_decision
        return last_decision

    def verify(self, annot):
        #if is_repetitive(annot['output']):
        #    return False   
        self.pred_parse.parse(annot)
        for verifier_type, label_type in zip(annot['verifier_type'], annot['label_type']):                
            if verifier_type == "temporal_range":
                success = self.one_dim_range(annot['output_extracted'][label_type], annot['label'][label_type])

            elif verifier_type == "float":        
                success = self.abs_float(annot['output_extracted'][label_type], annot['label'][label_type])
                
            elif verifier_type == "timestamp":        
                success = self.normalized_float(annot['output_extracted'][label_type], annot['label'][label_type])
            
            elif verifier_type == "bbox":        
                success = self.two_dim_range(annot['output_extracted'][label_type], annot['label'][label_type])

            elif verifier_type == "same_or_not":        
                success = self.same_or_not(annot['output_extracted'][label_type], annot['label'][label_type])

            elif verifier_type == "classifier":
                success = self.sequence_with_cls(annot['output_extracted'][label_type], annot['label'][label_type], order_matters=False, similarity_threshold = 0.6)

            elif verifier_type == "classifier_sequence":
                success = self.sequence_with_cls(annot['output_extracted'][label_type], annot['label'][label_type], order_matters=True, similarity_threshold = 0.25)
    
            elif verifier_type == "sentence_sequence":
                success = self.sequence_with_embeddings(annot['output_extracted'][label_type], annot['label'][label_type], order_matters=True, similarity_threshold = 0.25)
                
            elif verifier_type == "sentence" or (verifier_type == "llm" and not self.llm):
                #success = self.sequence_with_cls(annot['output_extracted'][label_type], annot['label'][label_type], order_matters=False, similarity_threshold = 0.67)
                success = self.sequence_with_embeddings(annot['output_extracted'][label_type], annot['label'][label_type], order_matters=False, similarity_threshold = 0.6)
                
            elif verifier_type in ['obj_compare', 'obj_identify', 'simple']:
                success = self.word_matching(annot['output_extracted'][label_type], annot['label'][label_type])
            
            elif verifier_type == "llm" and self.llm:
                success = self.llm_verify(annot['output_extracted'][label_type], annot['label'][label_type], order_matters=True)
            
            else:
                import ipdb; ipdb.set_trace()
                raise("unidentified verifier type:", verifier_type)

            if not success:
                return False
                
        return True


class PredictionParser(object):
    def __init__(self, 
                 timestamp_nlp = 'videostar/extractors_supp/timestamp_model'
                ):
        self.timestamp_model = spacy.load(timestamp_nlp)
        self.nlp = spacy.load("en_core_web_sm") 
        self.extractor = pipeline("question-answering")
        
    def extract_temporal_range(self, annot):
        """
        Extracts the last occurrences of START_TIME and END_TIME entities from the given text.
    
        Parameters:
        nlp (spacy.Language): The trained NLP model.
        text (str): The input text from which to extract the times.
    
        Returns:
        tuple: A tuple containing the last start time and the last end time as strings.
               Returns (None, None) if either is not found.
        """
        tmp = self.extractor(context=annot['output'], question=annot['question'])
        if tmp['score']>0.6:
            text = tmp['answer']
        else:
            text = annot['output']
            
        doc = self.timestamp_model(text)
        timestamps = [float(ent.text) for ent in doc.ents if ent.label_ == 'TIME' and is_float(ent.text)]
    
        # Check from the last to the first to find the last pair where start_time < finish_time
        for i in range(len(timestamps) - 2, -1, -1):  # Start from the second to last element
            finish_time = timestamps[i + 1]
            start_time = timestamps[i]
            if start_time < finish_time and finish_time<1 and start_time>0:
                # Return the pair as soon as a valid pair is found
                return (start_time, finish_time)
        
        # If no suitable pair is found, return default values
        return None, None

    def extract_last_float(self, text):
        """
        Extracts the last floating-point number from the given text.
        
        Parameters:
        text (str): The input text from which to extract the last floating-point number.
        
        Returns:
        float or None: The last floating-point number found in the text, or None if no numbers are found.
        """
        # Regular expression to find floating-point numbers
        float_regex = r"[-+]?\d*\.\d+|[-+]?\d+"  # Matches both integers and floats, including signed ones
    
        # Find all matches of the regex in the text
        matches = re.findall(float_regex, text)
    
        # Check if we have any matches
        if matches:
            # Convert the last match to float and return
            return float(matches[-1])
        else:
            # Return None if no matches are found
            return None

    def extract_float(self, text):
        """
        Extracts the last floating-point number from the given text.
        
        Parameters:
        text (str): The input text from which to extract the last floating-point number.
        
        Returns:
        float or None: The last floating-point number found in the text, or None if no numbers are found.
        """
        # Regular expression to find floating-point numbers
        float_regex = r"[-+]?\d*\.\d+|[-+]?\d+"  # Matches both integers and floats, including signed ones
    
        # Find all matches of the regex in the text
        matches = re.findall(float_regex, text)
    
        # Check if we have any matches
        if matches:
            # Convert the last match to float and return
            return [float(m) for m in matches]
        else:
            # Return None if no matches are found
            return None
    
    def extract_timestamp(self, text):
        """
        Extracts the timestamp entities from the given text.
    
        Parameters:
        nlp (spacy.Language): The trained NLP model.
        text (str): The input text from which to extract the times.
    
        Returns:
        tuple: float
        """
        doc = self.timestamp_model(text)
        timestamps = [float(ent.text) for ent in doc.ents if ent.label_ == 'TIME' and is_float(ent.text)]
        
        if len(timestamps)>0:
            return timestamps[-1]
            
        return None

    def extract_bbox(self, text):
        """
        Extracts bounding boxes from the given text.
    
        Parameters:
        text (str): The input text containing bounding boxes.
    
        Returns:
        list: A list of bounding boxes, each represented as a list of floats.
        """
        pattern = r"\[(\d+\.\d+,\s*\d+\.\d+,\s*\d+\.\d+,\s*\d+\.\d+)\]"
        matches = re.findall(pattern, text)
        if not matches:
            return [None, None, None, None]  # Return None if no bounding boxes are found
    
        # Convert the last bbox string to a list of floats
        last_bbox = matches[-1].split(',')
        bbox_list = [str_to_float(num.strip()) for num in last_bbox]
        
        return bbox_list

    def extract_from_question(self, annot):
        """
        Extracts the last object in the given text
    
        Parameters:
        text (str): The input text
    
        Returns:
        str: the relevant label
        """
        return self.extractor(context=annot['output'], question=annot['question'])['answer']
    
    def parse(self, annot):
        annot['output_extracted'] = {}
        for verifier_type, label_type in zip(annot['verifier_type'], annot['label_type']):
            if verifier_type == "temporal_range":
                annot['output_extracted'][label_type] = self.extract_temporal_range(annot)
            elif verifier_type == "timestamp":
                annot['output_extracted'][label_type] = self.extract_last_float(annot['output'])
            elif verifier_type == "float":
                annot['output_extracted'][label_type] = self.extract_float(annot['output'])
            elif verifier_type == "bbox":
                annot['output_extracted'][label_type] = self.extract_bbox(annot['output'])
            elif verifier_type in ["obj_identify"] or label_type in ["question_from_answer", "answer_question", "action_after_action", "action_before_action"]:
                annot['output_extracted'][label_type] = self.extract_from_question(annot)
            else:
                annot['output_extracted'][label_type] = annot['output']
                
