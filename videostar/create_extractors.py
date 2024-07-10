import spacy
from spacy.tokens import DocBin, Span
from openai import OpenAI
from tqdm import tqdm
import json
import re
from spacy.training import Example
import random


def extract_numbers(answer):
    """
    Extracts two numbers from the given string.
    
    Parameters:
    answer (str): The input string containing two numbers.
    
    Returns:
    tuple: A tuple containing two numbers as floats, or None if not enough numbers are found.
    """
    # Regex pattern to find numbers (including integers and floats)
    pattern = r"\d+\.\d+|\d+"
    
    # Find all occurrences of the pattern
    numbers = re.findall(pattern, answer)
    
    # Convert extracted strings to floats
    numbers = [float(num) for num in numbers]
    
    # Return the first two numbers if available
    if len(numbers) >= 2:
        return numbers[0], numbers[1]
    else:
        return False, False

def save_training_data(training_data, filename):
    with open(filename, 'w') as f:
        json.dump(training_data, f)
    print(f"Training data saved to {filename}")

def load_training_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def save_model(nlp, path):
    nlp.to_disk(path)
    print(f"Model saved to {path}")

def load_model(path):
    return spacy.load(path)


def check_and_correct_alignment(nlp, text, entities):
    doc = nlp.make_doc(text)
    valid_entities = []
    for start, end, label in entities:
        span = doc.char_span(start, end, label=label)
        if span is not None:
            valid_entities.append((span.start_char, span.end_char, span.label_))
        else:
            print(f"Skipping entity due to misalignment: {text[start:end]} ({start}, {end})")
    return valid_entities

def annotate_training_data(training_data, nlp):
    annotated_data = []
    for text, start, end in training_data:
        entities = [
            (text.find(str(start)), text.find(str(start)) + len(str(start)), 'START_TIME'),
            (text.find(str(end)), text.find(str(end)) + len(str(end)), 'END_TIME')
        ]
        entities = check_and_correct_alignment(nlp, text, entities)
        if entities:
            annotated_data.append((text, {'entities': entities}))
    return annotated_data
    
def train_custom_ner(nlp, train_data, train_dataset_max_size=5000, nepochs=15):
    annotated_data = annotate_training_data(train_data, nlp)
    random.shuffle(annotated_data)
    if len(annotated_data)> train_dataset_max_size:
        annotated_data = annotated_data[:train_dataset_max_size]
        
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    for _, annotations in annotated_data:
        for ent in annotations['entities']:
            ner.add_label(ent[2])

    optimizer = nlp.initialize()
    for itn in range(nepochs):
        random.shuffle(annotated_data)
        losses = {}
        for text, annotations in annotated_data:
            doc = nlp.make_doc(text)
            if annotations['entities']:
                example = Example.from_dict(doc, annotations)
                nlp.update([example], sgd=optimizer, losses=losses)
        print(f"Losses at iteration {itn}: {losses}")
    return nlp
    

def create_temporal_sampler():
    num_samples = 5000
    lmm_generated_reasoning = "./data/STARB/star_annotations/STaR0/lmm_reasoning.json"

    with open(lmm_generated_reasoning , 'r') as file:
        anns = json.load(file)
        
    anns = [a for a in anns if 'start' in a['answer'] and 'end' in a['answer']]
    anns_training = []
    for ann in anns:
        start, end = extract_numbers(ann['answer'])
        if end:
            if str(start) in ann['output'] and str(end) in ann['output']:
                anns_training.append((ann['output'], start, end))    
    
    refined_training = [(t,s,e) for (t,s,e) in  training_data if str(s) in t and str(e) in t]

    save_training_data(anns_training, "/extractors_supp/temporal_range_data.json")
        
    # Create a blank English model 
    nlp = spacy.blank("en_core_web_lg")
    
    # Train the model
    nlp = train_custom_ner(nlp, anns_training, num_samples)
    save_model(nlp, "/extractors_supp/temporal_range_model")


def create_bbox_sampler():
    num_samples = 5000
    lmm_generated_reasoning = "./data/STARB/star_annotations/STaR0/lmm_reasoning.json"

    with open(lmm_generated_reasoning , 'r') as file:
        anns = json.load(file)
        
    pattern = r"\[\s*[^\[\],]+\s*,\s*[^\[\],]+\s*,\s*[^\[\],]+\s*,\s*[^\[\],]+\s*\]"
    anns = [a for a in anns if bool(re.search(pattern,  a['answer']))]
    anns_training = []
    for ann in anns:
        start, end = extract_numbers(ann['answer'])
        if end:
            if str(start) in ann['output'] and str(end) in ann['output']:
                anns_training.append((ann['output'], start, end))    
    
    refined_training = [(t,s,e) for (t,s,e) in  training_data if str(s) in t and str(e) in t]

    save_training_data(anns_training, "/extractors_supp/temporal_range_data.json")
        
    # Create a blank English model
    nlp = spacy.blank("en_core_web_lg")
    
    # Train the model
    nlp = train_custom_ner(nlp, anns_training, num_samples)
    save_model(nlp, "/extractors_supp/temporal_range_model")






