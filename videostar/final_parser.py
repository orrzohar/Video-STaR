from openai import OpenAI
import random, json, os
import ast


gpt_model = "gpt-3.5-turbo-0125"
SYSTEM_MESSAGE = """
You are a helpful assistant, and are  to perform only one task. You will be provided a question, ground-truth label, and predicted answer. 
You are to:

1. Rephrase the generated answer into a concise chain of thought process, finishing off with the answer to the question
2. Provide a direct answer to the question.

In both cases, if the answer is incorrect, fix it. 
You must respond in python list format, i.e.: ["chain of thought answer", "direct answer"]
"""

cot_prompt = [
    "Please explain your reasoning step by step.",
    "Can you break down the process for me?",
    "Walk me through your thought process, please.",
    "Detail your approach from start to finish.",
    "Could you elaborate on how you reached that conclusion?",
    "Describe the sequence of actions you would take.",
    "Explain the rationale behind each decision.",
    "How would you tackle this problem, step by step?",
    "Please provide a detailed explanation of your solution.",
    "Step-by-step, how did you come to this understanding?",
    "What sequence of steps would you follow here?",
    "Explain each phase of your approach.",
    "What are the logical steps you would follow to address this?",
    "Please explain the stages of your thinking process.",
    "How do you systematically approach this problem?",
    "Detail the incremental steps you took to reach the solution.",
    "Explain your thought sequence in tackling this problem.",
    "Please sequentially outline how you would solve this.",
    "Walk through each stage of your reasoning.",
    "Break down your analytical process, step by step."
]

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


class FinalParser(object):
    def __init__(self, api_key, train_path):
        # Assuming an API client setup; adjust based on actual usage
        self.client = OpenAI(api_key=api_key)  # Placeholder for actual client initialization
        with open(train_path, 'r') as file:
            all_labels = json.load(file)

        label_types = {str(l['label_type']) for l in all_labels}

        self.train_examples = {}
        for label_type in label_types:
            self.train_examples[label_type] = [label_dict_to_string(l['label']) for l in all_labels if l['label_type']==label_type]

    def summerize_cot(self, question, label, prediction):
        response = self.client.chat.completions.create(
            model=gpt_model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_MESSAGE
                    },
                {
                    "role": "user",
                    "content": f"Question: {question}\nLabel: {label}\nPrediction: {prediction}"
                    },
                ],
            temperature=0, 
            max_tokens=500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        data_list = ast.literal_eval(response.choices[0].message.content)
        return data_list
        
    def open_ended_qa(self, example):
        cot, a = self.summerize_cot(example['conversations'][0]['value'], 
                                    label_dict_to_string(example['label']), 
                                    example['conversations'][1]['value'])
        out = []
        exp = example.copy()
        exp['qa_type'] = "cot"
        exp['conversations'][0]['value'] += " "+ random.choice(cot_prompt)
        exp['conversations'][1]['value'] = cot
        out.append(exp)
        
        exp = example.copy()
        exp['qa_type'] = "qa"
        exp['conversations'][1]['value'] = a
        out.append(exp)
        return out

    def mc_question(self, question, gt, wrong_options):
        # Define different formatting styles
        format_options = {
            '(A)': '({}) {}',     # (A) Paris
            '(a)': '({}) {}',     # (a) Paris
            'A.': '{}. {}',       # A. Paris
            'a.': '{}. {}',       # a. Paris
            '(1)': '({}) {}',     # (1) Paris
            '1.': '{}. {}'        # 1. Paris
        }
    
        # Randomly choose a formatting style
        format_style, chosen_format = random.choice(list(format_options.items()))
    
        # Combine the correct answer with the wrong options
        options = wrong_options + [gt]
        # Shuffle the options to randomize their order
        random.shuffle(options)
        
        # Append formatted choices to the question
        question += " Choices are"
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if 'A' in format_style or 'a' in format_style else range(1, len(options) + 1)
        for idx, option in enumerate(options):
            question += " " + chosen_format.format(labels[idx], option)
        
        # Determine the correct answer label
        if 'A' in format_style or 'a' in format_style:
            answer_label = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[options.index(gt)]
        else:
            answer_label = str(options.index(gt) + 1)
        
        # Return the question and the correct answer label
        return question, answer_label
    
    def multi_choice_qa(self, example):
        out = []
        if example['label_type'] in [["sentence_sequence"], ["action_sequence"]]:
            labels = [label_dict_to_string(l) for l in self.train_examples[example['label_type']] if l != example['label']]
            random.shuffle(labels)
            if len(labels)>4:
                wrong_options = labels[:3]   # Select up to 4 wrong options randomly
            else:
                wrong_options = labels
            if gt_label_string in wrong_options:
                wrong_options.remove(gt_label_string)  # Ensure the ground truth is not in the wrong options

        else:
            possible_options = [l for l in self.train_examples[str(example['label_type'])] if l != gt_label_string]
            random.shuffle(possible_options)
            wrong_options = possible_options[:3]
                    
        exp = example.copy()
        q, a = self.mc_question(exp['conversations'][0]['value'], 
                               label_dict_to_string(example['label']),
                               wrong_options)
        
        exp['conversations'][0]['value'], exp['conversations'][1]['value'] = q, a
        out.append(exp)
        return out
        

    def __call__(self, example):
        """Determine the appropriate method to use based on the label type and call it."""
        label_type = example['label_type']
        out = []
        out += self.multi_choice_qa(example)
        out += self.open_ended_qa(example)
        return out



