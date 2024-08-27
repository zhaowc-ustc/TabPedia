# Copyright (c) OpenMMLab. All rights reserved.
import base64
import copy
import io
from io import BytesIO
from itertools import chain

import numpy as np
import requests
from PIL import Image

from tools.utils import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX


def get_bos_eos_token_ids(tokenizer):
    if tokenizer.__class__.__name__ in [
            'QWenTokenizer', 'QWen2Tokenizer', 'Qwen2TokenizerFast'
    ]:
        bos_token_id = []
        eos_token_id = tokenizer.eos_token_id
        assert eos_token_id is not None, \
            'Please set eos_token for Qwen tokenizer!'
    elif tokenizer.__class__.__name__ == 'ChatGLMTokenizer':
        bos_token_id = [64790, 64792]
        eos_token_id = tokenizer.eos_token_id
    else:
        bos_token_id = tokenizer.bos_token_id
        eos_token_id = tokenizer.eos_token_id
    if isinstance(bos_token_id, int):
        bos_token_id = [bos_token_id]
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    return bos_token_id, eos_token_id


def encode_fn(example,
              tokenizer,
              max_length,
              input_ids_with_output=True,
              with_image_token=False):
    """We only support the following three scenarios:

    1. Incremental pretraining dataset.
        example['conversation'] = [
                {
                    'input': '',
                    'output': '### Human: Can you write xxx'
                }
            ]

    2. Single-turn conversation dataset.
        example['conversation'] = [
                {
                    'input': 'Give three tips for staying healthy.',
                    'output': '1.Eat a balanced diet xxx'
                }
            ]

    3. Multi-turn conversation dataset.
        example['conversation'] = [
                {
                    'input': 'Give three tips for staying healthy.',
                    'output': '1.Eat a balanced diet xxx'
                },
                {
                    'input': 'Please expand on the second point.',
                    'output': 'Here is an expanded explanation of the xxx'
                }
            ]
    """
    bos_token_id, eos_token_id = get_bos_eos_token_ids(tokenizer)
    is_multi_turn_conversation = len(example['conversation']) > 1
    if is_multi_turn_conversation:
        assert input_ids_with_output

    input_ids, labels = [], []
    next_needs_bos_token = True
    for single_turn_conversation in example['conversation']:
        input = single_turn_conversation['input']
        if DEFAULT_IMAGE_TOKEN in input and with_image_token:
            chunk_encode = [
                tokenizer.encode(chunk, add_special_tokens=False)
                for chunk in input.split(DEFAULT_IMAGE_TOKEN)
            ]
            assert len(chunk_encode) == 2
            input_encode = []
            for idx, cur_chunk_encode in enumerate(chunk_encode):
                input_encode.extend(cur_chunk_encode)
                if idx != len(chunk_encode) - 1:
                    input_encode.append(IMAGE_TOKEN_INDEX)
        else:
            input_encode = tokenizer.encode(input, add_special_tokens=False)
        if next_needs_bos_token:
            input_ids += bos_token_id
            labels += [IGNORE_INDEX] * len(bos_token_id)
        input_ids += input_encode
        labels += [IGNORE_INDEX] * len(input_encode)
        if input_ids_with_output:
            # Add output
            output_with_loss = single_turn_conversation.get(
                'output_with_loss', True)
            output = single_turn_conversation['output']
            output_encode = tokenizer.encode(output, add_special_tokens=False)
            input_ids += output_encode
            if output_with_loss:
                labels += copy.deepcopy(output_encode)
            else:
                labels += [IGNORE_INDEX] * len(output_encode)
            # Add EOS_TOKEN (with loss)
            if single_turn_conversation.get('need_eos_token', True):
                next_needs_bos_token = True
                input_ids += eos_token_id
                if output_with_loss:
                    labels += copy.deepcopy(eos_token_id)
                else:
                    labels += [IGNORE_INDEX] * len(eos_token_id)
            else:
                next_needs_bos_token = False
            # Add SEP (without loss)
            sep = single_turn_conversation.get('sep', '')
            if sep != '':
                sep_encode = tokenizer.encode(sep, add_special_tokens=False)
                input_ids += sep_encode
                labels += [IGNORE_INDEX] * len(sep_encode)

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    return {'input_ids': input_ids, 'labels': labels}


class Packer:
    """Pack multiple pieces of data into one."""

    def __init__(self,
                 chunk_size=2048,
                 use_varlen_attn=False,
                 drop_last=False):
        self.chunk_size = chunk_size
        self.residual = {'input_ids': [], 'labels': []}
        self.use_varlen_attn = use_varlen_attn
        self.drop_last = drop_last
        if use_varlen_attn:
            self.residual_cumulative_len = [0]

    def get_cumulative_len(self, chunk_num):
        ptr_l = 0
        cumulative_len = []
        for chunk_idx in range(chunk_num):
            length_train = (chunk_idx + 1) * self.chunk_size
            ptr_r = np.searchsorted(
                self.residual_cumulative_len, length_train, side='left')
            if self.residual_cumulative_len[ptr_r] == length_train:
                cumulative_len_cur = \
                    self.residual_cumulative_len[ptr_l:ptr_r + 1]
                ptr_l = ptr_r + 1
            else:
                cumulative_len_cur = self.residual_cumulative_len[
                    ptr_l:ptr_r] + [length_train]
                ptr_l = ptr_r
            cumulative_len_cur = [
                num - chunk_idx * self.chunk_size for num in cumulative_len_cur
            ]
            if cumulative_len_cur[0] != 0:
                cumulative_len_cur = [0] + cumulative_len_cur

            cumulative_len.append(cumulative_len_cur)

        self.residual_cumulative_len = [
            num - length_train for num in self.residual_cumulative_len[ptr_l:]
        ]
        if len(self.residual_cumulative_len) == 0:
            self.residual_cumulative_len = [0]
        elif self.residual_cumulative_len[0] != 0:
            self.residual_cumulative_len = [0] + self.residual_cumulative_len

        return cumulative_len

    def get_indexes(self, cumulative_len):
        indexes = []
        for cumulative_len_cur in cumulative_len:
            index_cur = []
            for i in range(len(cumulative_len_cur) - 1):
                index_cur.extend(
                    list(
                        range(cumulative_len_cur[i + 1] -  # noqa: W504
                              cumulative_len_cur[i])))
            indexes.append(index_cur)
        return indexes

    def __call__(self, batch):
        concatenated_samples = {
            k: v + list(chain(*batch[k]))
            for k, v in self.residual.items()
        }

        if self.use_varlen_attn:
            for input_id in batch['input_ids']:
                self.residual_cumulative_len.append(
                    self.residual_cumulative_len[-1] + len(input_id))

        total_length = len(concatenated_samples[list(
            concatenated_samples.keys())[0]])

        if total_length >= self.chunk_size:
            chunk_num = total_length // self.chunk_size
            result = {
                k: [
                    v[i:i + self.chunk_size] for i in range(
                        0,
                        chunk_num *  # noqa: W504
                        self.chunk_size,
                        self.chunk_size)
                ]
                for k, v in concatenated_samples.items()
            }
            self.residual = {
                k: v[(chunk_num * self.chunk_size):]
                for k, v in concatenated_samples.items()
            }

            if self.use_varlen_attn:
                cumulative_len = self.get_cumulative_len(chunk_num)
                result['cumulative_len'] = cumulative_len
                result['indexes'] = self.get_indexes(cumulative_len)
        else:
            if self.drop_last:
                result = {k: [] for k, v in concatenated_samples.items()}
            else:
                result = {k: [v] for k, v in concatenated_samples.items()}

            self.residual = {k: [] for k in concatenated_samples.keys()}

            if self.use_varlen_attn:
                result['cumulative_len'] = [] if self.drop_last else [
                    self.residual_cumulative_len
                ]
                result['indexes'] = [] if self.drop_last else self.get_indexes(
                    [self.residual_cumulative_len])
                self.residual_cumulative_len = [0]

        return result


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image



import random


# 创建一个任务映射到其编号的字典
task_to_id = {
    "detection": 1,
    "recognition": 2,
    "spotting": 3,
    "box_query": 4,
    "reading": 5
}

terms = ["photo", "picture", "image", "snapshot", "photograph", "pic", "graphic", "shot", "frame", "capture"]
determiners = ["this", "the"]

# 识别QA对模板
recognition_templates = [
    "Extract all the text in <determiner> <term>.",
    "Can you identify the words in <determiner> <term>?",
    "Tell me the text you see in <determiner> <term>.",
    "What's written in <determiner> <term>?",
    "Decode the text from <determiner> <term>.",
    "Reveal the text in <determiner> <term> for me.",
    "Detail the text present in <determiner> <term>.",
    "Elaborate on the words in <determiner> <term>.",
    "Illuminate me with the text from <determiner> <term>.",
    "Narrate the words you find in <determiner> <term>.",
    "Can you spell out the text from <determiner> <term>?",
    "I'd like to know the words in <determiner> <term>.",
    "Dissect <determiner> <term> and tell me its text.",
    "I'm curious about the text in <determiner> <term>. What is it?",
    "Help me decipher the words in <determiner> <term>.",
    "Shed light on the text of <determiner> <term>.",
    "Convey the written part of <determiner> <term> to me.",
    "Identify the text in <determiner> <term>.",
    "What's the text in <determiner> <term>.",
    "Recognize all the text in <determiner> <term>."
]

# 检测QA对模板
detection_templates = [
    "Output all the text's locations in <determiner> <term> based on <T_IMG>.",
    "Where are the texts located in <determiner> <term> based on <T_IMG>?",
    "Locate the texts in <determiner> <term> for me based on <T_IMG>.",
    "Highlight the positions of all the texts in <determiner> <term> based on <T_IMG>.",
    "I need the coordinates of the texts in <determiner> <term> based on <T_IMG>.",
    "Show me the regions with texts in <determiner> <term> based on <T_IMG>.",
    "Determine the areas with text in <determiner> <term> based on <T_IMG>.",
    "Can you pinpoint the text locations in <determiner> <term> based on <T_IMG>?",
    "Map out the texts in <determiner> <term> for me based on <T_IMG>.",
    "Display the text positions in <determiner> <term> based on <T_IMG>.",
    "Give me the areas where texts are present in <determiner> <term> based on <T_IMG>."
]

# spotting的QA对模板
spotting_templates = [
    "Recognize all the text in <determiner> <term> and return their positions [x1, y1, x2, y2] based on <T_IMG>.",
    "Extract all the text in <determiner> <term> and return their coordinates in the format of [x1, y1, x2, y2] based on <T_IMG>.",
    "Identify the text in <determiner> <term> and return their [x1, y1, x2, y2] coordinates based on <T_IMG>.",
    "Identify the text in <determiner> <term> and return their positions in the format of [x1, y1, x2, y2] based on <T_IMG>.",
    "Recognize all the text in <determiner> <term> return their coordinates in the format of [x1, y1, x2, y2] based on <T_IMG>.",
    "Detect the text in <determiner> <term> and return their coordinates in the format of [x1, y1, x2, y2] based on <T_IMG>.",
    "Find all the text in <determiner> <term> and return their coordinates in the format of [x1, y1, x2, y2] based on <T_IMG>.",
    "Find all the text in <determiner> <term> and return their positions represented in the format of [x1, y1, x2, y2] based on <T_IMG>.",
    "Parse all the text in <determiner> <term> and return their coordinates in the format of [x1, y1, x2, y2] based on <T_IMG>."
]

# 框查询QA对模板
box_query_templates = [
    "What text is inside the region {bbox} in <determiner> <term> based on <T_IMG>?",
    "Tell me about the content in the area marked as {bbox} of <determiner> <term> based on <T_IMG>.",
    "I'm curious about the text in the section denoted by {bbox} of <determiner> <term> based on <T_IMG>.",
    "What does <determiner> <term> say in the area outlined by {bbox} based on <T_IMG>?",
    "Decipher the words within the space defined by {bbox} in <determiner> <term> based on <T_IMG>.",
    "Can you read the text from the section specified as {bbox} in <determiner> <term> based on <T_IMG>?",
    "Reveal the content of the region marked as {bbox} in <determiner> <term> based on <T_IMG>.",
    "Extract the words from the portion designated by {bbox} in <determiner> <term> based on <T_IMG>.",
    "I'd like to know the text from the highlighted box given by {bbox} in <determiner> <term> based on <T_IMG>.",
    "Decode the inscription inside the region defined by {bbox} in <determiner> <term> based on <T_IMG>."
]

# 读取QA对模板
reading_templates = [
    "Give a detailed reading of <determiner> <term>.",
    "Can you provide a full reading from <determiner> <term>?",
    "Read aloud the text from <determiner> <term>.",
    "Convey the entire content of <determiner> <term> to me.",
    "I'd like a comprehensive reading of <determiner> <term>.",
    "Recite the content from <determiner> <term> for me.",
    "Provide a transcript of the text from <determiner> <term>.",
    "I want to hear the text from <determiner> <term>.",
    "Narrate the entire text of <determiner> <term> for me.",
    "Please read out the full text from <determiner> <term>."
]

def apply_template(templates):
    chosen_template = random.choice(templates)
    chosen_determiner = random.choice(determiners)
    chosen_term = random.choice(terms)
    chosen_template = chosen_template.replace('<determiner>', chosen_determiner)
    chosen_template = chosen_template.replace('<term>', chosen_term)
    return chosen_template, chosen_term

def generate_qa(text_input, bbox_input, task_input, block_text_input=None):
    texts = text_input.split('\n')
    bboxes = bbox_input.split('\n')
    tasks = task_input.split('\n')

    if not texts and not bboxes:
        assert len(bboxes) == len(texts)

    # 根据输入的任务列表选择一个任务
    task_type = random.choice(tasks)
    qa_type = task_to_id[task_type]
    Q, term = apply_template(detection_templates if qa_type == 1 else 
                             recognition_templates if qa_type == 2 else 
                             spotting_templates if qa_type == 3 else 
                             box_query_templates if qa_type == 4 else 
                             reading_templates)

    if random.choice([True, False]):
        Q = "<image>\n" + Q
    else:
        Q = Q + "\n<image>"

    if qa_type == 1:  # 检测QA对
        A = f"Here is a list of all the locations in the {term}:\n" + '\n'.join(bboxes)

    elif qa_type == 2:  # 识别QA对
        A = f"Here is a list of all the text in the {term}:\n" + '\n'.join(texts)

    elif qa_type == 3:  # spotting的QA对
        A = f"Here is a list of all the text and locations in the {term}:\n" + '\n'.join(f"{text} {bbox}" for text, bbox in zip(texts, bboxes))

    elif qa_type == 4:  # 框查询QA对
        idx = random.randint(0, len(bboxes) - 1)  # 随机选择一个bbox
        chosen_bbox = bboxes[idx]
        Q = Q.replace("{bbox}", chosen_bbox)  # 在问题模板中替换bbox坐标
        A = f"Here is the text in the region {chosen_bbox} of the {term}:\n{texts[idx]}"

    elif qa_type == 5:  # 读取QA对
        assert block_text_input != None
        A = f"Here is a detailed reading from the {term}:\n{block_text_input}"

    return Q, A, task_type

# ========================================================================分割线======================================================================================================

# 创建一个任务映射到其编号的字典
element_task_to_id = {
    "LayoutDet": 1,
    "Rec": 2,
    "Spot": 3,
    "LayoutDet-SE": 1
}

terms = ["photo", "picture", "image", "snapshot", "photograph", "pic", "graphic", "shot", "frame", "capture"]
determiners = ["this", "the"]
element_types = {"picture":"figure", "table":"table", "formula":"formula", "catalogue":"catalogue", "reference":"reference", "code_block":"code_block"}

# 元素检测QA对模板
element_detection_templates = [
    "Output all the <element> element's locations in <determiner> <term> based on <T_IMG>.",
    "Where are the <element> elements located in <determiner> <term> based on <T_IMG>?",
    "Locate the <element> elements in <determiner> <term> for me based on <T_IMG>.",
    "Highlight the positions of all the <element> elements in <determiner> <term> based on <T_IMG>.",
    "I need the coordinates of the <element> elements in <determiner> <term> based on <T_IMG>.",
    "Show me the regions with <element> elements in <determiner> <term> based on <T_IMG>.",
    "Determine the areas with <element> element in <determiner> <term> based on <T_IMG>.",
    "Can you pinpoint the <element> element locations in <determiner> <term> based on <T_IMG>?",
    "Map out the <element> elements in <determiner> <term> for me based on <T_IMG>.",
    "Display the <element> element positions in <determiner> <term> based on <T_IMG>.",
    "Give me the areas where <element> elements are present in <determiner> <term> based on <T_IMG>."
]

# 元素spotting的QA对模板
element_spotting_templates = [
    "Recognize the elements including figure, table, formula, catalogue, reference and code_block in <determiner> <term> and return their positions [x1, y1, x2, y2] based on <T_IMG>.",
    "Extract the elements including figure, table, formula, catalogue, reference and code_block in <determiner> <term> and return their coordinates in the format of [x1, y1, x2, y2] based on <T_IMG>.",
    "Identify the elements including figure, table, formula, catalogue, reference and code_block in <determiner> <term> and return their [x1, y1, x2, y2] coordinates based on <T_IMG>.",
    "Identify the elements including figure, table, formula, catalogue, reference and code_block in <determiner> <term> and return their positions in the format of [x1, y1, x2, y2] based on <T_IMG>.",
    "Recognize the elements including figure, table, formula, catalogue, reference and code_block in <determiner> <term> return their coordinates in the format of [x1, y1, x2, y2] based on <T_IMG>.",
    "Detect the elements including figure, table, formula, catalogue, reference and code_block in <determiner> <term> and return their coordinates in the format of [x1, y1, x2, y2] based on <T_IMG>.",
    "Find the elements including figure, table, formula, catalogue, reference and code_block in <determiner> <term> and return their coordinates in the format of [x1, y1, x2, y2] based on <T_IMG>.",
    "Find the elements including figure, table, formula, catalogue, reference and code_block in <determiner> <term> and return their positions represented in the format of [x1, y1, x2, y2] based on <T_IMG>.",
    "Parse the elements including figure, table, formula, catalogue, reference and code_block in <determiner> <term> and return their coordinates in the format of [x1, y1, x2, y2] based on <T_IMG>."
]

# 元素框查询QA对模板
element_box_query_templates = [
    "What element is inside the region {bbox} in <determiner> <term> based on <T_IMG>?",
    "Tell me about the element in the area marked as {bbox} of <determiner> <term> based on <T_IMG>.",
    "Identify the element located within the region {bbox} in <determiner> <term> based on <T_IMG>.",
    "I'm curious about the element in the section denoted by {bbox} of <determiner> <term> based on <T_IMG>.",
    "I'd like to know the element from the highlighted box given by {bbox} in <determiner> <term> based on <T_IMG>.",
    "Recognize the element from the portion designated by {bbox} in <determiner> <term> based on <T_IMG>.",
    "I want to find out the element in the region {bbox} of <determiner> <term> based on <T_IMG>.",
    "Determine the element from the section specified by {bbox} in <determiner> <term> based on <T_IMG>."
]


def apply_element_template(templates, elements, ratio=0.2):
    chosen_template = random.choice(templates)
    chosen_determiner = random.choice(determiners)
    chosen_term = random.choice(terms)
    chosen_template = chosen_template.replace('<determiner>', chosen_determiner)
    chosen_template = chosen_template.replace('<term>', chosen_term)
    if '<element>' in chosen_template:
        # 以ratio的概率随机从所有的元素类型列表中采样一个元素类型，以（1-ratio）的概率随机从图片包含的元素类型列表中采样一个元素类型
        if random.uniform(0, 1) > ratio:
            chosen_element_type = random.choice(elements)
        else:
            chosen_element_type = random.choice(list(element_types.keys()))
        chosen_template = chosen_template.replace('<element>', element_types[chosen_element_type])
        return chosen_template, chosen_term, element_types[chosen_element_type], chosen_element_type
    return chosen_template, chosen_term, None, None


def generate_zeroelement_qa(task_input, val_mode, task_key):
    tasks = task_input.split('\n')
    elements = ["picture", "table", "formula", "catalogue", "reference", "code_block"]

    if not val_mode:
        # 根据输入的任务列表选择一个任务
        task_type = random.choice(tasks)
        qa_type = element_task_to_id[task_type]
        Q, term, element_type, text_name = apply_element_template(element_detection_templates if qa_type == 1 else 
                                        element_box_query_templates if qa_type == 2 else 
                                        element_spotting_templates, elements=elements, ratio=0.0 if task_type != 'LayoutDet-SE' else 0.0)
        if random.choice([True, False]):
            Q = "<image>\n" + Q
        else:
            Q = Q + "\n<image>"

        if qa_type == 1:  # 元素检测QA对
            A = f"There is no {element_type} element in the {term}.\n"
        elif qa_type == 2:  # 元素框查询QA对
            raise ValueError(f"Incorrect qa_type: {qa_type}!")
        elif qa_type == 3:  # 元素spotting的QA对
            A = f"There is no any element in the {term}.\n"
        else:
            raise ValueError(f"Unrecognized qa_type: {qa_type}!")
    else:
        task = task_key.split('-')[0]
        term = 'picture'
        if 'spot' in task:
            task_type = 'Spot'
            
            Q = f"Recognize the elements including figure, table, formula, catalogue, reference and code_block in this {term} and return their positions [x1, y1, x2, y2] based on <T_IMG>."
            Q = Q + "\n<image>"
            
            A = f"There is no any element in the {term}.\n"
        elif 'det' in task:
            task_type = 'LayoutDet'
            text_name = task_key.split('-')[1]
            element_type = element_types[text_name]
            
            Q = f"Output all the {element_type} element's locations in this {term} based on <T_IMG>."
            Q = Q + "\n<image>"

            A = f"There is no {element_type} element in the {term}.\n"
        else:
             raise ValueError(f"Unrecognized qa_type: {task}!")
    return Q, A, task_type


def generate_element_qa(text_input, bbox_input, task_input, val_mode, task_key, block_text_input=None):
    texts = text_input
    bboxes = bbox_input
    tasks = task_input.split('\n')

    if not texts and not bboxes:
        assert len(bboxes) == len(texts)

    if not val_mode:
        # 根据输入的任务列表选择一个任务
        task_type = random.choice(tasks)
        qa_type = element_task_to_id[task_type]
        Q, term, element_type, text_name = apply_element_template(element_detection_templates if qa_type == 1 else 
                                        element_box_query_templates if qa_type == 2 else 
                                        element_spotting_templates, texts, ratio=0.0 if task_type != 'LayoutDet-SE' else 0.0)

        if random.choice([True, False]):
            Q = "<image>\n" + Q
        else:
            Q = Q + "\n<image>"

        if qa_type == 1:  # 元素检测QA对
            selected_bboxes = []
            for i, text in enumerate(texts):
                if text == text_name:
                    selected_bboxes.append(bboxes[i])
            if len(selected_bboxes) > 0:
                A = f"Here is a list of all the locations of {element_type} element in the {term}:\n" + '\n'.join(selected_bboxes)
            else:
                A = f"There is no {element_type} element in the {term}.\n"

        elif qa_type == 2:  # 元素框查询QA对
            idx = random.randint(0, len(bboxes) - 1)  # 随机选择一个bbox
            chosen_bbox = bboxes[idx]
            Q = Q.replace("{bbox}", chosen_bbox)  # 在问题模板中替换bbox坐标
            A = f"Here is the element in the region {chosen_bbox} of the {term}:\n{element_types[texts[idx]]}"

        elif qa_type == 3:  # 元素spotting的QA对
            A = f"Here is a list of the elements and locations in the {term}:\n" + '\n'.join(f"{element_types[text]} {bbox}" for text, bbox in zip(texts, bboxes))

        else:
            raise ValueError(f"Unrecognized qa_type: {qa_type}!")
    else:
        task = task_key.split('-')[0]
        term = 'picture'
        if 'spot' in task:
            task_type = 'Spot'
            
            Q = f"Recognize the elements including figure, table, formula, catalogue, reference and code_block in this {term} and return their positions [x1, y1, x2, y2] based on <T_IMG>."
            Q = Q + "\n<image>"

            A = f"Here is a list of the elements and locations in the {term}:\n" + '\n'.join(f"{element_types[text]} {bbox}" for text, bbox in zip(texts, bboxes))
        
        elif 'det' in task:
            task_type = 'LayoutDet'
            text_name = task_key.split('-')[1]
            element_type = element_types[text_name]
            
            Q = f"Output all the {element_type} element's locations in this {term} based on <T_IMG>."
            Q = Q + "\n<image>"
            
            selected_bboxes = []
            for i, text in enumerate(texts):
                if text == text_name:
                    selected_bboxes.append(bboxes[i])

            if len(selected_bboxes) == 0: # 考虑如果图片中没有该元素的情况
                A = f"There is no {element_type} element in the {term}.\n"
            else:
                A = f"Here is a list of all the locations of {element_type} element in the {term}:\n" + '\n'.join(selected_bboxes)
        
        else:
             raise ValueError(f"Unrecognized qa_type: {task}!")

    return Q, A, task_type


# text_input = np.array(['formula'])
# bbox_input = np.array(['[345,557,78,944]'])
# task_input = "LayoutDet"
# _ = generate_element_qa(text_input, bbox_input, task_input)
# _ = generate_zeroelemesnt_qa(task_input="LayoutDet\nSpot")


# ============================================划分线==================================================

# 表格结构识别QA对模板
element_table_structure_templates = [
    "Recognize the structural information of the cropped table in <determiner> <term> based on <T_IMG>.",
    "Parse the structural information of the cropped table in <determiner> <term> based on <T_IMG>.",
    "Detect the structural information of the cropped table in <determiner> <term> based on <T_IMG>.",
    "Extract the structural information of the cropped table in <determiner> <term> based on <T_IMG>.",
    "Identify the structural information of the cropped table in <determiner> <term> based on <T_IMG>.",
    "Determine the structural information of the cropped table in <determiner> <term> based on <T_IMG>."
]


def generate_table_structure_qa(text_input, bbox_input, val_mode, task_key):
    texts = text_input
    bboxes = bbox_input

    if not texts and not bboxes:
        assert len(bboxes) == len(texts)

    if not val_mode:
        # generate Q 
        chosen_template = random.choice(element_table_structure_templates)
        chosen_determiner = random.choice(determiners)
        chosen_term = random.choice(terms)
        chosen_template = chosen_template.replace('<determiner>', chosen_determiner)
        chosen_template = chosen_template.replace('<term>', chosen_term)
        Q = chosen_template

        if random.choice([True, False]):
            Q = "<image>\n" + Q
        else:
            Q = Q + "\n<image>"
    else:
        # generate Q 
        Q = "Parse the structural information of the cropped table in this picture based on <T_IMG>."
        chosen_term = 'picture'
        Q = Q + "\n<image>"

    # generate A ; delete the bbox of the whole table
    A = f"The table structural information in the {chosen_term} is listed as following:\n" + '\n'.join(f"{text} {bbox}" for text, bbox in zip(texts[1:], bboxes[1:]))

    return Q, A


# ============================================划分线==================================================

# 表格查询QA对模板
element_table_query_templates = [
    "Parse the table structure within the region {bbox} in <determiner> <term> based on <T_IMG>.",
    "What is the structure of the table located in the {bbox} area in <determiner> <term> based on <T_IMG>?",
    "Analyze the table structure in the region {bbox} of <determiner> <term> based on <T_IMG>.",
    "Describe the specific structure of the table in the region {bbox} of <determiner> <term> based on <T_IMG>.",
    "I'm looking to find out the table structure within the designated {bbox} area in <determiner> <term> based on <T_IMG>.",
    "Examine the table structure within the region {bbox} of <determiner> <term> based on <T_IMG>."
]

def generate_table_query_qa(text_input, bbox_input, val_mode, task_key):
    texts = text_input
    bboxes = bbox_input

    if not texts and not bboxes:
        assert len(bboxes) == len(texts)

    if not val_mode:
        # generate Q 
        chosen_template = random.choice(element_table_query_templates)
        chosen_determiner = random.choice(determiners)
        chosen_term = random.choice(terms)
        chosen_template = chosen_template.replace('<determiner>', chosen_determiner)
        chosen_template = chosen_template.replace('<term>', chosen_term)
        Q = chosen_template

        if random.choice([True, False]):
            Q = "<image>\n" + Q
        else:
            Q = Q + "\n<image>"
    else:
        # generate Q 
        Q = "Parse the table structure within the region {bbox} in this picture based on <T_IMG>."
        chosen_term = 'picture'
        Q = Q + "\n<image>"

    # generate A ; delete the bbox of the whole table
    Q = Q.replace("{bbox}", bboxes[0])  # 在问题模板中替换bbox坐标
    A = f"The table structural information in the region {bboxes[0]} of the {chosen_term} is listed as following:\n" + '\n'.join(f"{text} {bbox}" for text, bbox in zip(texts[1:], bboxes[1:]))

    return Q, A


# ============================================划分线==================================================

# 表格VQA对模板

def generate_table_vqa_qa(question, answer, val_mode, task_key):
    q_list = question
    a_list = answer

    if not q_list and not a_list:
        assert len(q_list) == len(a_list)

    if not val_mode:
        # generate Q
        selected_id = random.choice([i for i in range(len(q_list))])
        Q = q_list[selected_id]

        if random.choice([True, False]):
            Q = "<image>\n" + Q
        else:
            Q = Q + "\n<image>"
            
        # generate A
        A = a_list[selected_id]
    else:
        # generate Q
        Q = q_list[0]
        Q = Q + "\n<image>"

        # generate A
        A = a_list[0]
        
    return Q, A

        
# ============================================划分线==================================================

# 表格单元格查询QA对模板
element_table_cell_templates = [
    "Recognize all contents within the regions {bbox} in <determiner> <term> in turn based on <T_IMG>.",
    "Identify all contents inside the regions {bbox} in <determiner> <term> sequentially based on <T_IMG>.",
    "Detect all items within the areas {bbox} in <determiner> <term> one by one based on <T_IMG>.",
    "Examine each content within the areas {bbox} in <determiner> <term> consecutively based on <T_IMG>.",
    "Extract all contents in the boundaries {bbox} in <determiner> <term> one by one based on <T_IMG>.",
    "Scan all items in the areas {bbox} in <determiner> <term> sequentially based on <T_IMG>.",
    "Survey all contents in the boundaries {bbox} in <determiner> <term> one by one using <T_IMG>."
]

def generate_table_cell_recognition_qa(text_input, bbox_input, val_mode, task_key, max_cells=30):
    texts = text_input
    bboxes = bbox_input

    if not texts and not bboxes:
        assert len(bboxes) == len(texts)
    
    candidate_index = [i for i in range(len(bboxes))]
    random.shuffle(candidate_index)
    selected_idx = candidate_index[:random.randint(1, min(len(bboxes), max_cells))]
    selected_idx = sorted(selected_idx)

    selected_texts = []
    selected_bboxes = []

    for idx in selected_idx:
        selected_texts.append(texts[idx])
        selected_bboxes.append(bboxes[idx])


    if not val_mode:
        # generate Q 
        chosen_template = random.choice(element_table_cell_templates)
        chosen_determiner = random.choice(determiners)
        chosen_term = random.choice(terms)
        chosen_template = chosen_template.replace('<determiner>', chosen_determiner)
        chosen_template = chosen_template.replace('<term>', chosen_term)
        Q = chosen_template

        if random.choice([True, False]):
            Q = "<image>\n" + Q
        else:
            Q = Q + "\n<image>"
    else:
        # generate Q 
        Q = "Recognize all contents within the regions {bbox} in this picture in turn based on <T_IMG>."
        chosen_term = 'picture'
        Q = Q + "\n<image>"

    # generate A ; delete the bbox of the whole table
    Q = Q.replace("{bbox}", '\n'.join(selected_bboxes))  # 在问题模板中替换bbox坐标
    A = f"The corresponding contents in all regions are listed as following:\n" + '\n'.join(selected_texts)

    return Q, A    
