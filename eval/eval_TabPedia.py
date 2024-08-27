import os
import os.path as osp
import argparse
import json
from typing import Dict, Sequence, List

import PIL
from PIL import Image, ImageOps
import torch
from tqdm import tqdm
from transformers import BitsAndBytesConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from xtuner.dataset.utils import expand2square, load_image
from xtuner.registry import BUILDER
from tools.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from tools.model.utils import guess_load_checkpoint
from tools.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX
from tools.dataset.utils import expand2square, generate_qa, generate_element_qa, generate_zeroelement_qa, generate_table_structure_qa, generate_table_query_qa, generate_table_vqa_qa, generate_table_cell_recognition_qa
from tools.utils.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_PAUSE_TOKEN
from tools.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX
from tools.model.tabpedia import TabPedia

from mmengine.config import Config, DictAction, ConfigDict
from torch.utils.data import Dataset, DataLoader
import jsonlines
from PIL import Image
import math
import re
from torch.nn.utils.rnn import pad_sequence

from datasets import DatasetDict, load_from_disk
from functools import partial
from typing import Optional, Tuple, Union

import torch.nn.functional as F
from typing import Callable, Dict, Mapping, Tuple
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD 
from torchvision import transforms as T
from torchvision.transforms.functional import resize
import numpy as np
import copy
import random
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data._utils.collate import default_collate

from transformers import PreTrainedModel, PretrainedConfig


class TabPediaModel(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, 
                llm,
                high_res_visual_encoder,
                high_res_visual_encoder_type='swin',
                low_res_visual_encoder_type='clip',
                low_res_visual_encoder=None,
                freeze_llm=False,
                freeze_high_res_visual_encoder=False,
                freeze_low_res_visual_encoder=True,
                visual_select_layer=-2,
                pretrained_pth=None,
                global_pretrained_pth=None,
                llm_lora=None,
                high_res_visual_encoder_lora=None,
                low_res_visual_encoder_lora=None,
                use_activation_checkpointing=True,
                model_dtype=torch.bfloat16,
                tokenizer=None,
                special_tokens=None,
                load_pretrained_mode='all',
                phase='pretrain',
                **kwargs
                ):
        super(TabPediaModel, self).__init__(PretrainedConfig())
        self.model = TabPedia(
            llm=llm,
            high_res_visual_encoder=high_res_visual_encoder,
            high_res_visual_encoder_type=high_res_visual_encoder_type,
            low_res_visual_encoder_type=low_res_visual_encoder_type,
            low_res_visual_encoder=low_res_visual_encoder,
            freeze_llm=freeze_llm,
            freeze_high_res_visual_encoder=freeze_high_res_visual_encoder,
            freeze_low_res_visual_encoder=freeze_low_res_visual_encoder,
            visual_select_layer=visual_select_layer,
            pretrained_pth=pretrained_pth,
            global_pretrained_pth=global_pretrained_pth,
            llm_lora=llm_lora,
            high_res_visual_encoder_lora=high_res_visual_encoder_lora,
            low_res_visual_encoder_lora=low_res_visual_encoder_lora,
            use_activation_checkpointing=use_activation_checkpointing,
            model_dtype=model_dtype,
            tokenizer=tokenizer,
            special_tokens=special_tokens,
            load_pretrained_mode=load_pretrained_mode,
            phase=phase
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        low_res_image: Optional[torch.FloatTensor] = None,
        high_res_image: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
        ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            low_res_image=low_res_image,
            high_res_image=high_res_image,
            return_dict=return_dict,
            kwargs=kwargs
        )
        return output

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "low_res_image": kwargs.get("low_res_image", None),
                "high_res_image": kwargs.get("high_res_image", None)
            }
        )
        return model_inputs

class TabPedia_dataset(Dataset):

    def __init__(self,
                args,
                json_file_path,
                image_folder,
                tokenizer,
                image_processor,
                high_res_size,
                global_image_processor=None,
                split='train',
                phase='prertain',
                prompt_template=None):
        super().__init__()

        self.tokenizer = tokenizer

        self.prompt_template = prompt_template

        if not os.path.isfile(json_file_path):
            json_data = []
            json_files = os.listdir(json_file_path)
            json_files = [os.path.join(json_file_path, i) for i in json_files if i.endswith('.json')]
            for json_file in json_files:
                json_data.extend(json.load(open(json_file)))
            self.json_data = json_data
        else:
            self.json_data = json.load(open(json_file_path))
        
        self.json_data = self.json_data[:min(len(self.json_data), args.eval_samples)]
        split_interval = math.ceil(len(self.json_data) / args.all_processes)
        self.json_data = self.json_data[split_interval*args.process_id:split_interval*(args.process_id+1)]

        self.image_folder = image_folder
        if isinstance(image_processor, dict) or isinstance(
                image_processor, Config) or isinstance(image_processor,
                                                       ConfigDict):
            self.image_processor = BUILDER.build(image_processor)
        else:
            self.image_processor = image_processor

        if isinstance(global_image_processor, dict) or isinstance(
                global_image_processor, Config) or isinstance(global_image_processor,
                                                    ConfigDict):
            self.global_image_processor = BUILDER.build(global_image_processor)
        else:
            self.global_image_processor = global_image_processor

        self.split = split
        self.phase = phase
        self.high_res_size = high_res_size
        Learnable_token = [f'<extra_token_{i}>' for i in range(256)]
        self.LEARN_TOKENS = "".join(Learnable_token)

        self.image_transform =  T.Compose([
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2))]) if split=='train' else None
        self.to_tensor = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )


    def __len__(self):
        return len(self.json_data)

    def prepare_input(self, img: PIL.Image.Image, random_padding: bool = False, first_resize: bool = True) -> torch.Tensor:
        """
        Convert PIL Image to tensor according to specified input_size after following steps below:
            - resize if first_resize else None
            - rotate (if align_long_axis is True and image is not aligned longer axis with canvas)
            - pad
        """
        img = img.convert("RGB")
        if first_resize:
            img = resize(img, min(self.high_res_size))
            img.thumbnail((self.high_res_size[1], self.high_res_size[0]))
        delta_width = self.high_res_size[1] - img.width
        delta_height = self.high_res_size[0] - img.height
        if random_padding:
            pad_width = np.random.randint(low=0, high=delta_width + 1)
            pad_height = np.random.randint(low=0, high=delta_height + 1)
        else:
            pad_width = delta_width // 2
            pad_height = delta_height // 2
        padding = (
            pad_width,
            pad_height,
            delta_width - pad_width,
            delta_height - pad_height,
        )
        return self.to_tensor(ImageOps.expand(img, padding))

    def process_image(self, image):
        first_resize = True
        if image.size[0] > self.high_res_size[1] or image.size[1] > self.high_res_size[0]:
            image = resize(image, min(self.high_res_size))
            image.thumbnail((self.high_res_size[1], self.high_res_size[0]))
            first_resize = False

        if self.image_transform:
            image = self.image_transform(image)

        high_res_image = self.prepare_input(copy.deepcopy(image), first_resize=first_resize)
        low_res_image = expand2square(
                    copy.deepcopy(image),
                    tuple(
                        int(x * 255) for x in self.global_image_processor.image_mean))
        low_res_image = self.global_image_processor.preprocess(
            low_res_image, return_tensors='pt')['pixel_values'][0]
        return high_res_image, low_res_image

    def insert_random_pauses(self, s, num_pauses=3):
        """
        Inserts '<pause>' between words in a given string.

        :param s: The original string.
        :param num_pauses: Number of pauses to insert.
        :return: Modified string with pauses.
        """
        if num_pauses < 0:
            raise ValueError("Number of pauses must be non-negative")

        # Find positions of spaces
        space_positions = [i for i, char in enumerate(s[:-1]) if char == ' ']
        
        # If there are not enough spaces for the requested number of pauses,
        # we use the maximum number possible
        num_pauses = min(num_pauses, len(space_positions))

        # Randomly select positions from the list of spaces
        selected_positions = random.sample(space_positions, num_pauses)
        
        # Insert pauses at the selected positions
        for pos in sorted(selected_positions, reverse=True):
            s = s[:pos] + ' <Pause_Token>' + s[pos:]
        
        return s

    def process_qa(self, q_str, a_str):
        # Prepared placeholder for image tokens, i.e., 256 for low-res tokens, and 1200 for high-res tokens
        replace_token = "".join([DEFAULT_IM_START_TOKEN] + ["<C_IMG>"] + [DEFAULT_IMAGE_PATCH_TOKEN] * 256 + ["<T_IMG>"] + [DEFAULT_IMAGE_PATCH_TOKEN] * 1200 + [DEFAULT_IM_END_TOKEN])

        q_str = q_str.replace(DEFAULT_IMAGE_TOKEN, replace_token)
        a_str = a_str + self.tokenizer.eos_token

        q_str = q_str + self.LEARN_TOKENS
        qa_str = q_str + a_str
        assert replace_token in qa_str

        q_input_ids = self.tokenizer(
            q_str,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        ).input_ids

        qa_input_ids = self.tokenizer(
            qa_str,
            return_tensors="pt",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        ).input_ids

        targets = qa_input_ids.clone()
        q_len = q_input_ids.size(1)
        targets[:, :q_len] = -100
        targets[targets == DEFAULT_PAUSE_TOKEN] = -100
        return q_input_ids, qa_input_ids, targets


    def adjust_bbox_single(self, w, h, bbox):
        # If the width is greater than the height, fill in the vertical direction. Otherwise, fill horizontally

        # Calculate the bbox coordinates on the filled 224x224 image
        x1_ori, y1_ori, x2_ori, y2_ori = bbox

        if w >= h * 0.75:
            padding = w * 2. / 3. - h / 2.
            y1_new = y1_ori + padding
            y2_new = y2_ori + padding
            x1, y1, x2, y2 = x1_ori, y1_new, x2_ori, y2_new
            x1, y1, x2, y2 = x1 / w, y1 / w * 0.75, x2 / w, y2 / w * 0.75
        else:
            padding = h * 3. / 8. - w / 2. 
            x1_new = x1_ori + padding
            x2_new = x2_ori + padding
            x1, y1, x2, y2 = x1_new, y1_ori, x2_new, y2_ori
            x1, y1, x2, y2 = x1 / h * 4. / 3., y1 / h, x2 / h * 4. / 3., y2 / h

        x1, y1, x2, y2 = round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)
        return [x1, y1, x2, y2]


    def extract_and_adjust_bbox(self, w, h, s):
        # check for bbox coordinates
        if not re.search(r'\[(\d+(?:\.\d+)?),(\d+(?:\.\d+)?),(\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\]', s):
            return s

        def replace_bbox(match):
            # get bbox coordinates
            bbox = list(map(float, match.groups()))
            # normalizr bbox coordinates
            adjusted_bbox = self.adjust_bbox_single(w, h, bbox)                
            # return alternative str
            return "[{:.3f},{:.3f},{:.3f},{:.3f}]".format(*adjusted_bbox)
        
        # replacement operations using regular expressions
        adjusted_str = re.sub(r'\[(\d+(?:\.\d+)?),(\d+(?:\.\d+)?),(\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\]', replace_bbox, s)
        
        return adjusted_str

    def generate_qa_sft(self, data_dict):
        if data_dict['task'] in ['LayoutDet-SE']:
            q_str = data_dict['conversations'][0]['value']
            a_str = data_dict['conversations'][1]['value']
            text_input = data_dict.get("text", None)
            bbox_input = data_dict.get("bbox", None)
            task_type = data_dict.get("task", None)
            if 'None' not in text_input:
                q_str, a_str, task_type = generate_element_qa(text_input, bbox_input, task_type, val_mode=self.split=='test', task_key='det-table', block_text_input=None)
            else:
                q_str, a_str, task_type = generate_zeroelement_qa(task_input="LayoutDet\nSpot", val_mode=self.split=='test', task_key=None)
        elif data_dict['task'] in ['Table_Structure']:
            text_input = data_dict.get("text", None)
            bbox_input = data_dict.get("bbox", None)
            task_type = data_dict.get("task", None)
            q_str, a_str = generate_table_structure_qa(text_input, bbox_input, val_mode=self.split=='test', task_key=None)
        elif data_dict['task'] in ['Table_Query']:
            text_input = data_dict.get("text", None)
            bbox_input = data_dict.get("bbox", None)
            selected_id = data_dict.get("selected_id", None)[0]
            selected_text_input = text_input[selected_id]
            selected_bbox_input = bbox_input[selected_id]
            q_str, a_str = generate_table_query_qa(selected_text_input, selected_bbox_input, val_mode=self.split=='test', task_key=None)
        elif data_dict['task'] in ['Table_VQA']:
            question_input = data_dict.get('questions', None)
            answer_input = data_dict.get("answers", None)
            if question_input is None or answer_input is None:
                print("error")
            q_str, a_str = generate_table_vqa_qa(question_input, answer_input, val_mode=self.split=='test', task_key=None)
        else:
            raise ValueError("Wrong task type: ", data_dict['task'])

        q_str = self.prompt_template.SYSTEM + self.prompt_template.INSTRUCTION.format(input=q_str)

        # normalize coordinations to [0, 1] in q_str and a_str
        q_str = self.extract_and_adjust_bbox(data_dict['width'], data_dict['height'], q_str)
        a_str = self.extract_and_adjust_bbox(data_dict['width'], data_dict['height'], a_str)
        return q_str, a_str


    def __getitem__(self, index):
        data_dict = copy.deepcopy(self.json_data[index])
        if self.phase == 'finetune':
            q_str, a_str = self.generate_qa_sft(data_dict)
        else:
            raise ValueError("Unrecognized phase type", self.phase)
        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']
            image_dir = data_dict['img_dir']
            image = Image.open(os.path.join(self.image_folder,
                                            image_dir,
                                            image_file)).convert('RGB')
            raw_image = image.copy()
            high_res_image, low_res_image = self.process_image(raw_image)
            data_dict['high_res_image'] = high_res_image
            data_dict['low_res_image'] = low_res_image

        q_input_ids, qa_input_ids, targets = self.process_qa(q_str=q_str, a_str=a_str)
        data_dict['input_ids'] = q_input_ids[0]
        data_dict['target'] = a_str
        data_dict['pad_token_id'] = self.tokenizer.pad_token_id

        return data_dict


def tabpedia_collate_fn(instances: Sequence[Dict],
                       pad_index: int = DEFAULT_PAD_TOKEN_INDEX,
                       return_hf_format: bool = False,
                       use_varlen_attn: bool = False,
                       model_dtype=torch.bfloat16):

    out_batch = {}
    input_ids = []
    input_len = []
    for example in instances:
        input_ids.append(torch.LongTensor(example['input_ids']))
        input_len.append(len(example['input_ids']))
    new_input_ids = torch.ones((len(input_len), max(input_len)), dtype=torch.int64).fill_(instances[0]['pad_token_id'])
    for i, token_tensor in enumerate(input_ids):
        new_input_ids[i, -token_tensor.shape[0]:] = token_tensor
    out_batch['input_ids'] = new_input_ids

    out_batch['attention_mask'] = out_batch['input_ids'].ne(
        instances[0]['pad_token_id']).long()
    out_batch['low_res_image'] = default_collate([data['low_res_image'] for data in instances]).to(dtype=model_dtype)
    out_batch['high_res_image'] = default_collate([data['high_res_image'] for data in instances]).to(dtype=model_dtype)
    out_batch['img_name'] = [instance['image'] for instance in instances]
    out_batch['target'] = [instance['target'] for instance in instances]
    if return_hf_format:
        return out_batch
    else:
        return out_batch


class Eval():

    def __init__(
        self,
        args,
        model_cfg,
        adapter=None,
        bits=None,
    ) -> None:
        super().__init__()

        self.model, self.tokenizer = self.build_components(
            args.model_name_or_path, model_cfg, adapter, bits)

        from transformers import GenerationConfig as HFGenerationConfig
        self._generation_config = HFGenerationConfig(
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None else
            self.tokenizer.eos_token_id,
        )

    def build_components(self, model_name_or_path, cfg, adapter=None, bits=None):

        tokenizer = BUILDER.build(cfg.tokenizer)
        tokenizer.add_tokens(cfg.special_tokens, special_tokens=False)

        model = TabPediaModel(**cfg.model)
        
        # load pretrained weights
        state_dict = torch.load(model_name_or_path)
        missing_keys, unexcepted_keys = model.model.load_state_dict(state_dict, strict=False)
        print(f'Load PTH model from {model_name_or_path}')
        print(f"missing keys: ", missing_keys)
        print(f"unexcepted keys: ", unexcepted_keys)

        model.eval()
        return model, tokenizer

    def predict(self,
                args,
                dataloader: DataLoader = None):
        
        from transformers import GenerationConfig as HFGenerationConfig
        hf_gen_config = HFGenerationConfig(
            max_new_tokens=3000,
            do_sample=False,
            temperature=0.7,
            repetition_penalty=args.repetition_penalty,
            seed=42,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            num_beams=1,
            use_cache=True)

        results_record = []
        for k, data in enumerate(tqdm(dataloader)):
            input_data = {}
            input_data['input_ids'] = data['input_ids'].to(device=args.device)
            input_data['attention_mask'] = data['attention_mask'].to(device=args.device)
            input_data['low_res_image'] = data['low_res_image'].to(dtype=torch.bfloat16, device=args.device)
            input_data['high_res_image'] = data['high_res_image'].to(dtype=torch.bfloat16, device=args.device)
            with torch.inference_mode():
                outputs = self.model.generate(**input_data, generation_config=hf_gen_config)
                input_token_len = input_data['input_ids'].shape[1]
                predictions = self.tokenizer.batch_decode(outputs[:, input_token_len:], skip_special_tokens=True)

            for i in range(len(predictions)):
                sample ={
                    "img_name": data['img_name'][i],
                    "pred": predictions[i],
                    "gt": data['target'][i]
                }
                results_record.append(sample)


        with open(args.save_path, 'w') as f:
            json.dump(results_record, f, ensure_ascii=False, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model_name_or_path", type=str, help='the huggingface format path of pretrained model')
    parser.add_argument("--jsonl_file_path", type=str, help='the jsonl file path for evaluationn')
    parser.add_argument("--img_root_path", type=str, help='the root path of image dir')
    parser.add_argument("--eval",  action="store_true", default=True, help='the mode eval|test')
    parser.add_argument("--config", type=str, help='the config model name or path' )
    parser.add_argument("--batch_size", type=int, default=2, help='the batch size for inference')
    parser.add_argument("--device", type=str, default='cuda', help='the device for inference')
    parser.add_argument("--save_path", type=str, default='./work_dirs/donut_table_cell_query_baseline/cell_pred_trunk_num_1.json' ,help='save json file path')
    parser.add_argument("--eval_samples", type=int, default=100)
    parser.add_argument("--process_id", type=int, default=None)
    parser.add_argument("--all_processes", type=int, default=8)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    args = parser.parse_args()
    return args


def main(args):
    assert args.model_name_or_path is not None

    # load config
    cfg = Config.fromfile(args.config)

    # Initialize model and tokenizer for generation
    evaluator = Eval(args=args, model_cfg=cfg)


    dataset = TabPedia_dataset(
        args=args,
        json_file_path=args.jsonl_file_path,
        image_folder=cfg.image_folder,
        tokenizer=evaluator.tokenizer,
        image_processor=cfg.image_processor,
        high_res_size=[cfg.high_res_image_height, cfg.high_res_image_width],
        global_image_processor=cfg.global_image_processor,
        split='test',
        phase='finetune',
        prompt_template=cfg.prompt_template,
    )
    print("\033[31m >>> Initializing evaluator for TabPedia inference \033[0m")

    dataloader = DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=2,
        collate_fn=partial(tabpedia_collate_fn)
    )

    if args.device == 'cuda':
        evaluator.model.cuda()
    
    evaluator.predict(args=args, dataloader=dataloader)


if __name__=='__main__':
    args = parse_args()
    main(args)