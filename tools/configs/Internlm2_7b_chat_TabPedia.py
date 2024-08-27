# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          CLIPImageProcessor, CLIPVisionModel)
from transformers import DonutImageProcessor

from xtuner.dataset import LLaVADataset, tabpedia_v1_dataset
from xtuner.dataset.collate_fns import default_collate_fn, tabpedia_collate_fn
from xtuner.dataset.map_fns import llava_map_ocr_fn, template_map_fn_factory
from xtuner.dataset.map_fns import llava_map_ocr_fn_pad2square, llava_map_ocr_fn_donut_align, llava_map_ocr_fn_donut_unalign, llava_map_ocr_fn_pix2struct, tabpedia_map_ocr_base_fn_add_sys_prompt
from xtuner.dataset.samplers import LengthGroupedSampler, TaskSampler
from mmengine.dataset import DefaultSampler
from xtuner.engine.hooks import DatasetInfoHook, EvaluateChatHook
from xtuner.engine.runner import TrainLoop
from tools.model import TabPedia, DonutModel
from mmengine.visualization import Visualizer, WandbVisBackend
from tools.utils import PROMPT_TEMPLATE

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
CLIP_L_224px_pretrained_pth = 'Your_Download_CLIP-ViT-L-224'
llm_name_or_path = 'Your_Download_InternLM-7B-chat'
donut_pretrained_pth = 'pretrained_pth/donut-base'


# Tokenizer
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
DEFAULT_PAUSE_TOKEN = "<Pause_Token>"
special_tokens = [DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]+ [f'<extra_token_{i}>' for i in range(256)] \
                + [DEFAULT_PAUSE_TOKEN, "<T_IMG>", "<C_IMG>"]


prompt_template = PROMPT_TEMPLATE.internlm2_chat
max_length = 4096
high_res_image_height = 2560
high_res_image_width = 1920
phase='finetune'
image_folder=""



#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
    padding_side='right',
    model_max_length=max_length)

image_processor = dict(
    type=DonutImageProcessor.from_pretrained,
    pretrained_model_name_or_path='ImageProcessor_file/High_Resolution_Imageprocessor/preprocessor_config.json',
    trust_remote_code=True)

global_image_processor = dict(
    type=CLIPImageProcessor.from_pretrained,
    pretrained_model_name_or_path=CLIP_L_224px_pretrained_pth,
    trust_remote_code=True)

model = dict(
    type=TabPedia,
    freeze_llm=False,
    freeze_high_res_visual_encoder=False,
    freeze_low_res_visual_encoder=True,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ),
    high_res_visual_encoder=dict(
        type=DonutModel.from_pretrained,
        pretrained_model_name_or_path=donut_pretrained_pth,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ),
    low_res_visual_encoder=dict(
        type=CLIPVisionModel.from_pretrained,
        pretrained_model_name_or_path=CLIP_L_224px_pretrained_pth,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ),
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    load_pretrained_mode='all',
    phase=phase
    )



