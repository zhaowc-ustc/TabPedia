# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
import logging

import torch.nn as nn
import torch
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from peft import get_peft_model, prepare_model_for_kbit_training
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss

from xtuner.registry import BUILDER
from .modules import dispatch_modules, LowResProjectorConfig, LowResProjectorModel,HighResProjectorConfig, HighResProjectorModel
from .utils import (LoadWoInit, find_all_linear_names,
                    get_peft_model_state_dict, guess_load_checkpoint,
                    make_inputs_require_grad,
                    prepare_inputs_labels_for_multimodal, traverse_dict)


from ..utils.constants import *


class TabPedia(BaseModel):

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
                 phase='pretrain'
                 ):
        super().__init__()
        self.freeze_llm = freeze_llm
        self.freeze_high_res_visual_encoder = freeze_high_res_visual_encoder
        self.freeze_low_res_visual_encoder = freeze_low_res_visual_encoder
        self.low_res_visual_encoder_type = low_res_visual_encoder_type
        self.high_res_visual_encoder_type = high_res_visual_encoder_type
        self.phase = phase

        with LoadWoInit():
            self.llm = self._build_from_cfg_or_module(llm)
            self.high_res_visual_encoder = self._build_from_cfg_or_module(high_res_visual_encoder)
            self.low_res_visual_encoder = self._build_from_cfg_or_module(low_res_visual_encoder)

        self.llm.config.use_cache = False
        dispatch_modules(self.llm)

        if self.high_res_visual_encoder_type == 'swin':
            high_res_projector_config = HighResProjectorConfig(
                visual_hidden_size= 1024,
                llm_hidden_size=self.llm.config.hidden_size
            )
            self.high_res_projector = HighResProjectorModel(
                high_res_projector_config
            ).to(model_dtype)
        else:
            raise ValueError("Unrecognized high_res_visual_encoder_type: ", high_res_visual_encoder_type)
        
        if self.low_res_visual_encoder_type == 'clip':
            low_res_projector_config = LowResProjectorConfig(
                visual_hidden_size=self.low_res_visual_encoder.config.hidden_size,
                llm_hidden_size=self.llm.config.hidden_size
            )
            self.low_res_projector = LowResProjectorModel(
                low_res_projector_config
            ).to(model_dtype)
        else:
            raise ValueError("Unrecognized high_res_visual_encoder_type: ", high_res_visual_encoder_type)

        if tokenizer is not None:
            self.tokenizer = BUILDER.build(tokenizer)
            if special_tokens is not None:
                num_new_tokens = self.tokenizer.add_tokens(special_tokens, special_tokens=True)
                self.special_tokens = special_tokens
            self.llm.resize_token_embeddings(len(self.tokenizer))
            if num_new_tokens > 0:
                input_embeddings = self.llm.get_input_embeddings().weight.data
                output_embeddings = self.llm.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg
            logging.info("Refresh the word embedding with vocab size: ", len(self.tokenizer))

        if self.freeze_llm:
            self.llm.requires_grad_(False)
        if self.freeze_high_res_visual_encoder:
            self.high_res_visual_encoder.requires_grad_(False)
        if self.freeze_low_res_visual_encoder:
            self.low_res_visual_encoder.requires_grad_(False)

        if use_activation_checkpointing:
            # For backward compatibility
            if hasattr(self.llm, 'enable_input_require_grads'):
                self.llm.enable_input_require_grads()
            else:
                self.llm.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad)

            if hasattr(self.low_res_visual_encoder, 'enable_input_require_grads'):
                self.low_res_visual_encoder.enable_input_require_grads()
            else:
                self.low_res_visual_encoder.get_input_embeddings(
                ).register_forward_hook(make_inputs_require_grad)
            self.low_res_projector.enable_input_require_grads()

            if hasattr(self.high_res_visual_encoder, 'enable_input_require_grads'):
                self.high_res_visual_encoder.enable_input_require_grads()
            else:
                self.high_res_visual_encoder.get_input_embeddings(
                ).register_forward_hook(make_inputs_require_grad)
            self.high_res_projector.enable_input_require_grads()

            # enable gradient (activation) checkpointing for memory efficiency
            self.gradient_checkpointing_enable()

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            if type(pretrained_state_dict) is dict:
                pretrained_state_dict = pretrained_state_dict['module']
            self.load_pretrained_weights(pretrained_state_dict, load_pretrained_mode)
            print(f'Load pretrained weight from {pretrained_pth}')
        
        if self.phase == 'pretrain':
            # activate requires_grad of input_embeddings for special token learning
            self.only_activate_word_embedding()

        self._is_init = True

    
    def only_activate_word_embedding(self,):
        if 'internlm2' in self.llm.config.model_type.lower():
            self.llm.model.tok_embeddings.requires_grad_(True)
        elif 'qwen' in self.llm.config.model_type.lower():
            self.llm.model.embed_tokens.requires_grad_(True)
        else:
            raise ValueError("Unknown llm model type: ", self.llm.config.model_type.lower())


    def load_pretrained_weights(self, pretrained_state_dict, load_pretrained_mode):
        if load_pretrained_mode == 'all':
            missing_keys, _ = self.load_state_dict(pretrained_state_dict, strict=False)
            print("missing keys: ", missing_keys)
        elif load_pretrained_mode == 'finetune':
            # load clip projector weight
            projector_weight = OrderedDict()
            for k in pretrained_state_dict:
                if "low_res_projector" in k:
                    projector_weight[k.replace('low_res_projector.', '')] = pretrained_state_dict[k]
            self.low_res_projector.load_state_dict(projector_weight)

            # load donut projector weight
            projector_weight = OrderedDict()
            for k in pretrained_state_dict:
                if "high_res_projector" in k:
                    projector_weight[k.replace('high_res_projector.', '')] = pretrained_state_dict[k]
            self.high_res_projector.load_state_dict(projector_weight)

            # load vocab embedding weight
            input_embeddings = self.llm.get_input_embeddings().weight.data
            num_new_tokens = len(self.special_tokens)
            if 'qwen' in self.llm.config.model_type.lower():
                input_embeddings[-num_new_tokens:] = pretrained_state_dict['llm.model.embed_tokens.weight'][-num_new_tokens:]
            elif 'internlm2' in self.llm.config.model_type.lower():
                input_embeddings[-num_new_tokens:] = pretrained_state_dict['llm.model.tok_embeddings.weight'][-num_new_tokens:]
            elif "vicuna" in self.llm.config.model_type.lower():
                input_embeddings[-num_new_tokens:] = pretrained_state_dict['llm.embed_tokens.weight'][-num_new_tokens:]

            # load high res visual encoder
            projector_weight = OrderedDict()
            for k in pretrained_state_dict:
                if "high_res_visual_encoder" in k:
                    projector_weight[k.replace('high_res_visual_encoder.', '')] = pretrained_state_dict[k]
            self.high_res_visual_encoder.load_state_dict(projector_weight)


    def _parse_lora_config(self, lora_config):
        if isinstance(lora_config, dict) or isinstance(
                lora_config, Config) or isinstance(lora_config, ConfigDict):
            lora_config = BUILDER.build(lora_config)
        return lora_config

    def _prepare_llm_for_lora(self,
                              lora_config,
                              use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        self.llm = prepare_model_for_kbit_training(
            self.llm, use_activation_checkpointing)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.llm)
            lora_config.target_modules = modules
        self.llm = get_peft_model(self.llm, lora_config)

    def _prepare_visual_encoder_for_lora(self,
                                         lora_config,
                                         use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.visual_encoder)
            lora_config.target_modules = modules
        self.visual_encoder = get_peft_model(self.visual_encoder, lora_config)

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        self.llm.gradient_checkpointing_enable()
        self.high_res_visual_encoder.gradient_checkpointing_enable()
        self.high_res_projector.gradient_checkpointing_enable()
        self.low_res_projector.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()
        self.visual_encoder.gradient_checkpointing_disable()
        self.projector.gradient_checkpointing_disable()

    def init_weights(self):
        pass

    def _build_from_cfg_or_module(self, cfg_or_mod):
        if isinstance(cfg_or_mod, nn.Module):
            return cfg_or_mod
        elif isinstance(cfg_or_mod, dict):
            traverse_dict(cfg_or_mod)
            return BUILDER.build(cfg_or_mod)
        else:
            raise NotImplementedError

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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Project Word Tokens
        if inputs_embeds is None:
            if 'internlm2' in self.llm.config.model_type.lower():
                inputs_embeds = self.llm.model.tok_embeddings(input_ids)
            elif 'qwen' in self.llm.config.model_type.lower():
                inputs_embeds = self.llm.model.embed_tokens(input_ids)
            else:
                inputs_embeds = self.llm.embed_tokens(input_ids)

        if low_res_image is not None and high_res_image is not None and past_key_values is None:
            # Extract Visual Embeddings
            low_res_outputs = self.low_res_visual_encoder(low_res_image, output_hidden_states=True)
            select_hidden_state_layer = getattr(self.low_res_visual_encoder.config, "mm_vision_select_layer", -1)
            select_hidden_state = low_res_outputs.hidden_states[select_hidden_state_layer] # torch.Size([1, 257, 1024])
            low_res_visual_embeds = select_hidden_state[:, 1:]
            high_res_visual_embeds = self.high_res_visual_encoder.inference(image_tensors=high_res_image)


            low_res_visual_tokens = self.low_res_projector(low_res_visual_embeds)
            batch_size = low_res_visual_tokens.size(0)
            high_res_visual_tokens = self.high_res_projector(high_res_visual_embeds.permute(0,2,1).view(batch_size, 1024, 80, 60))
            high_res_visual_tokens = high_res_visual_tokens.view(batch_size, self.llm.config.hidden_size, -1).permute(0, 2, 1)

            new_input_embeds = []
            cur_image_idx = 0
            batch_size = input_ids.size(0)
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                if cur_image_idx >= batch_size:
                    continue
                if (cur_input_ids == self.tokenizer.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN)).sum() != (cur_input_ids == self.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)).sum():
                    # raise ValueError("The number of image start tokens and image end tokens should be the same.")
                    new_input_embeds.append(cur_input_embeds.detach())
                    cur_image_idx += 1
                    continue
                image_start_tokens = torch.where(cur_input_ids == self.tokenizer.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN))[0]
                for image_start_token_pos in image_start_tokens:
                    cur_low_res_image_feature = low_res_visual_tokens[cur_image_idx].to(device=cur_input_embeds.device)
                    cur_high_res_image_feature = high_res_visual_tokens[cur_image_idx].to(device=cur_input_embeds.device)
                    num_low_res_patches = cur_low_res_image_feature.shape[0]
                    num_high_res_patches = cur_high_res_image_feature.shape[0]
                    if self.phase == 'pretrain':
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:image_start_token_pos].detach(), cur_input_embeds[image_start_token_pos:image_start_token_pos+2], cur_low_res_image_feature, \
                                                        cur_input_embeds[image_start_token_pos+2+256:image_start_token_pos+2+257], cur_high_res_image_feature, cur_input_embeds[image_start_token_pos + num_low_res_patches + num_high_res_patches + 3:image_start_token_pos + num_low_res_patches + num_high_res_patches + 4], \
                                                        cur_input_embeds[image_start_token_pos + num_low_res_patches + num_high_res_patches + 4:].detach()), dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:image_start_token_pos+2], cur_low_res_image_feature, cur_input_embeds[image_start_token_pos+2+256:image_start_token_pos+2+257], cur_high_res_image_feature, cur_input_embeds[image_start_token_pos + num_low_res_patches + num_high_res_patches + 3:]), dim=0)
                cur_image_idx += 1
                new_input_embeds.append(cur_new_input_embeds)
            inputs_embeds = torch.stack(new_input_embeds, dim=0)


        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.llm(
            input_ids=None if inputs_embeds is not None else input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if 'logits' in outputs:
            logits = outputs.logits # Qwen-7B | InternLM2
        else:
            logits = self.llm.lm_head(outputs[0]).float() # Vicuna-7B
        

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)



        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _forward(self, data, data_samples=None):

        outputs = self.forward(**data)

        return outputs

    def predict(self, data, data_samples=None):
        outputs = self.llm(**data)
        logits_dict = [{'logits': logits} for logits in outputs.logits]
        return logits_dict

    def compute_loss(self, data, data_samples=None):
        outputs = self.llm(**data)
        loss_dict = {'loss': outputs.loss}
        return loss_dict

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.llm, name)
