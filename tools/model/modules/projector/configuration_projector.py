# Copyright (c) OpenMMLab. All rights reserved.
from transformers import PretrainedConfig


class ProjectorConfig(PretrainedConfig):
    model_type = 'projector'
    _auto_class = 'AutoConfig'

    def __init__(
        self,
        visual_hidden_size=4096,
        llm_hidden_size=4096,
        depth=2,
        hidden_act='gelu',
        bias=True,
        **kwargs,
    ):
        self.visual_hidden_size = visual_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.depth = depth
        self.hidden_act = hidden_act
        self.bias = bias
        super().__init__(**kwargs)


class LowResProjectorConfig(PretrainedConfig):
    model_type = 'lowresprojector'
    _auto_class = 'AutoConfig'

    def __init__(
        self,
        visual_hidden_size=4096,
        llm_hidden_size=4096,
        bias=True,
        **kwargs,
    ):
        self.visual_hidden_size = visual_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.bias = bias
        super().__init__(**kwargs)


class HighResProjectorConfig(PretrainedConfig):
    model_type = 'highresprojector'
    _auto_class = 'AutoConfig'

    def __init__(
        self,
        visual_hidden_size=4096,
        llm_hidden_size=4096,
        kerner_size=3,
        stride=2,
        padding=1,
        **kwargs,
    ):
        self.visual_hidden_size = visual_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.kerner_size = kerner_size
        self.stride = stride
        self.padding = padding
        super().__init__(**kwargs)
