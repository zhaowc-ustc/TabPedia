# Copyright (c) OpenMMLab. All rights reserved.
from transformers import AutoConfig, AutoModel

from .configuration_projector import ProjectorConfig, HighResProjectorConfig, LowResProjectorConfig
from .modeling_projector import ProjectorModel, HighResProjectorModel, LowResProjectorModel

AutoConfig.register('projector', ProjectorConfig)
AutoModel.register(ProjectorConfig, ProjectorModel)
AutoConfig.register('highresprojector', HighResProjectorConfig)
AutoModel.register(HighResProjectorConfig, HighResProjectorModel)
AutoConfig.register('lowresprojector', LowResProjectorConfig)
AutoModel.register(LowResProjectorConfig, LowResProjectorModel)

__all__ = ['ProjectorConfig', 'ProjectorModel', 'HighResProjectorConfig', 'HighResProjectorModel', 'LowResProjectorConfig', 'LowResProjectorModel']
