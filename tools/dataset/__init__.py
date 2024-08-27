# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from .utils import generate_qa, generate_element_qa, generate_table_cell_recognition_qa, generate_table_query_qa, generate_table_structure_qa, generate_table_vqa_qa, generate_zeroelement_qa

# ignore FutureWarning in hf datasets
warnings.simplefilter(action='ignore', category=FutureWarning)

__all__ = [
    'generate_qa',
    'generate_element_qa',
    'generate_table_cell_recognition_qa',
    'generate_table_query_qa',
    'generate_table_structure_qa',
    'generate_table_vqa_qa',
    'generate_zeroelement_qa'
]
