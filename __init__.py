# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import paddleocr
# from .paddleocr import *

__version__ = paddleocr.VERSION
__all__ = [
#           'PaddleOCR', 'PPStructure', 'draw_ocr', 'draw_structure_result', 'save_structure_res','download_with_progressbar',
          'PaddleocrSAST']


import cv2
import linora as la
import numpy as np

from ppocr.utils.utility import get_image_file_list
from tools.infer.predict_det import TextDetector

args = la.utils.Config(det_model_dir='', use_gpu=True, ir_optim=True, use_tensorrt=False, min_subgraph_size=10, precision="fp32", gpu_mem=500,
det_algorithm="SAST", det_limit_side_len=960, det_limit_type='max', det_db_thresh=0.3, det_db_box_thresh=0.6, 
det_db_unclip_ratio=1.5, max_batch_size=10, use_dilation=False, det_db_score_mode="fast", det_east_score_thresh=0.8,
det_east_cover_thresh=0.1, det_east_nms_thresh=0.2, 
det_sast_score_thresh=0.5,
det_sast_nms_thresh=0.2,
det_sast_polygon=True,
rec_algorithm='CRNN',
rec_model_dir='',
rec_image_shape="3, 32, 320",
rec_char_type='ch',
rec_batch_num=6,
max_text_length=25,
rec_char_dict_path="./ppocr/utils/ppocr_keys_v1.txt",
use_space_char=True,
vis_font_path="./doc/fonts/simfang.ttf",
drop_score=0.5,
e2e_algorithm='PGNet',
e2e_model_dir='',
e2e_limit_side_len=768,
e2e_limit_type='max',
e2e_pgnet_score_thresh=0.5,
e2e_char_dict_path="./ppocr/utils/ic15_dict.txt",
e2e_pgnet_valid_set='totaltext',
e2e_pgnet_polygon=True,
e2e_pgnet_mode='fast',
use_angle_cls=False,
cls_model_dir='',
cls_image_shape="3, 48, 192",
label_list=['0', '180'],
cls_batch_num=6,
cls_thresh=0.9,
enable_mkldnn=False,
cpu_threads=10,
use_pdserving=False,
warmup=True,
use_mp=False,
total_process_num=1,
process_id=0,
benchmark=False,
save_log_path="./log_output/",
show_log=True)

class PaddleocrSAST():
    def __init__(self, det_model_dir):
        self.args = args
        self.args.det_model_dir = det_model_dir
        self.text_detector = TextDetector(self.args)
        img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
        for i in range(2):
            res = self.text_detector(img)
        
    def predict(self, image_dir):
        image_file_list = get_image_file_list(image_dir)
        img = cv2.imread(image_file_list[0])
        dt_boxes, _ = self.text_detector(img)
        return dt_boxes
