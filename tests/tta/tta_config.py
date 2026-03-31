import mmcv
from mmcv import imrotate
from mmcv.ops import box_iou_rotated as IoU

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.dataset import Compose

from typing import Dict
from mmengine.logging import HistoryBuffer
import torch
import numpy as np
import gaussian_conv as gc

from math import sqrt, ceil, floor

import cv2

from copy import deepcopy

from mmdet.utils import register_all_modules as register_all_modules_mmdet

from typing import List, Optional, Tuple, Union, no_type_check

from mmengine.structures import InstanceData
from mmengine.runner import load_checkpoint
from mmdet.structures import DetDataSample

from mmrotate.utils import register_all_modules

import torch.nn.functional as F



def run_tta(cfg, runner, angles):

    cfg.test_dataloader.dataset.test_mode = False

    evaluator = runner.build_evaluator(cfg.test_evaluator)  
    dataloader = runner.build_dataloader(cfg.test_dataloader)


    test_pipeline = cfg.val_pipeline
    
    
    dataloader.dataset.pipeline = Compose(test_pipeline)
    
    if hasattr(dataloader.dataset, 'metainfo'):
        evaluator.dataset_meta = dataloader.dataset.metainfo
        runner.visualizer.dataset_meta = \
            dataloader.dataset.metainfo
        
    """Launch test."""
    runner.call_hook('before_test')
    runner.call_hook('before_test_epoch')

    
    
   

    for i, data_batch in enumerate(dataloader):
        augmented_boxes = []
        all_scores = []
        print(f'Batch: {i}')
        padding_ok = False
        for angle in angles:
            batch_copy = data_batch.copy()
            

            if(angle == 0):
                outputs = runner.model.test_step(batch_copy)
                pred_sample = outputs[0]

                
                augmented_boxes.append(pred_sample.pred_instances.bboxes)
                all_scores.append(pred_sample.pred_instances.scores)
                
                

            else:
                metainfo = batch_copy['data_samples'][0].metainfo
               
                img = batch_copy['inputs'][0].permute(1,2,0)
                if padding_ok != True:
                    h,w = metainfo['img_shape'][0:2]
                    diagonal = sqrt(h*h + w*w)
                    diagonal = ceil(diagonal)
                    pad_top = floor((diagonal-h)/2)
                    pad_left = floor((diagonal-w)/2)
                    padded_img = mmcv.image.impad(img = img.cpu().numpy(), padding=(pad_left, pad_top))
                    shape = padded_img.shape
                    padding_ok = True
                    
                r_image = mmcv.imrotate(img = padded_img, angle = angle)
                new_img = r_image
                
                r_image= img.new_tensor(r_image).permute(2,0,1)
                
                batch_copy['inputs'][0] = r_image
                outputs = runner.model.test_step(batch_copy)
                
                pred_sample = outputs[0]
                
                ori_img = cv2.imread(metainfo['img_path'])
                
                if len(pred_sample.pred_instances.bboxes) == 0:
                    continue
                
                new_boxes = invert_rotation(pred_sample.pred_instances.bboxes, angle, shape, metainfo['scale_factor'] ,padding=(pad_left, pad_top))
                # draw_rotated_boxes(ori_img, new_boxes, f'tests/tta/images/{angle}check.jpg')
                
                augmented_boxes.append(new_boxes)
                all_scores.append(pred_sample.pred_instances.scores)
        
        if len(augmented_boxes) != 0:
                    
            augmented_boxes = torch.cat(augmented_boxes)
            all_scores = torch.cat(all_scores)
            sorted_scores = torch.sort(all_scores, descending=True)
            augmented_boxes = augmented_boxes[sorted_scores.indices]
            iou_trh = 0.5
            iou_matrix = IoU(augmented_boxes, augmented_boxes)
            
            

            iou_matrix = iou_matrix.triu(diagonal=1)
            iou_matrix[iou_matrix>=iou_trh] = 1.0
            iou_matrix[iou_matrix<iou_trh] = 0.0
            
            
            
            solo_ind = iou_matrix.sum(dim=0)==0
            
            iou_matrix.fill_diagonal_(1.0)
            
            iou_matrix = iou_matrix[solo_ind]
            
            valid_ind = iou_matrix.sum(dim=1)>=ceil(len(angles)*0.5)
            
            
            
            iou_matrix = iou_matrix[valid_ind]
       
            iou_matrix *= sorted_scores.values

            
            
            if iou_matrix.size()[0] != 0 :
                
            
                final_boxes, final_scores= merge_gaussian_boxes(augmented_boxes,iou_matrix,sorted_scores)
                
                new_instances = InstanceData()
                new_instances.bboxes = final_boxes  
                new_instances.scores = final_scores 
                new_instances.labels = torch.zeros(len(final_boxes), dtype=torch.long, device=final_boxes.device)
                

                pred_sample.pred_instances = new_instances

                evaluator.process(
                data_samples=[pred_sample],
                data_batch=data_batch
                )

                # draw_rotated_boxes(ori_img, f'tests/tta/images/{i}_img_check.jpg', final_boxes)
            else:
                evaluator.process(
                data_samples=[],
                data_batch=data_batch
                )
        
                # draw_rotated_boxes(ori_img, f'tests/tta/images/{i}_img_check.jpg')
        
       
        
    


    metrics = evaluator.evaluate(len(dataloader.dataset))
    runner.call_hook('after_val_epoch', metrics=metrics)
    runner.call_hook('after_run')
    print('Finalizado')
    return True



def invert_rotation(bboxes, angle_deg, img_shape, ori_scale, padding):

    angle_rad = angle_deg * (np.pi / 180.0)
    
    
    h_pad, w_pad = img_shape[:2]
    cx, cy = w_pad / 2.0, h_pad / 2.0
    
   
    x = bboxes[:, 0] * ori_scale[0] - cx
    y = bboxes[:, 1] * ori_scale[1] - cy
    

    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    
    
    new_x = (x * cos_a - y * sin_a) + cx - padding[0]
    new_y = (x * sin_a + y * cos_a) + cy - padding[1]
    
    new_angle = bboxes[:, 4] - angle_rad

    new_x, new_y = new_x/ori_scale[0], new_y/ori_scale[1]
    
    return torch.stack([new_x, new_y, bboxes[:, 2], bboxes[:, 3], new_angle], dim=-1)


def draw_rotated_boxes(img,save_path ,bboxes= None ):
   
    img_canvas = img.copy()
    if bboxes!=None:
        for box in bboxes:
            cx, cy, w, h, angle = box.tolist()
            
            rect = ((cx, cy), (w, h), angle * 180 / np.pi)
            box_pts = cv2.boxPoints(rect)
            box_pts = np.int0(box_pts)
            
            cv2.drawContours(img_canvas, [box_pts], 0, (0, 255, 0), 2)
        
    cv2.imwrite(save_path, img_canvas)

def merge_gaussian_boxes(augmented_boxes, iou_matrix ,sorted_scores):
    
    
    
    g_params = gc.rbbox_to_gaussian(augmented_boxes, scalar=1.0)
        
    aggregated_boxes = g_params.unsqueeze(0) * iou_matrix.unsqueeze(-1)
    
    
    aggregated_boxes = aggregated_boxes.sum(dim=1) / iou_matrix.sum(dim=1).unsqueeze(-1)

   
    # for i, list in enumerate(aggregated_boxes):
    #     # num_boxes = mask[i].sum(dim=0)
        
    #     # if (num_boxes.item()>1):
    #     #     ind = list.sum(dim=1) > 1
    #     #     final_scores.append(sorted_scores.values[ind].max())
    #     #     final_gaus.append(list.sum(dim=0))

        
    #     print(list)
    #     exit()
    #     # final_scores.append(sorted_scores.values[ind].max())
    # final_gaus.append(list)   
    
    # for tensor in final_gaus:
    final_boxes = gc.gaussian_to_rbbox(aggregated_boxes, 1.0)

    
    return final_boxes, iou_matrix.max(dim=1).values.unsqueeze(-1)


def main():

    register_all_modules_mmdet(init_default_scope=False)
    register_all_modules(init_default_scope=False)

    config_path = 'configs/rotated_retinanet/rotated-retinanet-hbox-oc_r50_fpn_rr-6x_hrsc.py'
    cfg = Config.fromfile(config_path)
    cfg.work_dir = 'work_dirs/tta_rotated_retinanet_test'
    cfg.load_from = 'checkpoint/rretinanet/epoch_72.pth'

    runner = Runner.from_cfg(cfg)

    load_checkpoint(runner.model, cfg.load_from, map_location='cuda:0')

    runner.model.eval()

        
    angles_for_aug= [0,45,90]

    run_tta(cfg, runner, angles_for_aug)
    
    
        

if __name__ == '__main__':
    main()


