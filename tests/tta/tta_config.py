import mmcv
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.dataset import Compose
from typing import Dict
from mmengine.logging import HistoryBuffer
import torch
import numpy as np
import gaussian_conv as gc
from mmcv.ops import box_iou_rotated as IoU

from mmdet.utils import register_all_modules as register_all_modules_mmdet

from mmengine.structures import InstanceData
from mmengine.runner import load_checkpoint
from mmdet.structures import DetDataSample

from mmrotate.utils import register_all_modules



def run_tta(cfg, runner, angles):

    cfg.test_dataloader.dataset.test_mode = False

    evaluator = runner.build_evaluator(cfg.test_evaluator)  
    dataloader = runner.build_dataloader(cfg.test_dataloader)


    test_pipeline = cfg.val_pipeline
    
    test_pipeline.insert(-1, dict(type='Rotate', rotate_angle=angles[0]))

    print(test_pipeline)
    
    
    dataloader.dataset.pipeline = Compose(test_pipeline)
    
    if hasattr(dataloader.dataset, 'metainfo'):
        evaluator.dataset_meta = dataloader.dataset.metainfo
        runner.visualizer.dataset_meta = \
            dataloader.dataset.metainfo
        
    """Launch test."""
    runner.call_hook('before_test')
    runner.call_hook('before_test_epoch')

    canon_boxes = []
    canon_scores = []
    aug_boxes = []
    aug_scores = []
    data_samples = []

    canonical_boxes = None
   

    for i, data_batch in enumerate(dataloader):
        for angle in angles:
            print(f'Angulo: {angle}; Batch: {i}')

            for transform in test_pipeline:
                    if transform.get('type') == 'Rotate':
                        transform['rotate_angle'] = angle
                        dataloader.dataset.pipeline = Compose(test_pipeline)
            with torch.no_grad():
                outputs = runner.model.test_step(data_batch)
                pred_sample = outputs[0]
                if angle == 0:
                    print(f"Base (0°): {pred_sample.pred_instances.bboxes[0, :2]}") # Centro da primeira detecção


            if(angle == 0):
                canonical_boxes = pred_sample.pred_instances.bboxes
                canonical_scores = pred_sample.pred_instances.scores
                augmented_boxes_list = [[box] for box in canonical_boxes]
                augmented_scores_list = [[score] for score in canonical_scores]

                canon_boxes.append(augmented_boxes_list)
                canon_scores.append(augmented_scores_list)
                data_samples.append(data_batch['data_samples'][0])

                final_sample = solo_evaluate(pred_sample, data_samples[i], evaluator)

                runner.call_hook('after_val_iter', 
                    batch_idx=i, 
                    data_batch=data_batch, 
                    outputs=[final_sample])

            else:
                new_boxes = invert_rotation(pred_sample.pred_instances.bboxes, angle, pred_sample.metainfo)
                print(f"Invertida ({angle}°): {new_boxes[0, :2]}")
                exit()
                print(new_boxes)

                var = IoU(canon_boxes[i], new_boxes)
                ind_max = var.argmax(dim=1)
                #Revisar a lógica por trás dos índices do var[j][ind_max[j]]
                for j, list_of_boxes in enumerate(canon_boxes[i]):
                    if var[j][ind_max[j]] > 0.5:  
                        list_of_boxes.append(new_boxes[ind_max[j]])  

                for j, list_of_scores in enumerate(canon_scores[i]):
                    if var[j][ind_max[j]] > 0.5:  
                        list_of_scores.append(pred_sample.pred_instances.scores[ind_max[j]])

                aug_boxes.append(list_of_boxes)
                aug_scores.append(list_of_scores)
                
    


    metrics = evaluator.evaluate(len(dataloader.dataset))
    runner.call_hook('after_val_epoch', metrics=metrics)
    runner.call_hook('after_run')
    print('Finalizado')
    return True

def solo_evaluate(boxes, data_sample, evaluator):
    final_boxes = boxes.pred_instances.bboxes
    final_score = boxes.pred_instances.scores
    final_label = boxes.pred_instances.labels

    pred_instances = InstanceData(
        bboxes=final_boxes,    
        scores=final_score, 
        labels=final_label         
    )

    final_sample = DetDataSample()
    final_sample.pred_instances = pred_instances
    final_sample.gt_instances =  data_sample.gt_instances
    
    final_sample.ignored_instances = data_sample.ignored_instances
    final_sample.set_metainfo(data_sample.metainfo)

    
    evaluator.process(
    data_samples=[final_sample]
)   

    return final_sample

def merge_n_evaluate(list_of_boxes):
    exit()

def merge_tta_output(bbox_list, scores_list):
    exit()
    for i in range (len(bbox_list)):
        print(bbox_list[i])
        final_bboxes, final_scores = self.merge_gaussian_boxes(bbox_list[i], scores_list[i])
        return final_bboxes, final_scores

def merge_gaussian_boxes(g_boxes_list, all_scores_list):
    # g_boxes_list: lista de tensores [N_i, 5]
    # all_scores_list: lista de tensores [N_i]
    
    if len(g_boxes_list) == 1:
        return g_boxes_list[0], all_scores_list[0]

    # 1. Concatenar tudo para processar em lote (batch)
    # Isso transforma as listas em tensores únicos [Total_N, 5] e [Total_N]
    all_boxes = torch.cat(g_boxes_list, dim=0)
    all_scores = torch.cat(all_scores_list, dim=0)
    weights = all_scores.view(-1, 1) # Shape [Total_N, 1] para multiplicar

    # 2. Converter caixas rotacionadas para parâmetros Gaussianos
    # Assumindo que gc.rbbox_to_gaussian aceite o tensor completo
    g_params = gc.rbbox_to_gaussian(all_boxes, scalar=1.0) 

    # 3. Calcular a Média Ponderada
    # Multiplicamos os parâmetros pelo peso (score)
    weighted_params = g_params * weights
    
    # Soma dos parâmetros ponderados dividida pela soma dos pesos
    # dim=0 calcula a média entre todas as detecções existentes
    sum_weights = weights.sum(dim=0) + 1e-6 # Evita divisão por zero
    mean_param = weighted_params.sum(dim=0, keepdim=True) / sum_weights

    # 4. Converter de volta para formato rbbox (x, y, w, h, angle)
    mean_box = gc.gaussian_to_rbbox(mean_param, scalar_div=1.0)
    
    # 5. Score médio final
    mean_score = all_scores.mean(dim=0, keepdim=True)

    return mean_box, mean_score
    

def invert_rotation(bboxes, angle_deg, metainfo):
    # 1. Ângulo e Centro (O MMDet rotaciona a imagem redimensionada)
    angle_rad = angle_deg * (np.pi / 180.0)
    
    # IMPORTANTE: Se o Resize ocorreu antes, o modelo já reescalou 
    # as predições para a ori_shape. O centro deve ser da ori_shape.
    h_ori, w_ori = metainfo['ori_shape'][:2]
    cx, cy = w_ori / 2, h_ori / 2
    
    # 2. Transladar para o centro
    x = bboxes[:, 0] - cx
    y = bboxes[:, 1] - cy
    
    # 3. Rotação Inversa (Note o sinal negativo no ângulo para desfazer)
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    
    new_x = (x * cos_a - y * sin_a) + cx
    new_y = (x * sin_a + y * cos_a) + cy
    
    # 4. Ângulo da Caixa
    # Se a imagem girou +X, a caixa precisa girar -X para voltar ao normal
    new_angle = bboxes[:, 4] - angle_rad
    
    return torch.stack([new_x, new_y, bboxes[:, 2], bboxes[:, 3], new_angle], dim=-1)





def main():

    register_all_modules_mmdet(init_default_scope=False)
    register_all_modules(init_default_scope=False)

    config_path = 'configs/rotated_retinanet/rotated_retinanet_hbb_r50_fpn_6x_hrsc_rr_oc.py'
    cfg = Config.fromfile(config_path)
    cfg.work_dir = 'work_dirs/tta_rotated_retinanet_test'
    cfg.load_from = 'checkpoint/rretinanet/epoch_72.pth'

    runner = Runner.from_cfg(cfg)

    runner.model.test_cfg.score_thr = 0.1

    load_checkpoint(runner.model, cfg.load_from, map_location='cuda:0')

    runner.model.eval()

        
    angles_for_aug= [0,45,90]

    run_tta(cfg, runner, angles_for_aug)
    
    exit()
        

if __name__ == '__main__':
    main()


