from mmengine.registry import LOOPS, HOOKS
from mmengine.runner import TestLoop
import torch
import numpy as np
from mmcv.ops import nms_rotated
import gaussian_conv as gc
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample

@LOOPS.register_module()
class TTALoop(TestLoop):    
    """Test Time Augmentation Loop.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): An iterator to generate one batch of
            dataset each iteration.
    """

    def __init__(self, runner, dataloader, evaluator, fp16=False):
        # Chama o construtor da classe pai (TestLoop)
        # É aqui que o 'evaluator' é devidamente registrado
        super().__init__(runner, dataloader, evaluator)
        self.fp16 = fp16

    def run(self) -> None:
        self.runner.call_hook('before_run')
        self.runner.call_hook('before_val_epoch')

        tta_angle = [0, 45, 90]

        for i, data_batch in enumerate(self.dataloader):
            print(f"Processing batch {i+1}/{len(self.dataloader)}")
            self.runner.call_hook('before_val_iter', batch_idx=i, data_batch=data_batch)
            
            num_augs = len(data_batch['inputs'])
            aug_results = []

            # Loop interno: processa cada aumento (TTA) individualmente
            for j in range(num_augs):
                raw_input = data_batch['inputs'][j]
                raw_sample = data_batch['data_samples'][j]

                if isinstance(raw_input, (list, tuple)):
                    raw_input = raw_input[0]
                if isinstance(raw_sample, (list, tuple)):
                    raw_sample = raw_sample[0]

                single_data = {
                    'inputs': [raw_input], 
                    'data_samples': [raw_sample]
                }
                
                
                with torch.no_grad():
                    result = self.runner.model.test_step(single_data)
                    pred_sample = result[0]
                    # Registra o ângulo para usar na inversão dentro do merge_tta_output
                    pred_sample.set_metainfo({'rotate_angle': tta_angle[j]})
                    aug_results.append(pred_sample)
            
            # --- Fim do loop de aumentos ---
            
            # Realiza o merge dos resultados (deve ocorrer uma vez por imagem)
            if len(aug_results) > 1:
                merged_box, merged_scores, final_label = self.merge_tta_output(aug_results)
            else:
                merged_box = aug_results[0].pred_instances.bboxes
                merged_scores = aug_results[0].pred_instances.scores
                final_label = aug_results[0].pred_instances.labels
        # '
            
            pred_instances = InstanceData(
                bboxes=merged_box,    
                scores=merged_scores, 
                labels=final_label         
            )

    
            merged_sample = DetDataSample()
            merged_sample.pred_instances = pred_instances
            merged_sample.gt_instances = aug_results[0].gt_instances
            merged_sample.ignored_instances = aug_results[0].ignored_instances
            # Copia metadados essenciais (ori_shape, img_id, etc.) da primeira amostra
            merged_sample.set_metainfo(aug_results[0].metainfo)
            

            
            # Processa os resultados no avaliador
            self.evaluator.process(
                data_samples=[merged_sample], 
                data_batch=data_batch
)

            # Chama o hook de pós-iteração
            self.runner.call_hook('after_val_iter', 
                                  batch_idx=i, 
                                  data_batch=data_batch, 
                                  outputs=[merged_sample])

        # --- Fim do loop de dataloader ---
        
        # Avalia as métricas acumuladas
        
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))

        self.runner.call_hook('after_val_epoch', metrics=metrics)
        self.runner.call_hook('after_run')

    

    def merge_tta_output(self, results):
        all_bboxes = []
        all_scores = []
        
        for res in results:
            if len(res.pred_instances) == 0:
                continue

            bboxes = res.pred_instances.bboxes.clone()
            scores = res.pred_instances.scores
            labels = res.pred_instances.labels
            
            # 1. Geometry Inversion
            angle = res.metainfo.get('rotate_angle', 0)
            
            if angle != 0:
                bboxes = self.invert_rotation(bboxes, res.metainfo)

            all_bboxes.append(bboxes)
            all_scores.append(scores)
            final_label = labels

        if not all_bboxes:
            return torch.empty((0, 5)), torch.empty((0,)), torch.empty((0,), dtype=torch.long)
        
        g_boxes, g_scores = self.merge_gaussian_boxes(all_bboxes, all_scores)

        # # 3. Use Rotated NMS instead of averaging everything
        # # This keeps separate ships as separate detections
        # keep_indices = nms_rotated(g_boxes, g_scores, iou_threshold=0.5)[1]
        
        final_bboxes = g_boxes
        final_scores = g_scores
        
        return final_bboxes, final_scores, final_label
    
    def merge_gaussian_boxes(self, g_boxes, all_scores):
        if len(g_boxes) == 0:
            print("No boxes to merge.")
            return g_boxes, all_scores
    
        g_boxes_tensor = torch.stack(g_boxes) if isinstance(g_boxes, list) else g_boxes

        all_scores_tensor = torch.cat(all_scores) if isinstance(all_scores, list) else all_scores
        weights = all_scores_tensor.view(-1, 1)  # Transforma [N] em [N, 1]
        weighted_boxes = g_boxes_tensor * weights  # Multiplica cada box pelo seu score
        sum_weights = weights.sum(dim=0)  # Soma dos scores
        mean_box = weighted_boxes.mean(dim=0, keepdim=True)/sum_weights  # Média ponderada com os scores como pesos

        mean_scores = all_scores_tensor.mean(dim=0, keepdim=True)  # Score médio

        return mean_box, mean_scores
       

    def invert_rotation(self, bboxes, metainfo):
        # 1. Get shapes
        # Current shape is what the model saw (e.g., 781x1100)
        curr_h, curr_w = metainfo['img_shape'][:2]
        

        angle_deg = metainfo.get('rotate_angle', 0)
        print(f"Rotation angle to invert: {angle_deg} degrees\n")
        
        # 2. Determine the scale used
        sx = metainfo.get('scale_factor', (1.0, 1.0))[0]
        sy = metainfo.get('scale_factor', (1.0, 1.0))[1]
        

        # 3. Move boxes to the ORIGINAL scale before rotating
        # (Or rotate first, but use the CURR center)
        # Let's rotate around the current center:
        cx, cy = curr_w / 2, curr_h / 2
        
        rad = np.deg2rad(-angle_deg)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        print(f"Center of rotation: ({cx}, {cy}), cos: {cos_a}, sin: {sin_a}")

        x = bboxes[:, 0] - cx
        y = bboxes[:, 1] - cy
        
        # Rotation math
        new_x = x * cos_a - y * sin_a + cx
        new_y = x * sin_a + y * cos_a + cy
        
        # 4. NOW scale back to original image coordinates
        # This ensures they match the Ground Truth
        final_x = new_x / sx
        final_y = new_y / sy
        final_w = bboxes[:, 2] / sx
        final_h = bboxes[:, 3] / sy
        
        new_angle = bboxes[:, 4] - np.deg2rad(angle_deg)
        
        return torch.stack([final_x, final_y, final_w, final_h, new_angle], dim=-1)

    
        