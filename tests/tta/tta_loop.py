import mmcv
import cv2
from mmengine.registry import LOOPS
from mmengine.runner import TestLoop
import torch
import numpy as np
from mmcv.ops import box_iou_rotated as IoU
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
            aug_list = []
            print(f"Processing batch {i+1}/{len(self.dataloader)}")
            self.runner.call_hook('before_val_iter', batch_idx=i, data_batch=data_batch)
            
            num_augs = len(data_batch['inputs'])
            canonical_boxes = None
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
                    gt_instances = raw_sample.gt_instances
                    ignored_instances = raw_sample.ignored_instances
                    result = self.runner.model.test_step(single_data)
                    pred_sample = result[0]
                    if canonical_boxes == None:
                        canonical_boxes = pred_sample.pred_instances.bboxes
                        canonical_scores = pred_sample.pred_instances.scores
                        augmented_boxes_list = [[box] for box in canonical_boxes]
                        augmented_scores_list = [[score] for score in canonical_scores]
                        aug_list.append(pred_sample)
                        continue
                    else:
                        pred_sample.set_metainfo({'rotate_angle': tta_angle[j]})
                        new_boxes = self.invert_rotation(pred_sample.pred_instances.bboxes, pred_sample.metainfo)
                        var = IoU(canonical_boxes, new_boxes)
                        ind_max = var.argmax(dim=1)
                        for i, list_of_boxes in enumerate(augmented_boxes_list):
                            if var[i][ind_max[i]] > 0.5:  # Threshold de IoU para considerar a caixa correspondente
                                list_of_boxes.append(new_boxes[ind_max[i]])                
                        for i, list_of_scores in enumerate(augmented_scores_list):
                            if var[i][ind_max[i]] > 0.5:  # Threshold de IoU para considerar a caixa correspondente
                                list_of_scores.append(pred_sample.pred_instances.scores[ind_max[i]])
            # --- Fim do loop de aumentos ---
            
            # Realiza o merge dos resultados (deve ocorrer uma vez por imagem)
            if len(augmented_boxes_list[0]) > 1:
                merged_box, merged_scores, final_label = self.merge_tta_output(augmented_boxes_list, augmented_scores_list)
                
            else:
                merged_box = aug_list[0].pred_instances.bboxes
                merged_scores = aug_list[0].pred_instances.scores
                final_label = aug_list[0].pred_instances.labels

            pred_instances = InstanceData(
                bboxes=merged_box,    
                scores=merged_scores, 
                labels=final_label         
            )

            
            
            merged_sample = DetDataSample()
            merged_sample.pred_instances = pred_instances
            merged_sample.gt_instances = gt_instances
            merged_sample.ignored_instances = ignored_instances
            merged_sample.set_metainfo(aug_list[0].metainfo)
            

            print(f"--- Debug Batch {i} ---")
            print(f"Num Preds: {len(merged_sample.pred_instances.bboxes)}")
            print(f"Num GTs: {len(merged_sample.gt_instances.bboxes)}")
            print(f"Exemplo Pred: {merged_sample.pred_instances.bboxes[0]}")
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


        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))

        self.runner.call_hook('after_val_epoch', metrics=metrics)
        self.runner.call_hook('after_run')


    def merge_tta_output(self, bbox_list, scores_list):
        
        for i in range (len(bbox_list)):
            print(bbox_list[i])
            final_bboxes, final_scores = self.merge_gaussian_boxes(bbox_list[i], scores_list[i])
            return final_bboxes, final_scores
    
    def merge_gaussian_boxes(self, g_boxes_list, all_scores_list):
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
       

    def invert_rotation(self, bboxes, metainfo):
        # 1. Recuperar metadados
        # scale_factor costuma ser [scale_w, scale_h]
        sc = metainfo.get('scale_factor', (1.0, 1.0))
        sx, sy = sc[0], sc[1]
        
        angle_deg = float(metainfo.get('rotate_angle', 0))
        angle_rad = angle_deg * (np.pi / 180.0)
        
        
        h, w = metainfo['img_shape'][:2]
        cx, cy = w / 2, h / 2
        
        # 3. Inverter Posição (x, y)
        # Criamos a matriz de rotação INVERSA (ângulo negativo)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        x = bboxes[:, 0] - cx
        y = bboxes[:, 1] - cy
        
        # Rotação do ponto (x, y) em torno da origem (agora cx, cy)
        new_x = x * cos_a - y * sin_a + cx
        new_y = x * sin_a + y * cos_a + cy
        
        # 4. Inverter Escala
        final_x = new_x / sx
        final_y = new_y / sy
        final_w = bboxes[:, 2] / sx
        final_h = bboxes[:, 3] / sy
        
        # 5. Inverter Orientação da Caixa (O ângulo theta)
        # Se a imagem girou +45, a caixa precisa girar -45 para voltar ao normal
        final_angle = bboxes[:, 4] - angle_rad
        
        return torch.stack([final_x, final_y, final_w, final_h, final_angle], dim=-1)

    
        