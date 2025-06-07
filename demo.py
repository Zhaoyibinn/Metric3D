import torch
import cv2
model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
rgb = torch.tensor(cv2.imread("/home/zhaoyibin/3DRE/3DGS/FatesGS/DTU/set_22_25_28/scan37/images/0000.png")).permute(2, 0, 1).unsqueeze(0)
pred_depth, confidence, output_dict = model.inference({'input': rgb})
pred_normal = output_dict['prediction_normal'][:, :3, :, :] # only available for Metric3Dv2 i.e., ViT models
normal_confidence = output_dict['prediction_normal'][:, 3, :, :] # see https://arxiv.org/abs/2109.09881 for details