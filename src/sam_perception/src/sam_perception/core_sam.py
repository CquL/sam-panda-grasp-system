import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

class SAMProcessor:
    def __init__(self, model_type="vit_h", checkpoint_path="sam_vit_h_4b8939.pth", device="cuda"):
        """初始化 SAM 模型"""
        self.device = device
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
        print(f"[SAMProcessor] 模型已加载: {checkpoint_path}")

    def predict_mask(self, image_np, bbox):
        """
        根据图像和边界框生成掩码
        :param image_np: RGB 图像的 numpy 数组 (H, W, 3)
        :param bbox: 边界框 [x_min, y_min, x_max, y_max]
        :return: mask (二维布尔数组)
        """
        # 设置图像
        self.predictor.set_image(image_np)
        
        # 转换 bbox 格式并进行推理
        input_box = np.array(bbox)
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        
        # 返回得分最高的 mask (格式为布尔型二维矩阵)
        return masks[0]