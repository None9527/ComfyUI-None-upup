"""
Edge-Aware Precision Enhancement + Luminosity Zone Processing
边缘感知精确锐化 + 亮度区域光感增强

完全从第一性原理设计：
1. 锐化：只在边缘区域进行，其他完全不动
2. 光感：按亮度分区，每个区域独立处理中频对比度
"""

import torch
import numpy as np


class CinematicEnhancerNode:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sharpness": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "边缘锐化 (高频细节)"
                }),
                "luminosity": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "光感层次 (中频对比度)"
                }),
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL",),
                "shadow_lift": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "阴影提亮 (透气感)"
                }),
                "highlight_roll": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "高光回退 (避免过曝)"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "enhance"
    CATEGORY = "None-upup"

    def enhance(self, image, sharpness, luminosity, upscale_model=None, 
                shadow_lift=0.15, highlight_roll=0.1):
        import cv2
        
        if upscale_model is not None:
            image = self._upscale(image, upscale_model)
        
        batch_size = image.shape[0]
        result = []
        
        for i in range(batch_size):
            img_np = (image[i].cpu().numpy() * 255).astype(np.float64)
            
            # === 模块1：边缘感知精确锐化 ===
            if sharpness > 0:
                img_np = self._edge_aware_sharpen(img_np, sharpness)
            
            # === 模块2：亮度区域光感增强 ===
            if luminosity > 0:
                img_np = self._luminosity_zone_enhance(
                    img_np, luminosity, shadow_lift, highlight_roll
                )
            
            img_out = np.clip(img_np, 0, 255).astype(np.uint8)
            result.append(torch.from_numpy(img_out.astype(np.float32) / 255.0))
        
        return (torch.stack(result),)

    def _edge_aware_sharpen(self, img, sharpness):
        """边缘感知精确锐化 - 只锐化边缘"""
        import cv2
        
        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
        L = lab[:, :, 0].astype(np.float64)
        
        # Sobel边缘检测
        grad_x = cv2.Sobel(L, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(L, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        
        # 边缘掩码
        grad_max = gradient.max()
        if grad_max > 0:
            edge_mask = gradient / grad_max
            edge_mask = np.clip((edge_mask - 0.1) / 0.9, 0, 1)
            edge_mask = cv2.GaussianBlur(edge_mask, (0, 0), 1.0)
        else:
            return img
        
        # 精确USM (极小radius)
        blur = cv2.GaussianBlur(L, (0, 0), 0.8)
        detail = L - blur
        detail = np.clip(detail, -50, 50)  # 限幅
        
        amount = 0.5 + sharpness * 1.5
        L_sharp = L + detail * amount * edge_mask
        
        # 合并
        L_out = np.clip(L_sharp, 0, 255).astype(np.uint8)
        lab[:, :, 0] = L_out
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float64)

    def _luminosity_zone_enhance(self, img, intensity, shadow_lift, highlight_roll):
        """
        亮度区域光感增强 - 突破性算法
        
        原理：按亮度值将图像分成三个区域，每个区域独立增强中频对比度
        """
        import cv2
        
        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
        L = lab[:, :, 0].astype(np.float64)
        L_norm = L / 255.0
        
        # === Step 1: 生成亮度区域掩码 (使用smoothstep平滑过渡) ===
        shadows_mask = self._smoothstep(0.35, 0.20, L_norm)      # 暗部
        highlights_mask = self._smoothstep(0.65, 0.80, L_norm)   # 亮部
        midtones_mask = 1.0 - shadows_mask - highlights_mask     # 中间调
        midtones_mask = np.clip(midtones_mask, 0, 1)
        
        # 掩码边缘软化
        shadows_mask = cv2.GaussianBlur(shadows_mask, (0, 0), 3.0)
        highlights_mask = cv2.GaussianBlur(highlights_mask, (0, 0), 3.0)
        midtones_mask = cv2.GaussianBlur(midtones_mask, (0, 0), 3.0)
        
        # === Step 2: 每个区域独立的中频对比度增强 ===
        # 使用大radius模糊提取低频，增强的是中频而非高频
        
        # 阴影区：轻微提亮，增加透气感
        shadow_base = cv2.GaussianBlur(L, (0, 0), 25)
        shadow_detail = L - shadow_base
        shadow_enhanced = shadow_detail * (1.0 + shadow_lift * intensity * 2.0)
        shadow_result = shadow_base + shadow_enhanced
        # 轻微整体提亮
        shadow_result = shadow_result + shadow_lift * intensity * 15
        
        # 中间调：增强局部对比度，这是视觉焦点
        mid_base = cv2.GaussianBlur(L, (0, 0), 40)
        mid_detail = L - mid_base
        mid_enhanced = mid_detail * (1.0 + intensity * 0.8)
        mid_result = mid_base + mid_enhanced
        
        # 高光区：轻微压缩，增加细节可见性
        high_base = cv2.GaussianBlur(L, (0, 0), 30)
        high_detail = L - high_base
        high_enhanced = high_detail * (1.0 + highlight_roll * intensity * 0.5)
        high_result = high_base + high_enhanced
        # 轻微压缩高光
        high_result = high_result - highlight_roll * intensity * 10
        
        # === Step 3: 按掩码混合 ===
        L_enhanced = (
            shadows_mask * shadow_result +
            midtones_mask * mid_result +
            highlights_mask * high_result
        )
        
        # 保持原始亮度分布中心（防止整体漂移）
        L_enhanced = L_enhanced - L_enhanced.mean() + L.mean()
        
        # 软限幅
        L_enhanced = np.clip(L_enhanced, 0, 255)
        
        # === Step 4: 合并回图像 ===
        lab[:, :, 0] = L_enhanced.astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float64)

    def _smoothstep(self, edge0, edge1, x):
        """平滑过渡函数 (Hermite插值)"""
        t = np.clip((x - edge0) / (edge1 - edge0 + 1e-6), 0, 1)
        return t * t * (3 - 2 * t)

    def _upscale(self, image, upscale_model):
        from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
        upscaler = ImageUpscaleWithModel()
        return upscaler.upscale(upscale_model, image)[0]
