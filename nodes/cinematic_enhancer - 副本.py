"""
Ultra Professional Image Enhancer
核心算法:
1. Laplacian金字塔多尺度增强 - 专业级无伪影
2. 边缘感知自适应处理 - 只增强有意义的细节
3. Soft Clipping防止过冲 - 消除halo
4. 保守参数策略 - 自然优先
"""

import torch
import numpy as np


class CinematicEnhancerNode:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "enhance_level": (["subtle", "natural", "enhanced", "strong"], {
                    "default": "natural",
                    "tooltip": "增强级别"
                }),
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL",),
                "fine_tune": ("FLOAT", {
                    "default": 0.0,
                    "min": -0.5,
                    "max": 0.5,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "微调 (负=更柔和, 正=更锐利)"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "enhance"
    CATEGORY = "None-upup"

    # 预设参数 (非常保守)
    PRESETS = {
        "subtle":   {"detail": 0.10, "contrast": 0.08, "texture": 0.05},
        "natural":  {"detail": 0.18, "contrast": 0.12, "texture": 0.10},
        "enhanced": {"detail": 0.28, "contrast": 0.18, "texture": 0.15},
        "strong":   {"detail": 0.40, "contrast": 0.25, "texture": 0.20},
    }

    def enhance(self, image, enhance_level, upscale_model=None, fine_tune=0.0):
        import cv2
        
        if upscale_model is not None:
            image = self._upscale(image, upscale_model)
        
        params = self.PRESETS[enhance_level]
        detail = params["detail"] + fine_tune
        contrast = params["contrast"] + fine_tune * 0.5
        texture = params["texture"] + fine_tune * 0.5
        
        # 限制在安全范围
        detail = np.clip(detail, 0.0, 0.6)
        contrast = np.clip(contrast, 0.0, 0.4)
        texture = np.clip(texture, 0.0, 0.3)
        
        batch_size = image.shape[0]
        result = []
        
        for i in range(batch_size):
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            
            # 转LAB，只处理L通道
            img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            L, A, B = cv2.split(img_lab)
            L_float = L.astype(np.float64)
            
            # === 核心算法：Laplacian金字塔增强 ===
            L_enhanced = self._laplacian_pyramid_enhance(L_float, detail, contrast, texture)
            
            # 输出
            L_out = np.clip(L_enhanced, 0, 255).astype(np.uint8)
            img_out = cv2.cvtColor(cv2.merge([L_out, A, B]), cv2.COLOR_LAB2RGB)
            result.append(torch.from_numpy(img_out.astype(np.float32) / 255.0))
        
        return (torch.stack(result),)

    def _laplacian_pyramid_enhance(self, L, detail_strength, contrast_strength, texture_strength):
        """
        Laplacian金字塔多尺度增强 - 专业级无伪影算法
        
        原理：
        1. 构建高斯金字塔 (多尺度分解)
        2. 从高斯金字塔构建拉普拉斯金字塔 (每层=该层高斯-上层高斯上采样)
        3. 对不同层级应用不同增强 (细节层轻增强，中层适度，粗层保守)
        4. 使用边缘感知权重避免伪影
        5. 重建时使用soft clipping防止过冲
        """
        import cv2
        
        h, w = L.shape
        
        # 构建4层高斯金字塔
        num_levels = 4
        gaussian_pyramid = [L.copy()]
        for i in range(num_levels - 1):
            down = cv2.pyrDown(gaussian_pyramid[-1])
            gaussian_pyramid.append(down)
        
        # 构建拉普拉斯金字塔 (高频细节)
        laplacian_pyramid = []
        for i in range(num_levels - 1):
            up = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=(gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))
            lap = gaussian_pyramid[i] - up
            laplacian_pyramid.append(lap)
        laplacian_pyramid.append(gaussian_pyramid[-1])  # 最后一层是低频残差
        
        # 计算边缘感知权重 (避免在平坦区域和强边缘过度增强)
        edge_weights = []
        for i, lap in enumerate(laplacian_pyramid[:-1]):
            # 边缘强度
            grad_x = cv2.Sobel(gaussian_pyramid[i], cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gaussian_pyramid[i], cv2.CV_64F, 0, 1, ksize=3)
            edge_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            # 归一化
            edge_norm = edge_mag / (edge_mag.max() + 1e-6)
            
            # 软阈值：中等边缘增强最多，太强或太弱都减少
            # 使用高斯曲线：在0.3处最大
            weight = np.exp(-((edge_norm - 0.3) ** 2) / 0.1)
            weight = cv2.GaussianBlur(weight, (0, 0), 2.0)
            edge_weights.append(weight)
        
        # 层级增强系数 (细节层最轻)
        level_factors = {
            0: texture_strength * 0.8,    # 最细节 - 最轻
            1: detail_strength * 1.0,     # 细节
            2: contrast_strength * 1.2,   # 中等尺度
        }
        
        # 增强拉普拉斯金字塔
        enhanced_laplacian = []
        for i, lap in enumerate(laplacian_pyramid[:-1]):
            if i in level_factors:
                factor = level_factors[i]
                weight = edge_weights[i]
                
                # 自适应增强
                boost = 1.0 + factor * weight
                enhanced = lap * boost
                
                # Soft clipping - 防止过冲/halo
                enhanced = self._soft_clip(enhanced, lap, factor * 30)
                
                enhanced_laplacian.append(enhanced)
            else:
                enhanced_laplacian.append(lap)
        enhanced_laplacian.append(laplacian_pyramid[-1])  # 低频不变
        
        # 重建图像
        reconstructed = enhanced_laplacian[-1]
        for i in range(num_levels - 2, -1, -1):
            up = cv2.pyrUp(reconstructed, dstsize=(enhanced_laplacian[i].shape[1], enhanced_laplacian[i].shape[0]))
            reconstructed = up + enhanced_laplacian[i]
        
        return reconstructed

    def _soft_clip(self, enhanced, original, threshold):
        """
        Soft Clipping - 防止增强过度产生halo
        当增强后的值偏离原始值过多时，使用tanh软限制
        """
        diff = enhanced - original
        # tanh软限制
        diff_clipped = threshold * np.tanh(diff / (threshold + 1e-6))
        return original + diff_clipped

    def _upscale(self, image, upscale_model):
        from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
        upscaler = ImageUpscaleWithModel()
        return upscaler.upscale(upscale_model, image)[0]
