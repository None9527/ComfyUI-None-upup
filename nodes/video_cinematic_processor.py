"""
Video Cinematic Processor - GPU BF16 多线程视频处理
视频拆帧 → 画质增强 → 补帧 → 合成视频

功能：
1. 视频拆帧：支持任意视频格式
2. GPU BF16处理：边缘锐化 + 光感增强
3. 补帧：RIFE光流补帧 2x/4x
4. 视频合成：支持多种编码格式
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import tempfile
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, List, Optional
import folder_paths


class VideoCinematicProcessor:
    """视频画质增强处理器 - GPU BF16多线程"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "sharpness": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "边缘锐化强度"
                }),
                "luminosity": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "光感层次强度"
                }),
                "frame_interpolation": (["none", "2x", "4x"], {
                    "default": "none",
                    "tooltip": "补帧倍数 (RIFE光流)"
                }),
            },
            "optional": {
                "shadow_lift": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "阴影提亮"
                }),
                "highlight_roll": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "高光回退"
                }),
                "batch_size": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 32,
                    "step": 1,
                    "tooltip": "GPU批处理大小"
                }),
                "num_workers": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 16,
                    "step": 1,
                    "tooltip": "多线程工作数"
                }),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "process_video"
    CATEGORY = "None-upup"

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.rife_model = None

    def process_video(self, video, sharpness, luminosity, frame_interpolation="none",
                      shadow_lift=0.15, highlight_roll=0.1, batch_size=4, num_workers=4):
        """
        主处理流程：拆帧 → 增强 → 补帧 → 合成
        """
        # Step 1: 拆帧
        frames, fps, audio_path = self._extract_frames(video)
        original_fps = fps
        
        # Step 2: GPU BF16批量处理
        enhanced_frames = self._batch_enhance_gpu(
            frames, sharpness, luminosity, shadow_lift, highlight_roll, batch_size
        )
        
        # Step 3: 补帧
        if frame_interpolation != "none":
            multiplier = 2 if frame_interpolation == "2x" else 4
            enhanced_frames = self._interpolate_frames(enhanced_frames, multiplier, batch_size)
            fps = original_fps * multiplier
        
        # Step 4: 合成视频
        output_video = self._compose_video(enhanced_frames, fps, audio_path)
        
        return (output_video,)

    def _extract_frames(self, video) -> Tuple[torch.Tensor, float, Optional[str]]:
        """
        从视频中提取帧
        返回: (frames_tensor [N,H,W,C], fps, audio_path)
        """
        import cv2
        
        # 如果是文件路径
        if isinstance(video, str):
            video_path = video
        elif hasattr(video, 'path'):
            video_path = video.path
        else:
            # 假设是tensor格式 [N,H,W,C]
            return video, 30.0, None
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        
        # 提取音频
        audio_path = self._extract_audio(video_path)
        
        # 转换为tensor [N,H,W,C] float32 0-1
        frames_np = np.stack(frames, axis=0).astype(np.float32) / 255.0
        frames_tensor = torch.from_numpy(frames_np)
        
        return frames_tensor, fps, audio_path

    def _extract_audio(self, video_path: str) -> Optional[str]:
        """提取音频轨道"""
        try:
            audio_path = os.path.join(tempfile.gettempdir(), "temp_audio.aac")
            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "copy", audio_path
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            return audio_path if os.path.exists(audio_path) else None
        except:
            return None

    def _batch_enhance_gpu(self, frames: torch.Tensor, sharpness: float, 
                           luminosity: float, shadow_lift: float, 
                           highlight_roll: float, batch_size: int) -> torch.Tensor:
        """
        GPU BF16批量增强处理
        frames: [N, H, W, C]
        """
        N, H, W, C = frames.shape
        results = []
        
        # 转换为 [N, C, H, W] 格式用于GPU处理
        frames_gpu = frames.permute(0, 3, 1, 2).to(self.device, dtype=self.dtype)
        
        for i in range(0, N, batch_size):
            batch = frames_gpu[i:i+batch_size]
            
            # 边缘感知锐化 (GPU)
            if sharpness > 0:
                batch = self._edge_sharpen_gpu(batch, sharpness)
            
            # 光感增强 (GPU)
            if luminosity > 0:
                batch = self._luminosity_enhance_gpu(batch, luminosity, shadow_lift, highlight_roll)
            
            results.append(batch.cpu())
            
            # 清理GPU内存
            if i % (batch_size * 4) == 0:
                torch.cuda.empty_cache()
        
        # 合并并转回 [N, H, W, C]
        enhanced = torch.cat(results, dim=0)
        enhanced = enhanced.permute(0, 2, 3, 1).to(torch.float32)
        enhanced = torch.clamp(enhanced, 0, 1)
        
        return enhanced

    def _edge_sharpen_gpu(self, batch: torch.Tensor, sharpness: float) -> torch.Tensor:
        """
        GPU边缘感知锐化
        batch: [B, C, H, W] BF16
        """
        # Sobel边缘检测核
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=self.dtype, device=self.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=self.dtype, device=self.device).view(1, 1, 3, 3)
        
        # 转灰度计算边缘
        gray = batch.mean(dim=1, keepdim=True)
        
        # 边缘检测
        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        gradient = torch.sqrt(grad_x**2 + grad_y**2)
        
        # 归一化边缘掩码
        grad_max = gradient.amax(dim=(2, 3), keepdim=True) + 1e-6
        edge_mask = gradient / grad_max
        edge_mask = torch.clamp((edge_mask - 0.1) / 0.9, 0, 1)
        
        # 高斯模糊
        gaussian_kernel = self._get_gaussian_kernel(5, 0.8).to(self.device, dtype=self.dtype)
        edge_mask = F.conv2d(edge_mask, gaussian_kernel, padding=2)
        
        # USM锐化
        blur = F.conv2d(batch, gaussian_kernel.repeat(3, 1, 1, 1), padding=2, groups=3)
        detail = batch - blur
        detail = torch.clamp(detail, -0.2, 0.2)  # 限幅
        
        amount = 0.5 + sharpness * 1.5
        sharpened = batch + detail * amount * edge_mask
        
        return sharpened

    def _luminosity_enhance_gpu(self, batch: torch.Tensor, intensity: float,
                                shadow_lift: float, highlight_roll: float) -> torch.Tensor:
        """
        GPU亮度区域光感增强
        batch: [B, C, H, W] BF16
        """
        # 计算亮度通道
        luminance = 0.299 * batch[:, 0:1] + 0.587 * batch[:, 1:2] + 0.114 * batch[:, 2:3]
        
        # 生成亮度区域掩码
        shadows_mask = self._smoothstep_gpu(0.35, 0.20, luminance)
        highlights_mask = self._smoothstep_gpu(0.65, 0.80, luminance)
        midtones_mask = torch.clamp(1.0 - shadows_mask - highlights_mask, 0, 1)
        
        # 高斯模糊掩码
        blur_kernel = self._get_gaussian_kernel(7, 3.0).to(self.device, dtype=self.dtype)
        shadows_mask = F.conv2d(shadows_mask, blur_kernel, padding=3)
        highlights_mask = F.conv2d(highlights_mask, blur_kernel, padding=3)
        midtones_mask = F.conv2d(midtones_mask, blur_kernel, padding=3)
        
        # 大半径模糊提取低频
        large_blur_kernel = self._get_gaussian_kernel(25, 8.0).to(self.device, dtype=self.dtype)
        base = F.conv2d(batch, large_blur_kernel.repeat(3, 1, 1, 1), padding=12, groups=3)
        detail = batch - base
        
        # 各区域独立增强
        shadow_factor = 1.0 + shadow_lift * intensity * 2.0
        mid_factor = 1.0 + intensity * 0.8
        high_factor = 1.0 + highlight_roll * intensity * 0.5
        
        enhanced = (
            shadows_mask * (base + detail * shadow_factor + shadow_lift * intensity * 0.06) +
            midtones_mask * (base + detail * mid_factor) +
            highlights_mask * (base + detail * high_factor - highlight_roll * intensity * 0.04)
        )
        
        # 保持原始亮度中心
        enhanced = enhanced - enhanced.mean(dim=(2, 3), keepdim=True) + batch.mean(dim=(2, 3), keepdim=True)
        
        return enhanced

    def _smoothstep_gpu(self, edge0: float, edge1: float, x: torch.Tensor) -> torch.Tensor:
        """GPU平滑过渡函数"""
        t = torch.clamp((x - edge0) / (edge1 - edge0 + 1e-6), 0, 1)
        return t * t * (3 - 2 * t)

    def _get_gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        """生成高斯卷积核"""
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        kernel = g.outer(g)
        return kernel.view(1, 1, size, size)

    def _interpolate_frames(self, frames: torch.Tensor, multiplier: int, 
                            batch_size: int) -> torch.Tensor:
        """
        RIFE光流补帧
        frames: [N, H, W, C]
        """
        if multiplier == 1:
            return frames
        
        # 尝试加载RIFE模型
        rife_model = self._load_rife_model()
        
        if rife_model is None:
            # 如果没有RIFE，使用简单双线性插值
            return self._simple_interpolate(frames, multiplier)
        
        N = frames.shape[0]
        interpolated = []
        
        # 转换格式 [N, C, H, W]
        frames_gpu = frames.permute(0, 3, 1, 2).to(self.device, dtype=self.dtype)
        
        for i in range(N - 1):
            frame0 = frames_gpu[i:i+1]
            frame1 = frames_gpu[i+1:i+2]
            
            interpolated.append(frame0.cpu())
            
            # 生成中间帧
            for t in range(1, multiplier):
                timestep = t / multiplier
                with torch.no_grad():
                    mid_frame = rife_model(frame0, frame1, timestep)
                interpolated.append(mid_frame.cpu())
            
            if i % 10 == 0:
                torch.cuda.empty_cache()
        
        # 添加最后一帧
        interpolated.append(frames_gpu[-1:].cpu())
        
        # 合并并转回 [N, H, W, C]
        result = torch.cat(interpolated, dim=0)
        result = result.permute(0, 2, 3, 1).to(torch.float32)
        
        return result

    def _simple_interpolate(self, frames: torch.Tensor, multiplier: int) -> torch.Tensor:
        """简单双线性补帧 (备用方案)"""
        N, H, W, C = frames.shape
        interpolated = []
        
        for i in range(N - 1):
            frame0 = frames[i]
            frame1 = frames[i + 1]
            
            interpolated.append(frame0)
            
            for t in range(1, multiplier):
                alpha = t / multiplier
                mid_frame = (1 - alpha) * frame0 + alpha * frame1
                interpolated.append(mid_frame)
        
        interpolated.append(frames[-1])
        
        return torch.stack(interpolated, dim=0)

    def _load_rife_model(self):
        """加载RIFE补帧模型"""
        if self.rife_model is not None:
            return self.rife_model
        
        try:
            # 尝试从ComfyUI-Frame-Interpolation加载
            from custom_nodes.ComfyUI_Frame_Interpolation.rife_model import RIFE
            self.rife_model = RIFE().to(self.device)
            self.rife_model.eval()
            return self.rife_model
        except ImportError:
            pass
        
        try:
            # 尝试从VFI加载
            from custom_nodes.ComfyUI_VFI.rife import load_rife
            self.rife_model = load_rife().to(self.device)
            return self.rife_model
        except ImportError:
            pass
        
        print("[VideoCinematicProcessor] RIFE模型未找到，使用双线性插值")
        return None

    def _compose_video(self, frames: torch.Tensor, fps: float, 
                       audio_path: Optional[str]) -> str:
        """
        合成输出视频
        frames: [N, H, W, C] float32 0-1
        """
        import cv2
        
        output_dir = folder_paths.get_output_directory()
        output_path = os.path.join(output_dir, f"cinematic_video_{os.getpid()}.mp4")
        temp_video = os.path.join(tempfile.gettempdir(), "temp_video.mp4")
        
        N, H, W, C = frames.shape
        
        # OpenCV写入视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(temp_video, fourcc, fps, (W, H))
        
        for i in range(N):
            frame = (frames[i].numpy() * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
        
        writer.release()
        
        # 使用FFmpeg合并音频并编码
        if audio_path and os.path.exists(audio_path):
            cmd = [
                "ffmpeg", "-y",
                "-i", temp_video,
                "-i", audio_path,
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                output_path
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-i", temp_video,
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                output_path
            ]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True)
        except subprocess.CalledProcessError:
            # 如果FFmpeg失败，直接复制临时文件
            import shutil
            shutil.copy(temp_video, output_path)
        
        # 清理临时文件
        if os.path.exists(temp_video):
            os.remove(temp_video)
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
        
        return output_path


class VideoFrameExtractor:
    """视频拆帧节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
            },
            "optional": {
                "frame_skip": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                    "tooltip": "跳帧数 (0=全部帧)"
                }),
                "max_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "最大帧数 (0=无限制)"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "INT")
    RETURN_NAMES = ("frames", "fps", "frame_count")
    FUNCTION = "extract"
    CATEGORY = "None-upup"

    def extract(self, video, frame_skip=0, max_frames=0):
        import cv2
        
        if isinstance(video, str):
            video_path = video
        elif hasattr(video, 'path'):
            video_path = video.path
        else:
            # 已经是tensor
            return (video, 30.0, video.shape[0])
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frames = []
        frame_idx = 0
        skip_counter = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if skip_counter == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                
                if max_frames > 0 and len(frames) >= max_frames:
                    break
            
            skip_counter = (skip_counter + 1) % (frame_skip + 1)
            frame_idx += 1
        
        cap.release()
        
        frames_np = np.stack(frames, axis=0).astype(np.float32) / 255.0
        frames_tensor = torch.from_numpy(frames_np)
        
        return (frames_tensor, fps, len(frames))


class VideoFrameComposer:
    """视频合成节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "fps": ("FLOAT", {
                    "default": 30.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1,
                }),
            },
            "optional": {
                "audio": ("AUDIO",),
                "codec": (["h264", "h265", "vp9"], {"default": "h264"}),
                "quality": ("INT", {
                    "default": 18,
                    "min": 0,
                    "max": 51,
                    "step": 1,
                    "tooltip": "CRF值 (越低质量越高)"
                }),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "compose"
    CATEGORY = "None-upup"

    def compose(self, frames, fps, audio=None, codec="h264", quality=18):
        import cv2
        
        output_dir = folder_paths.get_output_directory()
        output_path = os.path.join(output_dir, f"composed_video_{os.getpid()}.mp4")
        temp_video = os.path.join(tempfile.gettempdir(), "temp_compose.mp4")
        
        N, H, W, C = frames.shape
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(temp_video, fourcc, fps, (W, H))
        
        for i in range(N):
            frame = (frames[i].cpu().numpy() * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
        
        writer.release()
        
        # 编码器映射
        codec_map = {
            "h264": "libx264",
            "h265": "libx265",
            "vp9": "libvpx-vp9"
        }
        
        cmd = [
            "ffmpeg", "-y",
            "-i", temp_video,
            "-c:v", codec_map[codec],
            "-crf", str(quality),
            "-preset", "medium",
        ]
        
        if audio is not None:
            audio_path = self._save_audio(audio)
            if audio_path:
                cmd.extend(["-i", audio_path, "-c:a", "aac", "-b:a", "192k", "-shortest"])
        
        cmd.append(output_path)
        
        try:
            subprocess.run(cmd, capture_output=True, check=True)
        except:
            import shutil
            shutil.copy(temp_video, output_path)
        
        if os.path.exists(temp_video):
            os.remove(temp_video)
        
        return (output_path,)

    def _save_audio(self, audio) -> Optional[str]:
        """保存音频到临时文件"""
        try:
            import torchaudio
            audio_path = os.path.join(tempfile.gettempdir(), "temp_audio_compose.wav")
            torchaudio.save(audio_path, audio["waveform"], audio["sample_rate"])
            return audio_path
        except:
            return None


class GMFSSModelLoader:
    """
    GMFSS 模型加载器
    支持 GMFSS_Fortuna 系列模型 (gmfss / union)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": "",
                    "tooltip": "GMFSS模型路径 (.pkl文件)"
                }),
                "model_type": (["gmfss", "union"], {
                    "default": "union",
                    "tooltip": "模型类型：gmfss基础版 / union增强版"
                }),
            },
            "optional": {
                "scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.25,
                    "max": 2.0,
                    "step": 0.25,
                    "tooltip": "光流计算分辨率缩放 (越小越快，质量降低)"
                }),
            },
        }

    RETURN_TYPES = ("VFI_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "None-upup"

    def load_model(self, model_path: str, model_type: str = "union", scale: float = 1.0):
        """加载GMFSS模型"""
        import importlib.util
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        # 尝试从models目录加载
        if not model_path:
            models_dir = os.path.join(folder_paths.models_dir, "vfi")
            if model_type == "union":
                model_path = os.path.join(models_dir, "GMFSS_union.pkl")
            else:
                model_path = os.path.join(models_dir, "GMFSS.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"GMFSS模型未找到: {model_path}\n请下载模型并放置到 models/vfi/ 目录")
        
        # 加载模型
        model_data = {
            "type": "gmfss",
            "model_type": model_type,
            "model_path": model_path,
            "scale": scale,
            "device": device,
            "dtype": dtype,
            "model": None,  # 延迟加载
        }
        
        print(f"[GMFSSModelLoader] 模型配置完成: {model_type}, scale={scale}")
        
        return (model_data,)


class FrameInterpolator:
    """
    通用补帧节点
    支持 GMFSS / RIFE 模型，或使用线性插值
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "multiplier": (["2x", "4x", "8x"], {
                    "default": "2x",
                    "tooltip": "补帧倍数"
                }),
            },
            "optional": {
                "vfi_model": ("VFI_MODEL", {
                    "tooltip": "视频补帧模型 (来自GMFSSModelLoader)"
                }),
                "fallback_mode": (["linear", "rife"], {
                    "default": "linear",
                    "tooltip": "无模型时的回退模式"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "interpolate"
    CATEGORY = "None-upup"

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self._gmfss_model = None
        self._gmfss_config = None

    def interpolate(self, frames, multiplier, vfi_model=None, fallback_mode="linear"):
        mult = {"2x": 2, "4x": 4, "8x": 8}[multiplier]
        
        if vfi_model is not None:
            if vfi_model.get("type") == "gmfss":
                result = self._gmfss_interpolate(frames, mult, vfi_model)
            else:
                result = self._model_interpolate(frames, mult, vfi_model)
        elif fallback_mode == "rife":
            result = self._rife_interpolate(frames, mult)
        else:
            result = self._linear_interpolate(frames, mult)
        
        return (result,)

    def _gmfss_interpolate(self, frames: torch.Tensor, mult: int, config: dict) -> torch.Tensor:
        """GMFSS光流补帧"""
        N, H, W, C = frames.shape
        scale = config.get("scale", 1.0)
        device = config.get("device", self.device)
        dtype = config.get("dtype", self.dtype)
        model_path = config.get("model_path")
        model_type = config.get("model_type", "union")
        
        # 延迟加载模型
        if self._gmfss_model is None or self._gmfss_config != model_path:
            self._gmfss_model = self._load_gmfss(model_path, model_type, device)
            self._gmfss_config = model_path
        
        if self._gmfss_model is None:
            print("[FrameInterpolator] GMFSS加载失败，回退到线性插值")
            return self._linear_interpolate(frames, mult)
        
        # 转换格式 [N, C, H, W] 并归一化
        frames_gpu = frames.permute(0, 3, 1, 2).to(device, dtype=dtype)
        
        # 确保尺寸可被32整除 (GMFSS要求)
        ph = ((H - 1) // 32 + 1) * 32
        pw = ((W - 1) // 32 + 1) * 32
        padding = (0, pw - W, 0, ph - H)
        frames_padded = F.pad(frames_gpu, padding, mode='replicate')
        
        interpolated = []
        
        for i in range(N - 1):
            f0 = frames_padded[i:i+1]
            f1 = frames_padded[i+1:i+2]
            
            interpolated.append(f0[:, :, :H, :W].cpu())
            
            # 生成中间帧
            for t_idx in range(1, mult):
                timestep = t_idx / mult
                with torch.no_grad():
                    # GMFSS接口: inference(img0, img1, timestep, scale)
                    mid = self._gmfss_model.inference(f0, f1, timestep, scale)
                    mid = mid[:, :, :H, :W]  # 裁剪回原尺寸
                interpolated.append(mid.cpu())
            
            # 定期清理显存
            if i % 10 == 0:
                torch.cuda.empty_cache()
        
        # 添加最后一帧
        interpolated.append(frames_padded[-1:, :, :H, :W].cpu())
        
        # 合并结果
        result = torch.cat(interpolated, dim=0)
        result = result.permute(0, 2, 3, 1).to(torch.float32)
        return torch.clamp(result, 0, 1)

    def _load_gmfss(self, model_path: str, model_type: str, device):
        """加载GMFSS模型"""
        try:
            # 尝试加载本地GMFSS实现
            gmfss_dir = os.path.dirname(model_path)
            
            # 动态导入模型
            if model_type == "union":
                # Union模型
                try:
                    from model.GMFSS_union import Model
                except ImportError:
                    # 尝试从ComfyUI插件加载
                    try:
                        from custom_nodes.ComfyUI_VFI.gmfss_union import Model
                    except ImportError:
                        Model = self._create_gmfss_wrapper()
            else:
                # 基础GMFSS模型
                try:
                    from model.GMFSS import Model
                except ImportError:
                    try:
                        from custom_nodes.ComfyUI_VFI.gmfss import Model
                    except ImportError:
                        Model = self._create_gmfss_wrapper()
            
            if Model is None:
                return None
            
            model = Model()
            model.load_model(model_path, -1)  # -1 = auto select GPU
            model.eval()
            model.device()
            
            print(f"[FrameInterpolator] GMFSS模型加载成功: {model_type}")
            return model
            
        except Exception as e:
            print(f"[FrameInterpolator] GMFSS加载失败: {e}")
            return None

    def _create_gmfss_wrapper(self):
        """创建GMFSS包装器 (当无法导入原始模型时)"""
        # 返回None，将回退到线性插值
        print("[FrameInterpolator] GMFSS模型定义未找到，请安装GMFSS_Fortuna")
        return None

    def _model_interpolate(self, frames: torch.Tensor, mult: int, model_data: dict) -> torch.Tensor:
        """通用模型补帧"""
        model = model_data.get("model")
        if model is None:
            return self._linear_interpolate(frames, mult)
        
        N = frames.shape[0]
        frames_gpu = frames.permute(0, 3, 1, 2).to(self.device, dtype=self.dtype)
        
        interpolated = []
        for i in range(N - 1):
            f0 = frames_gpu[i:i+1]
            f1 = frames_gpu[i+1:i+2]
            
            interpolated.append(f0.cpu())
            for t in range(1, mult):
                with torch.no_grad():
                    mid = model(f0, f1, t / mult)
                interpolated.append(mid.cpu())
        
        interpolated.append(frames_gpu[-1:].cpu())
        
        result = torch.cat(interpolated, dim=0)
        result = result.permute(0, 2, 3, 1).to(torch.float32)
        return torch.clamp(result, 0, 1)

    def _rife_interpolate(self, frames: torch.Tensor, mult: int) -> torch.Tensor:
        """RIFE光流补帧 (回退模式)"""
        try:
            from custom_nodes.ComfyUI_Frame_Interpolation.rife_model import RIFE
            model = RIFE().to(self.device)
            model.eval()
        except:
            print("[FrameInterpolator] RIFE不可用，回退到线性插值")
            return self._linear_interpolate(frames, mult)
        
        N = frames.shape[0]
        frames_gpu = frames.permute(0, 3, 1, 2).to(self.device, dtype=self.dtype)
        
        interpolated = []
        for i in range(N - 1):
            f0 = frames_gpu[i:i+1]
            f1 = frames_gpu[i+1:i+2]
            
            interpolated.append(f0.cpu())
            for t in range(1, mult):
                with torch.no_grad():
                    mid = model(f0, f1, t / mult)
                interpolated.append(mid.cpu())
        
        interpolated.append(frames_gpu[-1:].cpu())
        
        result = torch.cat(interpolated, dim=0)
        result = result.permute(0, 2, 3, 1).to(torch.float32)
        return torch.clamp(result, 0, 1)

    def _linear_interpolate(self, frames: torch.Tensor, mult: int) -> torch.Tensor:
        """线性插值补帧"""
        N = frames.shape[0]
        interpolated = []
        
        for i in range(N - 1):
            f0, f1 = frames[i], frames[i + 1]
            interpolated.append(f0)
            for t in range(1, mult):
                alpha = t / mult
                interpolated.append((1 - alpha) * f0 + alpha * f1)
        
        interpolated.append(frames[-1])
        return torch.stack(interpolated, dim=0)


# ComfyUI节点注册
NODE_CLASS_MAPPINGS = {
    "VideoCinematicProcessor": VideoCinematicProcessor,
    "VideoFrameExtractor": VideoFrameExtractor,
    "VideoFrameComposer": VideoFrameComposer,
    "GMFSSModelLoader": GMFSSModelLoader,
    "FrameInterpolator": FrameInterpolator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoCinematicProcessor": "🎬 Video Cinematic Processor",
    "VideoFrameExtractor": "📽️ Video Frame Extractor",
    "VideoFrameComposer": "🎥 Video Frame Composer",
    "GMFSSModelLoader": "🔄 GMFSS Model Loader",
    "FrameInterpolator": "⏩ Frame Interpolator",
}
