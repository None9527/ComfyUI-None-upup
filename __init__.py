"""
ComfyUI-None-upup: 电影级 AI 画质渲染引擎节点
"""

from .nodes.cinematic_enhancer import CinematicEnhancerNode
from .nodes.video_cinematic_processor import (
    VideoCinematicProcessor,
    VideoFrameExtractor,
    VideoFrameComposer,
    FrameInterpolator,
)

NODE_CLASS_MAPPINGS = {
    "CinematicEnhancer": CinematicEnhancerNode,
    "VideoCinematicProcessor": VideoCinematicProcessor,
    "VideoFrameExtractor": VideoFrameExtractor,
    "VideoFrameComposer": VideoFrameComposer,
    "FrameInterpolator": FrameInterpolator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CinematicEnhancer": "🎨 Cinematic Enhancer",
    "VideoCinematicProcessor": "🎬 Video Cinematic Processor",
    "VideoFrameExtractor": "📽️ Video Frame Extractor",
    "VideoFrameComposer": "🎥 Video Frame Composer",
    "FrameInterpolator": "⏩ Frame Interpolator",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

