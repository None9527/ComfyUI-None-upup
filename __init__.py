"""
ComfyUI-None-upup: 电影级 AI 画质渲染引擎节点
"""

from .nodes.cinematic_enhancer import CinematicEnhancerNode

NODE_CLASS_MAPPINGS = {
    "CinematicEnhancer": CinematicEnhancerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CinematicEnhancer": " Cinematic Enhancer",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
