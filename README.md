# ComfyUI-None-upup

电影级 AI 画质渲染引擎 ComfyUI 节点集合

## 节点列表

### 🎨 Cinematic Enhancer
图像画质增强节点，支持边缘感知锐化和亮度区域光感增强。

**参数：**
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| sharpness | FLOAT | 0.5 | 边缘锐化强度 (0-1) |
| luminosity | FLOAT | 0.3 | 光感层次强度 (0-1) |
| shadow_lift | FLOAT | 0.15 | 阴影提亮 |
| highlight_roll | FLOAT | 0.1 | 高光回退 |
| upscale_model | UPSCALE_MODEL | - | 可选超分模型 |

---

### 🎬 Video Cinematic Processor
**一体化视频处理节点** - GPU BF16多线程加速

功能：视频拆帧 → 画质增强 → 补帧 → 合成视频

**参数：**
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| video | VIDEO | - | 输入视频 |
| sharpness | FLOAT | 0.5 | 边缘锐化强度 |
| luminosity | FLOAT | 0.3 | 光感层次强度 |
| frame_interpolation | ENUM | none | 补帧倍数 (none/2x/4x) |
| shadow_lift | FLOAT | 0.15 | 阴影提亮 |
| highlight_roll | FLOAT | 0.1 | 高光回退 |
| batch_size | INT | 4 | GPU批处理大小 |
| num_workers | INT | 4 | 多线程工作数 |

---

### 📽️ Video Frame Extractor
视频拆帧节点

**参数：**
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| video | VIDEO | - | 输入视频 |
| frame_skip | INT | 0 | 跳帧数 (0=全部帧) |
| max_frames | INT | 0 | 最大帧数 (0=无限制) |

**输出：** frames (IMAGE), fps (FLOAT), frame_count (INT)

---

### 🎥 Video Frame Composer
帧序列合成视频节点

**参数：**
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| frames | IMAGE | - | 帧序列 |
| fps | FLOAT | 30.0 | 帧率 |
| audio | AUDIO | - | 可选音频 |
| codec | ENUM | h264 | 编码器 (h264/h265/vp9) |
| quality | INT | 18 | CRF质量 (越低越好) |

---

### ⏩ Frame Interpolator
独立补帧节点 (RIFE光流)

**参数：**
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| frames | IMAGE | - | 帧序列 |
| multiplier | ENUM | 2x | 补帧倍数 (2x/4x/8x) |
| use_rife | BOOL | True | 使用RIFE光流 |

---

## 安装

将此仓库克隆到 ComfyUI 的 `custom_nodes` 目录：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/None9527/ComfyUI-None-upup.git
```

## 依赖

- PyTorch (支持BF16)
- OpenCV (`pip install opencv-python`)
- FFmpeg (系统安装，用于视频处理)
- RIFE模型 (可选，用于光流补帧)

## License

MIT
