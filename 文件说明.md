# ComfyUI-None-upup 清晰度增强节点

## 节点:  Cinematic Enhancer

## 核心算法
| 参数 | 算法 | 效果 |
|------|------|------|
| clarity | USM锐化 | 边缘清晰度 |
| contrast | CLAHE | 局部光影层次 |
| detail_boost | 高反差保留 | 纹理细节 |

## 推荐值
- 人像: clarity=0.4, contrast=0.5, detail=0.2
- 风景: clarity=0.6, contrast=0.6, detail=0.4
