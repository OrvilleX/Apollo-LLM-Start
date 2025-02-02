# Siglip的使用

本仓库主要汇聚针对Siglip的官方目前最新的模型的使用方式包括但不限定以下几种的使用方式
* 根据官方的使用方式实现基于画面帧特征与用户描述特征的对齐（就是用于识别当前对帧的描述与实际画面中的内容是否一致）
* 根据提供的画面帧实现全局天调特征的提取，从而作为LLM的输入从而实现多模态大模型的能力（就是将图片/视频帧识别的结果作为特征作为大模型的上下文从而扩展其能力）
* 

> 需要的注意的是上述通过识别画面以及视频帧的正常的做法都是需要采用多个类似的模型提取特征并融合后传入到llm中，来提高对特征深度的分析提取

## Siglip Vision Extractor 脚本使用说明

本仓库中新增了 `siglip_vision_extractor.py` 文件，该脚本用于独立提取图片/视频帧的视觉特征，
并进一步展示如何将这些特征传入其他大语言模型（例如基于 langchain 实现的 qwen 或 deepseek）。

### 功能概述

- **视觉特征提取**: 利用预训练的 Siglip 模型对图像进行特征提取，其输出张量形状为 (B, T, W, H, hidden_size)。
- **特征图可视化**: 脚本中使用 matplotlib 将特征隐藏维度的平均值展示为热力图。
- **与 LLM 结合**: 提供示例函数 `send_features_to_llm` 展示如何将提取的视觉特征转换为提示信息传入大语言模型。

### 使用方法

1. **依赖安装**: 确保安装以下依赖：
   - torch
   - transformers
   - pillow (PIL)
   - opencv-python
   - matplotlib (可选，用于可视化)

2. **集成使用**:
   `siglip_vision_extractor.py` 文件并非设计为独立运行的脚本，而是提供了一个用于提取视觉特征的类 `SiglipVisionTower`，
   供用户在多模态应用中调用。你可以通过直接导入模块并实例化该类来集成使用：

   ```python
   from siglip.siglip_vision_extractor import SiglipVisionTower, VisionTowerConfig

   # 设置模型路径与配置参数（请替换为实际模型名称或路径）
   model_name_or_path = "siglip-base"
   config = VisionTowerConfig(
       vision_tower_name=model_name_or_path,
       img_size=224,       
       patch_size=16,      
       hidden_size=768,    
       num_frames=1        
   )

   # 初始化 SiglipVisionTower 模型
   model = SiglipVisionTower(model_name_or_path, config)

   # 加载图像并提取视觉特征
   from PIL import Image
   img = Image.open("example.jpg").convert("RGB")  # 请确保存在示例图像
   inputs = model.vision_processor(img, return_tensors="pt")
   features = model(inputs["pixel_values"])
   print("Extracted features shape:", features.shape)
   ```

   你可以在此基础上进一步设计降维或线性映射模块，将提取的高维视觉特征转换成适合大语言模型的输入格式，并结合其他 LLM（例如基于 langchain 的 qwen 或 deepseek）使用。

### 注意事项

- `siglip_vision_extractor.py` 文件并非作为独立运行的脚本，而是作为一个视觉特征提取模块提供相关类（如 `SiglipVisionTower`）。
- 请将该模块集成到实际的多模态工作流中，并根据任务需要设计合适的特征映射模块，以便将视觉特征传入其他大语言模型中使用。

## 