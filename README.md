---
license: apache-2.0
---
<div align="center">

# Apollo: 视频理解与多模态模型探索

<p align="center">
    <img src="assets/icon.jpg" width="150" style="margin-bottom: 0.2;"/>
<p>

</div>

## 项目简介

Apollo 是一个用于视频理解和多模态任务探索的项目，旨在通过结合先进的视觉模型与大语言模型，实现从视频中提取关键信息、进行视频内容对齐以及生成多模态回答。项目采用模块化设计，各组件均可独立使用，支持与多种开源 LLM（如 qwen、deepseek 等）结合，拓展视频与语言的交互能力。

## 项目结构

- **core/**: 存放项目中的基础类和工具模块，例如用于视觉特征提取的 `VisionTower` 类。
- **siglip/**: 提供 Siglip 模块的实现与使用说明，请参阅 [siglip/ReadMe.md](siglip/ReadMe.md) 了解具体用法。
- **demo.ipynb**: 一个演示 Notebook，展示了如何加载视频、提取视觉特征，并利用这些特征与大语言模型进行多模态推理，从而实现视频理解和问答。
- **其他组件**: 包括视频预处理、LLM 集成模块等，便于扩展到更多多模态应用场景。

## 快速开始

1. **环境安装**

   请首先安装项目依赖：
   ```bash
   pip install -e .
   pip install -r siglip/requirements.txt
   pip install flash-attn --no-build-isolation
   ```

2. **运行 Demo Notebook**

   打开 `demo.ipynb`，按 Notebook 内的说明执行示例。Notebook 展示了完整的流程，包括：
   - 视频加载及预处理；
   - 利用 Siglip 模块提取视频帧的视觉特征；
   - 将视觉特征传入大语言模型进行多模态推理；
   - 输出视频描述与问答结果。

3. **Siglip 模块使用**

   `siglip/ReadMe.md` 文件中详细介绍了 Siglip 模块的使用方法。模块中的 `SiglipVisionTower` 类用于提取图像/视频帧的视觉特征，可以直接导入后集成到多模态推理管道中：
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
   img = Image.open("example.jpg").convert("RGB")  # 请确保示例图像存在
   inputs = model.vision_processor(img, return_tensors="pt")
   features = model(inputs["pixel_values"])
   print("Extracted features shape:", features.shape)
   ```

## 项目用途

Apollo 主要面向以下应用场景：

- **视频理解**: 从视频中自动提取关键信息，实现视频内容描述、时间推理以及情节分析。
- **多模态对话**: 利用视频视觉特征与文本信息结合，支持视频问答、多轮对话等多模态任务。
- **上下文增强**: 将视频视觉特征作为上下文输入至大语言模型，从而增强其认知和回答能力。

## 更多信息

请参阅各模块下的 README 文档以及 `demo.ipynb` 文件以获得更多详细使用说明和示例。

## 发布信息
- **[2024年12月13日]** Apollo 正式发布！
- **[即将推出...]** 训练代码将在获得内部批准后发布。

## 引用

如果您在研究中使用了 Apollo，请引用：
```bibtex
@article{apollo,
    title={Apollo: An Exploration of Video Understanding in Large Multimodal Models},
    author={Orr Zohar, Xiaohan Wang, Yann Dubois, Nikhil Mehta, Tong Xiao, Philippe Hansen-Estruch, Licheng Yu, Xiaofang Wang, Felix Juefei-Xu, Ning Zhang, Serena Yeung-Levy, and Xide Xia},
    journal={arXiv preprint arXiv:2412.10360},
    year={2024}
}
```