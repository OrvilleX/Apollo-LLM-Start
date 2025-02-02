from transformers import PreTrainedModel, PretrainedConfig

class VisionTowerConfig(PretrainedConfig):
    """
    VisionTower 的配置类，包含图像尺寸、patch 尺寸、hidden_size、帧数等必要信息。
    """
    model_type = "vision_tower"

    def __init__(self, vision_tower_name: str = None, img_size: int = 224, patch_size: int = 16,
                 hidden_size: int = 768, num_frames: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.vision_tower_name = vision_tower_name
        self.image_size = img_size               # 输入图像尺寸（宽或高，通常是相等的）
        self.patch_size = patch_size             # patch 尺寸（用于划分特征图）
        self.hidden_size = hidden_size           # 最终的特征维度
        self.num_frames = num_frames             # 视频帧数

class VisionTower(PreTrainedModel):
    """
    VisionTower 的基类，封装了视觉模型的前向传播逻辑，并提供了特征选择接口。
    """
    config_class = VisionTowerConfig

    def __init__(self, model_name_or_path: str, config: PretrainedConfig, vision_config: PretrainedConfig = None):
        # 调用父类初始化（传入 vision_config 作为配置）
        super().__init__(vision_config)
        self.vision_tower_name = model_name_or_path
        self.vision_config = vision_config
        # 从外部配置中获取特征选择方案
        self.select_layer = getattr(config, "mm_vision_select_layer", None)
        self.select_feature = getattr(config, "mm_vision_select_feature", "patch")
        # 根据 vision_config 计算特征图尺寸（假设宽高相等）
        if vision_config is not None:
            self.W = vision_config.image_size // vision_config.patch_size
            self.H = vision_config.image_size // vision_config.patch_size
        else:
            self.W = self.H = 1
        self.T = 1  # 对于单帧或固定帧数的视频（SiglipVisionTower 中设置为 1）
        self.hidden_size = getattr(vision_config, "hidden_size", 768) if vision_config is not None else 768

    def feature_select(self, image_features):
        """
        根据配置从模型输出中选取所需的特征。
        如果模型返回多个 hidden_states，可通过 select_layer 进行选择。
        若 select_feature 为 "patch"，则去掉 cls token（假设第一 token 为 cls）。
        """
        if self.select_layer is not None and isinstance(image_features, dict) and "hidden_states" in image_features:
            image_features = image_features["hidden_states"][self.select_layer]
            
        if self.select_feature == "patch":
            # 去除 cls token（假设 token 0 为 cls）
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            # 保留所有 token
            pass
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
            
        return image_features

    def vision_tower_forward(self, image):
        """
        包装模型的前向传播接口。
        要求传入的 image 已经预处理为 tensor，传递至 vision_tower 模型。
        """
        # 输出 hidden_states 用于后续抽取特征
        return self.vision_tower(image, output_hidden_states=True)

    def _forward(self, images):
        """
        简化的前向传播：假设 images 为 4D tensor (B, C, H, W)。
        将经过 vision_tower_forward 后选取特征并 reshape 为 (B, T, W, H, hidden_size)。
        """
        # 将输入移动到模型 device 上，并转换为合适的数据类型
        image_features = self.vision_tower_forward(images.to(self.device, dtype=self.dtype))
        image_features = self.feature_select(image_features)
        B = images.shape[0]
        # 假设 T = 1，此处 reshape 为 (B, 1, W, H, hidden_size)
        features = image_features.reshape(B, self.T, self.W, self.H, self.hidden_size)
        return features

    def forward(self, images):
        return self._forward(images)

    @property
    def dtype(self):
        # 返回模型使用的数据类型
        return self.vision_tower.dtype

    @property
    def device(self):
        # 返回模型所在的 device
        return self.vision_tower.device 