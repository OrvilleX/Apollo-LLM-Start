# 导入必要的库
import torch
import sys
# 将apollo目录添加到系统路径中
sys.path.append('./apollo')
from transformers import AutoModelForCausalLM
from apollo.mm_utils import (
    KeywordsStoppingCriteria,  # 用于停止生成的关键词判断
    tokenizer_mm_token,        # 多模态token处理工具
    ApolloMMLoader            # 视频/图像数据加载器
)
from apollo.conversations import conv_templates, SeparatorStyle  # 对话模板和分隔符样式
from apollo.constants import X_TOKEN, X_TOKEN_INDEX  # 特殊token常量

# 模型参数配置
version = "qwen_2"  # 使用的对话模板版本
model_path = "/root/autodl-tmp/Apollo-LMMs-Apollo-7B-t32"  # 模型路径

# 输入参数
video_path = "904_1737365120.mp4"  # 视频文件路径
question = "Describe this video in detail"  # 用户提问
temperature = 0.4  # 采样温度，控制生成多样性
top_p = 0.7  # 核采样参数
max_output_tokens = 256  # 最大输出token数

# 设备配置
device = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择GPU或CPU
attn_implementation = "sdpa" if torch.__version__ > "2.1.2" else "eager"  # 注意力机制实现方式

# 加载预训练模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,  # 信任远程代码
    low_cpu_mem_usage=True,  # 减少CPU内存占用
    attn_implementation=attn_implementation  # 设置注意力实现方式
).to(device=device, dtype=torch.bfloat16)  # 将模型移动到指定设备并使用bfloat16精度

# 保存模型的关键组件
tokenizer = model.tokenizer  # 分词器
vision_processors = model.vision_tower.vision_processor  # 视觉处理器
config = model.config  # 模型配置

# 从配置中获取参数
max_length = config.llm_cfg['model_max_length']  # 模型最大长度
num_repeat_token = config.mm_connector_cfg['num_output_tokens']  # 多模态输出token数
mm_use_im_start_end = config.use_mm_start_end  # 是否使用特殊起止token

# 视频处理参数
frames_per_clip = 4  # 每个clip的帧数
clip_duration = getattr(config, 'clip_duration')  # 每个clip的持续时间

# 初始化多模态处理器
mm_processor = ApolloMMLoader(
    vision_processors,
    clip_duration,
    frames_per_clip,
    clip_sampling_ratio=0.65,  # clip采样比例
    model_max_length=config.model_max_length,  # 模型最大长度
    device=device,  # 设备
    num_repeat_token=num_repeat_token  # 重复token数
)

# 设置模型为评估模式
model.eval()

# 加载并处理视频数据
mm_data, replace_string = mm_processor.load_video(video_path)
message = replace_string + "\n\n" + question  # 构建完整输入信息

# 准备对话模板
conv = conv_templates[version].copy()  # 复制对话模板
conv.append_message(conv.roles[0], message)  # 添加用户消息
conv.append_message(conv.roles[1], None)  # 预留助手回复位置
prompt = conv.get_prompt()  # 获取完整prompt

# 将输入转换为模型可接受的格式
input_ids = tokenizer_mm_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0).to(device)

# 设置生成参数
pad_token_ids = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2  # 停止字符串
keywords = [stop_str]
stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)  # 停止条件

# 模型推理
with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        vision_input=[mm_data],  # 视频输入
        data_types=['video'],    # 数据类型
        do_sample=(temperature > 0),  # 是否采样
        temperature=temperature,  # 采样温度
        max_new_tokens=max_output_tokens,  # 最大新token数
        top_p=top_p,  # 核采样参数
        use_cache=True,  # 使用缓存
        num_beams=1,  # beam search参数
        stopping_criteria=[stopping_criteria]  # 停止条件
    )

# 解码输出
pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
print(pred)  # 打印最终结果