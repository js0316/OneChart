import argparse
import torch
import torch.nn.functional as F
import os
from vary.utils.conversation import conv_templates, SeparatorStyle
from vary.utils.utils import disable_torch_init
from transformers import AutoTokenizer
from vary.model.plug.transforms import test_transform
from PIL import Image
import json
import numpy as np
import re
import requests
from io import BytesIO
import time

from vllm import LLM, SamplingParams,ModelRegistry
from vary_opt_math_vllm import VaryVLLMForCausalLM

# 注册模型
ModelRegistry.register_model("OneChartOPTForCausalLM", VaryVLLMForCausalLM)

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'

def load_image(image_bytes):
    # 从bytes加载图像并转换为RGB格式
    if isinstance(image_bytes, bytes):
        try:       
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    # 如果是文件或者下载链接，用于测试
    image_file = image_bytes
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def list_json_value(json_dict):
    rst_str = []
    sort_flag = True
    try:
        for key, value in json_dict.items():
            if isinstance(value, dict):
                decimal_out = list_json_value(value)
                rst_str = rst_str + decimal_out
                sort_flag = False
            elif isinstance(value, list):
                return []
            else:
                if isinstance(value, float) or isinstance(value, int):
                    rst_str.append(value)
                else:
                    value = re.sub(r'\(\d+\)|\[\d+\]', '', value)
                    num_value = re.sub(r'[^\d.-]', '', str(value)) 
                    if num_value not in ["-", "*", "none", "None", ""]:
                        rst_str.append(float(num_value))
    except Exception as e:
        print(f"Error: {e}")
        print(json_dict)
        return []
    return rst_str

def norm_(rst_list):
    if len(rst_list) < 2:
        return rst_list
    min_vals = min(rst_list)
    max_vals = max(rst_list)
    rst_list = np.array(rst_list)
    normalized_tensor = (rst_list - min_vals) / (max_vals - min_vals + 1e-9)
    return list(normalized_tensor)

def eval_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    
    # 初始化vLLM模型
    llm = LLM(
        model=model_name,
        tokenizer=model_name,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        enforce_eager=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side="right")
    image_processor_high = test_transform  # 使用原始的transform
    image_token_len = 256

    # 设置停止条件
    conv_mode = "v1"
    conv = conv_templates[conv_mode].copy()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    keyword_ids = [tokenizer(keyword).input_ids for keyword in keywords]
    new_keyword_ids = [kid[0] for kid in keyword_ids]

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1,
        max_tokens=2048,
        stop_token_ids=new_keyword_ids
    )

    # 处理输入图像列表
    image_list = [
        '/processing_data/search/sunjie/model_zoo/test_chart.jpg',
    ]

    # 加载图像
    images = []
    for image_file in image_list:
        image = load_image(image_file)
        if image is None:
            print(f"Failed to load image: {image_file}")
            continue
        # 确保图像是RGB格式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        images.append(image)

    if not images:
        print("No valid images to process.")
        return

    # 处理图像
    image_tensors = []
    for image in images:
        # 使用原始的test_transform处理图像
        image_tensor = image_processor_high(image)
        
        # 打印调试信息
        print(f"处理后的图像tensor形状: {image_tensor.shape}")
        print(f"处理后的图像tensor类型: {image_tensor.dtype}")
        
        # 确保图像tensor形状正确 [C, H, W]
        if image_tensor.dim() == 3 and image_tensor.size(0) == 3:
            # 转换为bfloat16
            image_tensor = image_tensor.to(torch.bfloat16)
            image_tensors.append(image_tensor)
        else:
            print(f"Unexpected image tensor shape: {image_tensor.shape}")
            continue

    # 构建提示符
    query = "Convert the key information of the chart to a python dict:"
    qs = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_PATCH_TOKEN * image_token_len}{DEFAULT_IM_END_TOKEN}{query}\n"

    # 配置对话模板
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    inputs = tokenizer(prompt)
    input_ids = inputs.input_ids  # 形状 e.g. [309]

    # vLLM批量推理
    prompts = []
    for idx, image_tensor in enumerate(image_tensors):
        # 确保图像tensor维度正确
        if image_tensor.dim() != 3 or image_tensor.size(0) != 3:
            print(f"跳过形状不正确的图像tensor: {image_tensor.shape}")
            continue
            
        # 确保图像tensor在GPU上
        image_tensor = image_tensor.cuda()
        
        # 打印调试信息
        print(f"传递给模型的图像tensor形状: {image_tensor.shape}")
        print(f"传递给模型的图像tensor类型: {image_tensor.dtype}")
        print(f"传递给模型的图像tensor设备: {image_tensor.device}")
            
        prompts.append({
            'prompt_token_ids': input_ids,
            'multi_modal_data': {
                'image': image_tensor  # 使用 'image' 作为键名
            }
        })
        
    if not prompts:
        print("没有有效的prompts可以处理")
        return
        
    print(f"处理 {len(prompts)} 个prompts...")
    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)

    # 处理输出
    for idx, output in enumerate(outputs):
        print(f"\n处理图片 {os.path.basename(image_list[idx])}:")
        output_text = output.outputs[0].text.strip()
        print("Raw output:", output_text)
        
        try:
            outputs_json = json.loads(output_text)
            list_v = list_json_value(outputs_json['values'])
            list_v = [round(x,4) for x in norm_(list_v)]
            gt_nums = torch.tensor(list_v).reshape(1,-1)
            pred_nums = llm.model.pred_locs  # 获取型内部的预测位置信息 (示例)
            print("<Chart>: ", pred_nums[:len(list_v)])
            pred_nums_ = torch.tensor(pred_nums[:len(list_v)]).reshape(1,-1)
            reliable_distence = F.l1_loss(pred_nums_, gt_nums)
            print("reliable_distence: ", reliable_distence)
            if reliable_distence < 0.1:
                print("After OneChart checking, this prediction is reliable.")
            else:
                print("This prediction may have error!")
        except Exception as e:
            print("This prediction may have error!")
            print(e)
        print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--conv-mode", type=str, default='v1')
    args = parser.parse_args()

    start_time = time.time()
    eval_model(args)
    end_time = time.time()
    print(f"Time cost: {end_time - start_time:.3f} seconds") 