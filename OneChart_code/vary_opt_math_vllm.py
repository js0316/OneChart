from typing import List, Optional, Tuple, Union, Iterable, TypeVar, Protocol, ClassVar, Literal, Set
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import PretrainedConfig
from vllm.config import CacheConfig, MultiModalConfig, LoRAConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.model_executor.models.opt import OPTForCausalLM, OPTModel
from vllm.model_executor.models.interfaces import SupportsVision
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.sequence import IntermediateTensors, SamplerOutput
from vllm.inputs.registry import InputContext
from vllm.multimodal.base import MultiModalInputs
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.utils import is_pp_missing_parameter

from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from functools import partial

from vary.model.vision_encoder.sam import build_sam_vit_b
from vary.model.plug.transforms import test_transform

class SupportsMultiModal(Protocol):
    supports_multimodal: ClassVar[Literal[True]] = True
    def __init__(self, *, multimodal_config: "MultiModalConfig") -> None:
        ...

_T = TypeVar("_T")
MultiModalData = Union[_T, List[_T]]

class VaryConfig(PretrainedConfig):
    model_type = "vary"

class VaryVLLMModel(OPTModel):
    config_class = VaryConfig
    
    def __init__(self, config: PretrainedConfig, cache_config, quant_config):
        super().__init__(config, cache_config, quant_config)
        
        self.vision_tower = build_sam_vit_b()
        self.mm_projector = nn.Linear(1024, 768)

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """将输入token ID转换为嵌入向量"""
        return self.decoder.embed_tokens(input_ids)

    def initialize_vision_modules(
        self, 
        vision_tower,
        pretrained_stage1_model=None,
        freeze_vision_tower=False,
        use_im_start_end=True,
        vision_select_layer=-1,
        dtype=torch.float16,
        device="cuda"
    ):
        image_processor_high = test_transform
        
        self.vision_tower = self.vision_tower.to(dtype=dtype, device=device)
        self.mm_projector = self.mm_projector.to(dtype=dtype, device=device)

        image_token_len = 256

        self.config.vision_tower = vision_tower
        self.config.image_token_len = image_token_len
        self.config.use_im_start_end = use_im_start_end
        self.config.vision_select_layer = vision_select_layer
        self.config.freeze_vision_tower = freeze_vision_tower
        
        return dict(
            image_processor_high=image_processor_high,
            image_token_len=image_token_len,
        )

    def merge_embeddings(self, input_ids, inputs_embeds, images):
        im_patch_token = 50265
        im_start_token = 50266
        im_end_token = 50267

        image_features = []
        for image in images:
            # 打印调试信息
            print(f"merge_embeddings 接收到的图像形状: {image.shape}")
            print(f"merge_embeddings 接收到的图像类型: {image.dtype}")
            print(f"merge_embeddings 接收到的图像设备: {image.device}")
            
            # 确保图像是正确的格式和设备
            if not isinstance(image, torch.Tensor):
                raise ValueError(f"图像必须是 torch.Tensor，而不是 {type(image)}")
            
            # 确保图像是 [C, H, W] 格式
            if image.dim() == 3 and image.size(0) == 3:
                # 添加 batch 维度
                image = image.unsqueeze(0)  # [1, C, H, W]
            else:
                raise ValueError(f"图像必须是 [C, H, W] 格式，而不是 {image.shape}")
            
            # 打印处理后的形状
            print(f"处理后的图像形状: {image.shape}")
            
            with torch.set_grad_enabled(False):
                cnn_feature = self.vision_tower(image)  # [B, 1024, H', W']
                cnn_feature = cnn_feature.flatten(2).permute(0, 2, 1)  # [B, N, 1024]
                
            image_feature = self.mm_projector(cnn_feature)  # [B, N, 768]
            image_features.append(image_feature.squeeze(0))  # 移除 batch 维度
        
        dummy_image_features = torch.zeros(256, 1024, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        use_im_start_end = True
        new_input_embeds = []
        segment_lengths = []
        
        # 修改处理batch维度的逻辑
        if inputs_embeds.dim() == 2:
            batch_size = len(image_features)
            input_ids = input_ids.view(batch_size, -1)  # 重塑input_ids为[batch_size, seq_len]
            NB, D = inputs_embeds.shape
            inputs_embeds = inputs_embeds.view(batch_size, NB//batch_size, D)  # 重塑为[batch_size, seq_len, hidden_dim]
        
        for cur_input_ids, cur_input_embeds, cur_image_features in zip(input_ids, inputs_embeds, image_features):
            if (cur_input_ids == im_patch_token).sum() == 0:
                cur_input_embeds = cur_input_embeds + (0. * dummy_image_features).sum()
                new_input_embeds.append(cur_input_embeds)
                segment_lengths.append(len(cur_input_embeds))
                continue

            if use_im_start_end:
                if (cur_input_ids == im_start_token).sum() != (cur_input_ids == im_end_token).sum():
                    raise ValueError("The number of image start tokens and image end tokens should be the same.")

                image_start_tokens = torch.where(cur_input_ids == im_start_token)[0]
                for image_start_token_pos, per_cur_image_features in zip(image_start_tokens, cur_image_features):
                    per_cur_image_features = per_cur_image_features.to(device=cur_input_embeds.device)
                    num_patches = per_cur_image_features.shape[0]

                    if cur_input_ids[image_start_token_pos + num_patches + 1] != im_end_token:
                        raise ValueError("The image end token should follow the image start token.")

                    cur_input_embeds = torch.cat(
                        (
                            cur_input_embeds[:image_start_token_pos + 1],
                            per_cur_image_features,
                            cur_input_embeds[image_start_token_pos + num_patches + 1:]
                        ),
                        dim=0
                    )

                new_input_embeds.append(cur_input_embeds)
                segment_lengths.append(len(cur_input_embeds))
            else:
                raise NotImplementedError
        
        # 使用pad_sequence确保每个样本独立且长度一致
        max_length = max(segment_lengths)
        padded_embeds = []
        
        for embed, length in zip(new_input_embeds, segment_lengths):
            # 如果需要，对每个样本进行padding
            if length < max_length:
                padding = torch.zeros(max_length - length, embed.size(-1), 
                                    device=embed.device, dtype=embed.dtype)
                padded_embed = torch.cat([embed, padding], dim=0)
            else:
                padded_embed = embed
            padded_embeds.append(padded_embed)
        
        # 将所有batch的样本连接成一个大的2D张量
        inputs_embeds = torch.cat(padded_embeds, dim=0)  # [(batch_size * max_length), hidden_dim]
        
        # 返回展平的embeds和每个样本的长度信息
        return inputs_embeds, segment_lengths

        
    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        kv_caches: List[Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs: object,
    ) -> Tensor:
        images = kwargs.pop("images", None)
        
        inputs_embeds = self.embed_tokens(input_ids).cuda()

        vision_tower = getattr(self, 'vision_tower', None)
        if vision_tower is not None and images is not None:
            inputs_embeds, segment_lengths = self.merge_embeddings(input_ids, inputs_embeds, images)
            # 更新attention_metadata以反映新的序列长度
            attn_metadata = self._update_attention_metadata(attn_metadata, segment_lengths)
        
        # 确保hidden_states是2D的
        if inputs_embeds.dim() > 2:
            batch_size, seq_len, hidden_size = inputs_embeds.shape
            inputs_embeds = inputs_embeds.reshape(-1, hidden_size)
        
        hidden_states = inputs_embeds

        for i in range(len(self.decoder.layers)):
            layer = self.decoder.layers[i]
            hidden_states = layer(
                hidden_states,
                kv_caches[i],
                attn_metadata
            )
        hidden_states = self.decoder.final_layer_norm(hidden_states)
        
        return hidden_states

    def _update_attention_metadata(self, attn_metadata: AttentionMetadata, segment_lengths: List[int]) -> AttentionMetadata:
        """更新attention_metadata以处理不同长度的序列"""
        # 更新attention_metadata中的相关属性
        # 需要根据vLLM的具体实现来调整
        # 例如更新seq_lens, prompt_lens等
        
        # 这里需要确保attention mask能正确处理padding部分
        # 并且每个样本的attention不会跨越到其他样本
        
        return attn_metadata

def get_max_vary_mm_tokens(ctx: InputContext, data_type_key: str) -> int:
    return 256

get_max_vary_image_tokens = partial(get_max_vary_mm_tokens, data_type_key="image")

def mm_input_mapper_for_vary(
        ctx: InputContext,
        data: MultiModalData[object],
    ) -> MultiModalInputs:
    batch_data = {"image": data}
    return MultiModalInputs(batch_data)

@MULTIMODAL_REGISTRY.register_image_input_mapper(mm_input_mapper_for_vary)
@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_vary_image_tokens)
class VaryVLLMForCausalLM(OPTForCausalLM, SupportsMultiModal):
    supports_multimodal = True
    
    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        multimodal_config: Optional[MultiModalConfig] = None,
    ):
        super().__init__(config, cache_config, quant_config)
        self.model = VaryVLLMModel(config, cache_config, quant_config)
        
        # 将num_decoder移到顶层
        self.num_decoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 256),
        )
        
        if config.tie_word_embeddings:
            self.lm_head = self.model.decoder.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.word_embed_proj_dim
            )
        
        self.pred_locs = []
        print("VaryVLLMForCausalLM initialized with multimodal support")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids, positions, kv_caches, attn_metadata, intermediate_tensors, **kwargs
        )
        return hidden_states
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
        return logits
    
    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens
    
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        
        # 获取模型所有参数
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        all_param_names = set(params_dict.keys())
        loaded_params: Set[str] = set()
        
        print(f"模型总参数数量: {len(all_param_names)}")
        
        for name, loaded_weight in weights:
            original_name = name
            
            if "lm_head.weight" in name and self.config.tie_word_embeddings:
                print(f"跳过tied参数: {name}")
                continue
                
            if name.startswith("decoder."):
                name = "model." + name
                print(f"重映射decoder前缀: {original_name} -> {name}")

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    print(f"跳过GPTQ额外bias: {name}")
                    continue
                if is_pp_missing_parameter(name, self):
                    print(f"跳过缺失参数: {name}")
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                print(f"已加载堆叠参数: {original_name} -> {name}")
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    print(f"跳过GPTQ额外bias: {name}")
                    continue
                if is_pp_missing_parameter(name, self):
                    print(f"跳过缺失参数: {name}")
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                print(f"已加载参数: {original_name} -> {name}")
                
            loaded_params.add(name)
        
        # 找出未加载的参数
        unloaded_params = all_param_names - loaded_params
        print(f"\n加载统计:")
        print(f"已加载参数数量: {len(loaded_params)}")
        print(f"未加载参数数量: {len(unloaded_params)}")
        
        if unloaded_params:
            print("\n未加载的参数列表:")
            for param_name in sorted(unloaded_params):
                print(f"- {param_name}")
                
        return loaded_params