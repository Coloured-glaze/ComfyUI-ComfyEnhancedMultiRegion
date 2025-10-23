import torch
import torch.nn.functional as F
import copy
import math 
import logging
from typing import List, Tuple, Union

import comfy
import comfy.model_management
from comfy.ldm.modules.attention import optimized_attention

# 配置日志
logger = logging.getLogger(__name__)

def get_masks_from_q(masks: List[Union[torch.Tensor, bool]], q: torch.Tensor, original_shape: Tuple[int, ...]) -> torch.Tensor:
    """从查询张量生成掩码，优化版本"""
    H = original_shape[2]
    W = original_shape[3]
    seq_len = q.shape[1]

    # 计算保持原始宽高比的目标空间维度
    aspect_ratio = H / W
    h_q = int(math.sqrt(seq_len * aspect_ratio))
    w_q = int(seq_len / h_q)

    # 精确调整以确保乘积等于序列长度
    while h_q * w_q < seq_len:
        h_q += 1
        w_q = max(1, int(seq_len / h_q))
    while h_q * w_q > seq_len:
        h_q -= 1
        w_q = max(1, int(seq_len / h_q))

    # 预分配结果列表
    ret_masks = []
    device = q.device
    dtype = q.dtype
    
    # 批量处理掩码
    for mask in masks:
        if isinstance(mask, torch.Tensor):
            size = (h_q, w_q)
            # 使用最近邻插值保持掩码的二元性质
            mask_downsample = F.interpolate(mask.unsqueeze(0), size=size, mode="nearest")
            mask_downsample = mask_downsample.view(1, -1, 1).repeat(q.shape[0], 1, q.shape[2])
            ret_masks.append(mask_downsample.to(device=device, dtype=dtype))
        else:  # 无耦合处理时
            ret_masks.append(torch.ones_like(q))

    # 一次性连接所有掩码
    if ret_masks:
        ret_masks = torch.cat(ret_masks, dim=0)
    else:
        ret_masks = torch.ones_like(q.unsqueeze(0))
    
    return ret_masks

def set_model_patch_replace(model, patch, key) -> None:
    """安全地设置模型补丁替换"""
    try:
        to = model.model_options["transformer_options"]
        if "patches_replace" not in to:
            to["patches_replace"] = {}
        if "attn2" not in to["patches_replace"]:
            to["patches_replace"]["attn2"] = {}
        to["patches_replace"]["attn2"][key] = patch
    except KeyError as e:
        logger.error(f"设置模型补丁时发生键错误: {e}")
        raise

class AttentionCouple:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "要应用注意力耦合的模型"}),
                "positive": ("CONDITIONING", {"tooltip": "正面提示词条件，用于生成图像"}),
                "negative": ("CONDITIONING", {"tooltip": "负面提示词条件，用于避免生成的内容"}),
                "mode": (["Attention", "Latent"], {"tooltip": "耦合模式：Attention=注意力耦合，Latent=潜在空间耦合"}),
                "isolation_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "隔离因子：0=无隔离，1=完全隔离，控制区域间的独立性"}),
            }
        }
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("model", "positive", "negative")
    FUNCTION = "attention_couple"
    CATEGORY = "loaders"

    def attention_couple(self, model, positive, negative, mode, isolation_factor):
        """注意力耦合主方法，添加了错误处理和内存管理"""
        if mode == "Latent":
            return (model, positive, negative)

        # 清理CUDA内存
        comfy.model_management.soft_empty_cache()
            
        self.negative_positive_masks = []
        self.negative_positive_conds = []
        self.isolation_factor = isolation_factor

        # 使用浅拷贝优化性能
        new_positive = copy.copy(positive)
        new_negative = copy.copy(negative)

        dtype = model.model.diffusion_model.dtype
        device = comfy.model_management.get_torch_device()

        # maskとcondをリストに格納する
        for conditions in [new_negative, new_positive]:
            conditions_masks = []
            conditions_conds = []
            if len(conditions) != 1:
                mask_norm = torch.stack([cond[1]["mask"].to(device, dtype=dtype) * cond[1]["mask_strength"] for cond in conditions])
                mask_norm = mask_norm / mask_norm.sum(dim=0)  # 合計が1になるように正規化(他が0の場合mask_strengthの効果がなくなる)
                conditions_masks.extend([mask_norm[i] for i in range(mask_norm.shape[0])])
                conditions_conds.extend([cond[0].to(device, dtype=dtype) for cond in conditions])
                del conditions[0][1]["mask"]  # latent coupleの無効化のため
                del conditions[0][1]["mask_strength"]
            else:
                conditions_masks = [False]
                conditions_conds = [conditions[0][0].to(device, dtype=dtype)]
            self.negative_positive_masks.append(conditions_masks)
            self.negative_positive_conds.append(conditions_conds)
        self.conditioning_length = (len(new_negative), len(new_positive))

        new_model = model.clone()
        self.sdxl = hasattr(new_model.model.diffusion_model, "label_emb")
        if not self.sdxl:
            for id in [1,2,4,5,7,8]:  # id of input_blocks that have cross attention
                set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.input_blocks[id][1].transformer_blocks[0].attn2), ("input", id))
            set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.middle_block[1].transformer_blocks[0].attn2), ("middle", 0))
            for id in [3,4,5,6,7,8,9,10,11]:  # id of output_blocks that have cross attention
                set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.output_blocks[id][1].transformer_blocks[0].attn2), ("output", id))
        else:
            for id in [4,5,7,8]:  # id of input_blocks that have cross attention
                block_indices = range(2) if id in [4, 5] else range(10)  # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.input_blocks[id][1].transformer_blocks[index].attn2), ("input", id, index))
            for index in range(10):
                set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.middle_block[1].transformer_blocks[index].attn2), ("middle", id, index))
            for id in range(6):  # id of output_blocks that have cross attention
                block_indices = range(2) if id in [3, 4, 5] else range(10)  # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.output_blocks[id][1].transformer_blocks[index].attn2), ("output", id, index))

        return (new_model, [new_positive[0]], [new_negative[0]])  # pool outputは・・・後回し

    def make_patch(self, module):           
        """创建注意力补丁，优化版本"""
        def patch(q, k, v, extra_options):
            len_neg, len_pos = self.conditioning_length
            cond_or_uncond = extra_options["cond_or_uncond"]
            q_list = q.chunk(len(cond_or_uncond), dim=0)
            b = q_list[0].shape[0]

            # 批量获取掩码
            with torch.no_grad():
                masks_uncond = get_masks_from_q(self.negative_positive_masks[0], q_list[0], extra_options["original_shape"])
                masks_cond = get_masks_from_q(self.negative_positive_masks[1], q_list[0], extra_options["original_shape"])

            ## Sizes of tensors must match except in dimension 0. Expected size 231 but got size 154 for tensor number 2 in the list.

            # maxi_prompt_size_uncond = max([cond.shape[1] for cond in self.negative_positive_conds[0]])
            # context_uncond = torch.cat([cond.repeat(1, maxi_prompt_size_uncond//cond.shape[1], 1) for cond in self.negative_positive_conds[0]], dim=0)

            # maxi_prompt_size_cond = max([cond.shape[1] for cond in self.negative_positive_conds[1]])
            # context_cond = torch.cat([cond.repeat(1, maxi_prompt_size_cond//cond.shape[1], 1) for cond in self.negative_positive_conds[1]], dim=0)
            
            # 替代方案：使用插值对齐维度
            maxi_prompt_size_uncond = max([cond.shape[1] for cond in self.negative_positive_conds[0]])
            context_uncond = torch.cat([
                F.interpolate(
                    cond.permute(0, 2, 1), 
                    size=maxi_prompt_size_uncond, 
                    mode='linear'
                ).permute(0, 2, 1)
                for cond in self.negative_positive_conds[0]
            ], dim=0)

            maxi_prompt_size_cond = max([cond.shape[1] for cond in self.negative_positive_conds[1]])
            context_cond = torch.cat([
                F.interpolate(
                    cond.permute(0, 2, 1), 
                    size=maxi_prompt_size_cond, 
                    mode='linear'
                ).permute(0, 2, 1)
                for cond in self.negative_positive_conds[1]
            ], dim=0)

            k_uncond = module.to_k(context_uncond)
            k_cond = module.to_k(context_cond)
            v_uncond = module.to_v(context_uncond)
            v_cond = module.to_v(context_cond)

            out = []
            for i, c in enumerate(cond_or_uncond):
                if c == 0:
                    masks = masks_cond
                    k = k_cond
                    v = v_cond
                    length = len_pos
                else:
                    masks = masks_uncond
                    k = k_uncond
                    v = v_uncond
                    length = len_neg

                q_target = q_list[i].repeat(length, 1, 1)
                k = torch.cat([k[i].unsqueeze(0).repeat(b,1,1) for i in range(length)], dim=0)
                v = torch.cat([v[i].unsqueeze(0).repeat(b,1,1) for i in range(length)], dim=0)

                # Convert all tensors to the same dtype as q_target
                k = k.to(dtype=q_target.dtype)
                v = v.to(dtype=q_target.dtype)
                masks = masks.to(dtype=q_target.dtype)

                # Apply sharpened masks based on isolation factor
                sharpened_masks = self.sharpen_masks(masks, self.isolation_factor)
                
                qkv = optimized_attention(q_target, k, v, extra_options["n_heads"])
                qkv = qkv * sharpened_masks
                qkv = qkv.view(length, b, -1, module.heads * module.dim_head).sum(dim=0)

                out.append(qkv)

            out = torch.cat(out, dim=0)
            return out
        return patch

    def sharpen_masks(self, masks, isolation_factor):
        if isolation_factor == 0:
            return masks
            
        # 转换隔离因子为张量
        isolation_factor_tensor = torch.tensor(isolation_factor, device=masks.device, dtype=masks.dtype)
        
        # 增强锐化算法，使效果更明显
        # 使用更大的指数范围和更强的对比度增强
        exponent = 1.0 + isolation_factor_tensor * 9.0  # 范围从1到10
        sharpened = torch.pow(masks, exponent)
        
        # 应用对比度增强
        if isolation_factor > 0.5:
            # 对于高隔离因子，进一步增强对比度
            contrast_factor = (isolation_factor_tensor - 0.5) * 4.0  # 范围从0到2
            mean_val = torch.mean(sharpened, dim=0, keepdim=True)
            sharpened = mean_val + (sharpened - mean_val) * (1.0 + contrast_factor)
        
        # 稳定的归一化，但保留一定的对比度
        sum_sharpened = sharpened.sum(dim=0, keepdim=True)
        
        # 避免除零，同时保持最小值
        sum_sharpened = torch.clamp(sum_sharpened, min=1e-8)
        
        # 归一化，但应用一个保留对比度的因子
        normalized = sharpened / sum_sharpened
        
        # 对于高隔离因子，进一步锐化边界
        if isolation_factor > 0.7:
            # 应用阈值处理来锐化边界
            threshold = 0.2 / isolation_factor_tensor  # 随隔离因子增加而降低的阈值
            normalized = torch.where(normalized > threshold, normalized, normalized * 0.1)
            # 重新归一化
            sum_norm = normalized.sum(dim=0, keepdim=True)
            sum_norm = torch.clamp(sum_norm, min=1e-8)
            normalized = normalized / sum_norm
        
        return normalized