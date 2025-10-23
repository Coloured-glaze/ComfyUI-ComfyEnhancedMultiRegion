import math
import torch
import logging
from typing import Dict, Any
from nodes import MAX_RESOLUTION, ConditioningCombine, ConditioningSetMask
from .attention_couple import AttentionCouple
import comfy.model_management


# 配置日志
logger = logging.getLogger(__name__)


def validate_size(width, height):
    if width % 64 != 0 or height % 64 != 0:
        # 自动对齐到最近合规尺寸
        new_w = (width // 64 + 1) * 64
        new_h = (height // 64 + 1) * 64
        print(f"comfy_couple Adjusted size: {width}x{height} → {new_w}x{new_h}")
        return new_w, new_h
    return width, height

class ComfyMultiRegion:
    """多区域处理节点，提供水平或垂直的区域分割功能"""
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """定义输入类型"""
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "要应用多区域处理的模型"}),
                "negative": ("CONDITIONING", {"tooltip": "负面提示词条件，应用于所有区域"}),
                "orientation": (["horizontal", "vertical"], {"tooltip": "区域分割方向：horizontal=水平分割，vertical=垂直分割"}),
                "num_regions": ("INT", {"default": 2, "min": 2, "max": 10, "step": 1, "tooltip": "要分割的区域数量"}),
                "width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8, "tooltip": "生成图像的宽度（必须是8的倍数）"}),
                "height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8, "tooltip": "生成图像的高度（必须是8的倍数）"}),
                "isolation_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "隔离因子：0=无隔离，1=完全隔离，控制区域间的独立性"}),
            },
            "optional": {
                # 以下代码 js 端已经定义
                # **{f"positive_{i+1}": ("CONDITIONING",) for i in range(10)},
                # **{f"ratio_{i+1}": ("FLOAT", {"default": 0.5, "min": 0, "max": 1.0, "step": 0.01}) for i in range(10)},
                # **{f"weight_{i+1}": ("FLOAT", {"default": 1.0, "min": 0, "max": 10.0, "step": 0.1}) for i in range(10)}
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("model", "positive", "negative",)
    FUNCTION = "process"
    CATEGORY = "loaders"

    def process(self, model, negative, orientation, num_regions, width, height, isolation_factor, **kwargs):
        """处理多区域生成"""
        try:
            # 清理CUDA内存
            comfy.model_management.soft_empty_cache()
            
            positives = [kwargs.get(f"positive_{i+1}") for i in range(num_regions)]
            ratios = [kwargs.get(f"ratio_{i+1}", 1.0 / num_regions) for i in range(num_regions - 1)]
            weights = [kwargs.get(f"weight_{i+1}", 1.0) for i in range(num_regions)]

            if any(pos is None for pos in positives):
                raise ValueError(f"Expected {num_regions} positive conditionings, but some are missing")

            # Normalize ratios
            ratios.append(max(0, 1.0 - sum(ratios)))  # Ensure non-negative
            total = sum(ratios)
            if total <= 0:
                raise ValueError("Ratio total must be positive")
            ratios = [r / total for r in ratios]

            # Create masks for each region
            width, height = validate_size(width, height)
            masks = self.create_masks(ratios, orientation, width, height)

            # Apply masks and weights to positive conditionings
            conditioned_masks = [ConditioningSetMask().append(pos, mask, "default", weight)[0] for pos, mask, weight in zip(positives, masks, weights)]

            # Combine all conditioned masks
            positive_combined = conditioned_masks[0]
            for mask in conditioned_masks[1:]:
                positive_combined = ConditioningCombine().combine(positive_combined, mask)[0]

            return AttentionCouple().attention_couple(model, positive_combined, negative, "Attention", isolation_factor)
        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA内存不足，请尝试减小区域数量或图像尺寸")
            raise
        except Exception as e:
            logger.error(f"多区域处理过程中发生错误: {e}")
            raise

    @staticmethod
    def create_masks(ratios, orientation, width, height):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        masks = torch.zeros((len(ratios), height, width), device=device)
        start = 0

        for i, ratio in enumerate(ratios):
            if orientation == "horizontal":
                region_width = math.floor(width * ratio)
                end = min(start + region_width, width)
                masks[i, :, start:end] = 1.0
            else:  # vertical
                region_height = math.floor(height * ratio)
                end = min(start + region_height, height)
                masks[i, start:end, :] = 1.0
            start = end

        return masks

class ComfyCoupleRegion:
    """单个区域处理节点，用于定义一个带掩码的区域"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """定义输入类型"""
        return {
            "required": {
                "positive": ("CONDITIONING", {"tooltip": "该区域的正面提示词条件"}),
                "mask": ("MASK", {"tooltip": "定义该区域形状的掩码"}),
                "weight": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01, "tooltip": "该区域的权重，影响提示词的强度"},
                ),
            },
        }

    RETURN_TYPES = ("ATTENTION_COUPLE_REGION",)
    RETURN_NAMES = ("region",)
    FUNCTION = "process"
    CATEGORY = "loaders"

    def process(self, positive, mask, weight):
        return ({"positive": positive, "mask": mask, "weight": weight},)

class ComfyCoupleMask:
    """自定义掩码组合节点，用于将多个区域的条件组合起来"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "要应用自定义掩码组合的模型"}),
                "negative": ("CONDITIONING", {"tooltip": "负面提示词条件，应用于所有区域"}),
                "isolation_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "隔离因子：0=无隔离，1=完全隔离，控制区域间的独立性"}),
                "inputcount": ("INT", {"default": 2, "min": 2, "max": 256, "step": 1, "tooltip": "要组合的区域数量"}),
                "region_1": ("ATTENTION_COUPLE_REGION", {"tooltip": "第一个区域定义"}),
                "region_2": ("ATTENTION_COUPLE_REGION", {"tooltip": "第二个区域定义"}),

            }
        }

    RETURN_TYPES = ( "MODEL", "CONDITIONING", "CONDITIONING", )
    RETURN_NAMES = ("model", "positive", "negative")
    FUNCTION = "process"
    CATEGORY = "loaders"

    def process(self, model, inputcount, negative, isolation_factor, **kwargs):

        first_cond = kwargs["region_1"]

        base_mask = torch.full(first_cond["mask"].shape, 1.0, dtype=torch.float32, device="cpu")

        # 使用第一个区域的权重
        first_weight = first_cond.get("weight", 1.0)
        positive_combined = ConditioningSetMask().append(first_cond["positive"], first_cond["mask"], "default", first_weight)[0]

        for c in range(1, inputcount):

            new_cond = kwargs[f"region_{c + 1}"]

            base_mask = base_mask - new_cond["mask"]

            # 使用每个区域的权重
            new_weight = new_cond.get("weight", 1.0)
            conditioning_mask_second = ConditioningSetMask().append(new_cond["positive"], new_cond["mask"], "default", new_weight)[0]

            positive_combined = ConditioningCombine().combine(positive_combined, conditioning_mask_second)[0]

        conditioning_mask_base = ConditioningSetMask().append(kwargs["region_1"]["positive"], base_mask, "default", 1.0)[0]

        positive_combined = ConditioningCombine().combine(positive_combined, conditioning_mask_base)[0]

        return AttentionCouple().attention_couple(model, positive_combined, negative, "Attention", isolation_factor)

    
NODE_CLASS_MAPPINGS = {
    "Attention couple": AttentionCouple,
    "ComfyMultiRegion": ComfyMultiRegion,
    "ComfyCoupleMask": ComfyCoupleMask,
    "ComfyCoupleRegion": ComfyCoupleRegion,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Attention couple": "Comfy Attention couple",
    "ComfyMultiRegion": "Comfy Multi Region",
    "ComfyCoupleMask": "Comfy Couple Mask",
    "ComfyCoupleRegion": "Comfy Couple Region",
}
