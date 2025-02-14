# Copyright (c) InternLM. All rights reserved.
from .megablock_dmoe import MegaBlockdMoE
from .megablock_moe import MegaBlockMoE
from .mlp import MegaBlockFeedForward, MegaBlockGroupedFeedForward

__all__ = ["MegaBlockdMoE", "MegaBlockMoE", "MegaBlockFeedForward", "MegaBlockGroupedFeedForward"]
