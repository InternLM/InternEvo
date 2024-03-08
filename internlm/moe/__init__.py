from .gshard_moe import GShardMOELayer

__all__ = ["GShardMOELayer"]

try:
    from megablocks import ops  # noqa # pylint: disable=W0611
except ModuleNotFoundError:
    pass
else:
    from internlm.moe.megablock.megablock_moe import (  # noqa # pylint: disable=W0611
        MegaBlockMoE,
    )

    __all__ += "MegaBlockMoE"

try:
    import stk  # noqa # pylint: disable=W0611
    from megablocks import ops  # noqa # pylint: disable=W0611
except ModuleNotFoundError:
    pass
else:
    from internlm.moe.megablock.megablock_dmoe import (  # noqa # pylint: disable=W0611
        MegaBlockdMoE,
    )

    __all__ += "MegaBlockdMoE"
