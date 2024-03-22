FlashCrossAttention, FlashSelfAttention = None, None
try:
    from flash_attn.modules.mha import FlashCrossAttention as FCA
    from flash_attn.modules.mha import FlashSelfAttention as FSA
except (ModuleNotFoundError, ImportError):
    print("python env don't have flash attention!")
else:
    global _FlashCrossAttention, _FlashSelfAttention
    FlashCrossAttention, FlashSelfAttention = FCA, FSA
