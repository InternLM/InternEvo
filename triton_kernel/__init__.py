from .int8_col_qunantize import quantize_columnwise_and_transpose
from .int8_row_qunantize import quantize_rowwise

__all__ = [
    "quantize_columnwise_and_transpose",
    "quantize_rowwise",
]
