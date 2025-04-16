import inspect

# 导入目标对象
from torch_npu import npu_fusion_attention

def get_object_path(obj):
    # 检查对象是否具有 __file__ 属性（适用于模块或类）
    if hasattr(obj, '__file__'):
        print(f"File path of the object: {obj.__file__}")
        return obj.__file__

    # 如果是函数或方法，尝试通过 inspect 获取其所在模块
    if inspect.isfunction(obj) or inspect.ismethod(obj):
        module = inspect.getmodule(obj)
        if module and hasattr(module, '__file__'):
            print(f"Function/Method is defined in module: {module.__name__}")
            print(f"Module file path: {module.__file__}")
            return module.__file__
        else:
            print("Could not determine the file path of the function/method.")
            return None

    # 如果对象没有明确的文件路径信息
    print("The object does not have a clear file path attribute.")
    return None

# 查找 npu_fusion_attention 的文件路径
get_object_path(npu_fusion_attention)