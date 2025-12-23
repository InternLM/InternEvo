import numpy as np
file_name = "/mnt/shared-storage-user/lusitian/data/data_jsonl/github/tokenized_llama2/train_folder/data/output.bin.meta"

with open(file_name, "rb") as f:
    meta = np.load(f)
    lengths = meta[:,1]
    lengths_total = sum(lengths)
    print("Total tokens:", lengths_total)
    
class test:
    
    def __init__(self):
        self.a = 1
        
    def print_a(self):
        return self.a
        
        
t = test()
t.a += 1
print(t.a, t.print_a())