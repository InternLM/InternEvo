file_name = "/data/wikipedia/en_test/train/data.jsonl"

with open(file_name, "r") as f:
    for i in range(2):
        line = f.readline()
        print(line, end='')