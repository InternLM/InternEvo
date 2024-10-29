import os
import sys

INFO=False

host_file = sys.argv[1]
host_name = sys.argv[2]
# host_file = "/share/work/sh/work/soft/test/hostfiledo.txt"
# host_name = "10.77.3.27"

with open(host_file, "r") as fp:
    rank = 0
    for line in fp:
        if len(line) > 10:
            lists = line.split(" ")
            if lists[0].replace(".", "-") == host_name:
                print(rank)
                break
            if INFO:
                print(lists[0], rank)
            rank += 1

# ip = int(host_file.split('gpuserver-')[1])
# print(ip_map[ip])
