import os
import sys

host_file = sys.argv[1]
# host_file = "/share/work/sh/work/soft/test/hostfiledo.txt"

with open(host_file, "r") as fp:
    for line in fp:
        if len(line) > 10:
            lists = line.split(" ")
            print(lists[0])
            break

# ip = int(host_file.split('gpuserver-')[1])
# print(ip_map[ip])
