#!/bin/bash  
  
# 使用ps、grep和awk找到所有由root用户运行且包含"torch"的行，并提取PID  
PIDS=$(ps -ef | awk '/[t]orch/ && $1=="root" {print $2}')  
HOST_NAME=$(hostname)
# 检查是否找到了PID  
if [ -n "$PIDS" ]; then  
    # 对每个PID执行kill命令  
    for PID in $PIDS; do  
        kill "$PID"  
        # 可选：检查kill命令是否成功  
        if [ $? -eq 0 ]; then  
            echo "${HOST_NAME}torch进程 $PID 已被杀死。"  
        else  
            echo "${HOST_NAME}无法杀死torch进程 $PID。"  
        fi  
    done  
else  
    echo "${HOST_NAME}没有找到由root用户运行且包含'torch'的进程。"  
fi

# # 使用ps、grep和awk找到所有由root用户运行且包含"torch"的行，并提取PID  
# PIDS=$(ps -ef | grep '/share/work' | grep '^root' | awk '{print $2}')  
# # 检查是否找到了PID  
# if [ -n "$PIDS" ]; then  
#     # 对每个PID执行kill命令  
#     for PID in $PIDS; do  
#         kill "$PID"  
#         # 可选：检查kill命令是否成功  
#         if [ $? -eq 0 ]; then  
#             echo "${HOST_NAME} /share/work进程 $PID 已被杀死。"  
#         else  
#             echo "${HOST_NAME}无法杀死/share/work进程 $PID。"  
#         fi  
#     done  
# else  
#     echo "${HOST_NAME}没有找到由root用户运行且包含'/share/work'的进程。"  
# fi
