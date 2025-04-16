from internlm.core.context import global_context as gpc
import subprocess
import os

gpc.load_config("./configs/7B_isp_sft.py")
job_name = gpc.config.JOB_NAME
task_name = gpc.config.TASK_NAME

if "MEMORY_PATH" not in gpc.config:
    task_folder = "NONE"
    save_index = False
else:
    task_folder = gpc.config.MEMORY_PATH
    save_index = True


output_path = os.path.join("/mnt/petrelfs/lusitian/workspace/InternEvo-fork/log_output", task_folder, f"{task_name}.out")

PARTITION = "llm_s"
NODES = 4
TOTAL_TASKS = 32
TASKS_PER_NODE = 8
GPUS_PER_TASK = 1
CONFIG_FILE = "./configs/7B_isp_sft.py"


def submit_job():

    # 构建命令列表
    command = [
        "srun",
        "-p", PARTITION,
        # "-x", "HOST-10-140-60-6",
        "-N", str(NODES),
        "-n", str(TOTAL_TASKS),
        "--ntasks-per-node", str(TASKS_PER_NODE),
        "--gpus-per-task", str(GPUS_PER_TASK),
    ]

    # 动态添加异步和输出参数
    if save_index:
        # 确保输出目录存在
        # os.makedirs(os.path.dirname(output_path), exist_ok=True)
        command[1:1] = ["--async", "-o", output_path]  # 在 -p 参数后插入

    # 添加固定尾部参数
    command += [
        "python", "train.py",
        "--config", CONFIG_FILE,
        "--profiling"
    ]

    try: # 
        # 执行命令
        subprocess.run(command, check=True)
        print(f"✅ 作业已提交，日志输出到: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ 提交失败: {e}")
    except Exception as e:
        print(f"❌ 发生意外错误: {str(e)}")
    

if __name__ == "__main__":
    submit_job()


