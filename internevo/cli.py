import subprocess
import sys
from enum import Enum, unique

from internevo import launcher

USAGE = (
    "-" * 70
    + "\n"
    + "| Usage:                                                             |\n"
    + "|   internevo-cli train -h: train models                             |\n"
    + "|   internevo-cli version: show version info                         |\n"
    + "-" * 70
)


@unique
class Command(str, Enum):
    TRAIN = "train"
    HELP = "help"


def main():
    command = sys.argv.pop(1) if len(sys.argv) != 1 else Command.HELP
    if command == Command.TRAIN:
        process = subprocess.run(
            ("srun -p llm_s -N 1 -n 8 --ntasks-per-node=8 --gpus-per-task=1 python {file_name} --config {args}")
            .format(
                file_name=launcher.__file__,
                args=" ".join(sys.argv[1:]),
            )
            .split()
        )
        sys.exit(process.returncode)
    elif command == Command.HELP:
        print(USAGE)
    else:
        raise NotImplementedError(f"Unknown command: {command}.")


if __name__ == "__main__":
    main()
