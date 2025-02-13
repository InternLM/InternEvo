import subprocess
import sys
from enum import Enum, unique
from .env import VERSION


HELP_MSG = (
    "-" * 70
    + "\n"
    + "| Usage:                                                             |\n"
    + "|   internevo-cli train: train models                                |\n"
    + "|   internevo-cli version: show version info                         |\n"
    + "|   internevo-cli help: show help info                               |\n"
    + "-" * 70
)


WELCOME = (
    "-" * 58
    + "\n"
    + f"| Welcome to InternEvo, version {VERSION}"
    + " " * (25 - len(VERSION))
    + "|\n|"
    + " " * 56
    + "|\n"
    + "| Project page: https://github.com/InternLM/InternEvo    |\n"
    + "-" * 58
)


@unique
class Command(str, Enum):
    TRAIN = "train"
    VERSION = "version"
    HELP = "help"


def main():
    command = sys.argv.pop(1) if len(sys.argv) != 1 else Command.HELP
    if command == Command.TRAIN:
        from internevo import launcher
        process = subprocess.run(  # noqa # pylint: disable=W1510
            ("srun -p llm_s -N 1 -n 8 --ntasks-per-node=8 --gpus-per-task=1 python {file_name} --config {args}")
            .format(
                file_name=launcher.__file__,
                args=" ".join(sys.argv[1:]),
            )
            .split()
        )
        sys.exit(process.returncode)
    elif command == Command.VERSION:
        print(WELCOME)
    elif command == Command.HELP:
        print(HELP_MSG)
    else:
        raise NotImplementedError(f"Unknown command: {command}.")


if __name__ == "__main__":
    main()
