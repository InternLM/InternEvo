# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import subprocess
import sys
from enum import Enum, unique

from internevo import train

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
            (
                "srun -p llm_s -N 1 -n 8 --ntasks-per-node=8 --gpus-per-task=1 python {file_name} --config {args}"
            )
            .format(
                file_name=train.__file__,
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