# Copyright (c) OpenMMLab. All rights reserved.
from contextlib import contextmanager


@contextmanager
def read_base():
    """Context manager to mark the base config.

    The pure Python-style configuration file allows you to use the import
    syntax. However, it is important to note that you need to import the base
    configuration file within the context of ``read_base``, and import other
    dependencies outside of it.

    You can see more usage of Python-style configuration in the `tutorial`_

    .. _tutorial: https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#a-pure-python-style-configuration-file-beta  # pylint: disable=line-too-long
    """  # noqa: E501
    yield
