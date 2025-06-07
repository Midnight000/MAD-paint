import os

def make_dirs(path):
    """
    递归生成目录
    :param path: 目录路径
    """
    if not os.path.exists(path):
        # 如果目录不存在，则递归生成它的父目录
        make_dirs(os.path.dirname(path))
        # 生成目录
        os.mkdir(path)
