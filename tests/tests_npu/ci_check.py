#!/usr/bin/env python3
"""CI gate check script for slime-ascend project."""

import os
import subprocess
import shlex
from pathlib import Path


def read_files_from_txt(txt_file):
    with open(txt_file, "r") as f:
        return [line.strip() for line in f.readlines()]


def is_examples(file):
    return file.startswith("examples/") or file.startswith("tests/poc/")


def is_markdown(file):
    return file.endswith(".md") or file.endswith(".MD")


def is_image(file):
    return file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg") or file.endswith(".gif") or file.endswith(".svg")


def is_txt(file):
    return file.endswith(".txt")


def is_owners(file):
    return file.startswith("OWNERS")


def is_license(file):
    return file.startswith("LICENSE")


def is_ut(file):
    return file.startswith("tests_npu/ut")


def is_no_suffix(file):
    return os.path.splitext(file)[1] == ''


def is_gitignore(file):
    return file == ".gitignore"


def skip_ci(files, skip_conds):
    for file in files:
        if not any(condition(file) for condition in skip_conds):
            return False
    return True


def choose_skip_ci(raw_txt_file):
    if not os.path.exists(raw_txt_file):
        return False

    file_list = read_files_from_txt(raw_txt_file)
    skip_conds = [
        is_examples,
        is_markdown,
        is_image,
        is_txt,
        is_owners,
        is_license,
        is_no_suffix,
        is_gitignore
    ]

    return skip_ci(file_list, skip_conds)


def filter_exec_ut(raw_txt_file):
    if not os.path.exists(raw_txt_file):
        return False, None

    file_list = read_files_from_txt(raw_txt_file)
    filter_conds = [
        is_ut,
        is_markdown
    ]
    for file in file_list:
        if not any(condition(file) for condition in filter_conds):
            return False, None
    return True, file_list


def acquire_exitcode(command):
    """不使用 shell 的更安全版本（推荐用于处理用户输入）"""
    args = shlex.split(command)
    process = subprocess.Popen(
        args,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output, end='', flush=True)

    return process.wait()


# =============================
# UT test, run with pytest
# =============================

class UTTest:
    def __init__(self):
        # 1. 明确根目录
        current_dir = Path(__file__).absolute().parent
        self.base_dir = current_dir.parent
        # 2. 明确 NPU 测试目录
        self.ut_files = os.path.join(self.base_dir, "tests_npu", "ut")
        # 3. 默认指向全量 UT 路径
        self.test_dir = os.path.join(self.base_dir, "tests_npu")

    def run_ut(self, raw_txt_file=None):
        # 如果有增量文件列表，尝试进入增量模式
        if raw_txt_file is not None and os.path.exists(raw_txt_file):
            filtered_results = filter_exec_ut(raw_txt_file)
            if filtered_results[0]:
                filtered_files = filtered_results[1]
                full_path = [os.path.join(self.base_dir, file) for file in filtered_files]
                exist_ut_files = [file for file in full_path if os.path.exists(file) and file.endswith(".py")]
                if exist_ut_files:
                    # 只有找到存在的 py 文件才替换，否则维持默认的全量路径
                    self.ut_files = " ".join(exist_ut_files)

        # 检查最终路径是否存在，避免 pytest 报错
        if not any(os.path.exists(f) for f in self.ut_files.split()):
            print(f"Error: UT path {self.ut_files} not found.")
            return

        print(f"Running UT: {self.ut_files}")
        command = f"pytest -x --log-cli-level=INFO {self.ut_files}"
        code = acquire_exitcode(command)
        
        if code == 0:
            print("UT test success")
        else:
            print(f"UT failed with exit code {code}")
            raise RuntimeError(f"UT execution failed with code {code}")            
        print(f"DEBUG: Final UT path is {self.ut_files}")
# ===============================================
# ST test, run with sh.
# ===============================================


class STTest:
    def __init__(self):
        current_dir = Path(__file__).absolute().parent
        self.base_dir = current_dir.parent  
        # 你的物理路径在 tests_npu/st 下
        self.test_dir = os.path.join(self.base_dir, 'tests_npu', 'st')
        self.st_shell = os.path.join(self.test_dir, "st_run.sh")

    def run_st(self):
        if not os.path.exists(self.st_shell):
            print(f"Warning: {self.st_shell} not found, skipping system tests")
            return

        rectify_case = f"bash {self.st_shell}"
        rectify_code = acquire_exitcode(rectify_case)
        if rectify_code != 0:
            print("rectify case failed, check it.")
            raise RuntimeError(f"ST execution failed with code {rectify_code}")


def run_tests(raw_txt_file):
    ut = UTTest()
    st = STTest()
    if os.path.exists(raw_txt_file) and filter_exec_ut(raw_txt_file)[0]:
        ut.run_ut(raw_txt_file)
    else:
        ut.run_ut()
        st.run_st()


def main():
    base_dir = Path(__file__).absolute().parents[1]
    raw_txt_file = os.path.join(base_dir, "modify.txt")

    skip_signal = choose_skip_ci(raw_txt_file)
    if skip_signal:
        print("Skipping CI")
    else:
        run_tests(raw_txt_file)


if __name__ == "__main__":
    main()
