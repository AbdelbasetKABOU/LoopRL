import os
import subprocess
import sys
from ctypes import RTLD_GLOBAL, PyDLL, c_char_p
from typing import Generic, TypeVar

AUTO_BUILD = False

dll_name = "pywrap.so"

if AUTO_BUILD:
    p = subprocess.run(
        ["dune", "build", "looprl/bin/pywrap.so"],
        capture_output=True)
    if p.returncode != 0:
        print(p.stderr.decode('utf8'))
        sys.exit("Failed to build 'pywrap.so'.")

dll_path = os.path.abspath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "..", "..", "_build", "default", "looprl", "bin", dll_name))

if not os.path.isfile(dll_path):
    msg = (f"Not found: {dll_path}. "
           f"Please use `make` to build the ocaml library.")
    sys.exit(msg)

dll = PyDLL(f"{dll_path}", RTLD_GLOBAL)
argv_t = c_char_p * 2
argv = argv_t(dll_name.encode('utf-8'), None)
dll.caml_startup(argv)

# This module is registered by the ocaml library
from looprl_ocaml_lib import *  # type: ignore

T = TypeVar("T", covariant=True)

class Graphable: ...

class SearchTree(Generic[T]): ...

class TeacherResult: ...

class TensorizerConfig: ...

class TokenizerConfig: ...

class GraphTensors: ...

class AgentSpec: ...
