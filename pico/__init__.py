"""Backward-compatible import path for projects using the old package name."""

import sys
from importlib import import_module, util
from importlib.abc import Loader, MetaPathFinder

from codeflow import *  # noqa: F401,F403
from codeflow import __all__ as __all__


class _AliasLoader(Loader):
    def __init__(self, target_name):
        self.target_name = target_name

    def create_module(self, spec):
        return import_module(self.target_name)

    def exec_module(self, module):
        sys.modules[module.__spec__.name] = module


class _AliasFinder(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("pico."):
            return None
        target_name = "codeflow" + fullname[len("pico") :]
        target_spec = util.find_spec(target_name)
        if target_spec is None:
            return None
        is_package = target_spec.submodule_search_locations is not None
        return util.spec_from_loader(
            fullname,
            _AliasLoader(target_name),
            is_package=is_package,
        )


sys.meta_path.insert(0, _AliasFinder())
