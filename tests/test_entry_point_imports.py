"""Guard against broken imports in the CLI entry-point scripts.

Collect every import statement in console-script modules (including lazily loaded ones)
and verify each target module and each import name resolves.
"""

import ast
import importlib
import importlib.util
from pathlib import Path

import pytest

# The two registered console entry points (pyproject ``[tool.poetry.scripts]``:
# ``model-train-2d`` and ``model-predict-2d``). These are the supported CLI
# surface, so a broken import in either is a user-facing regression.
ENTRY_POINT_MODULES = [
    "volume_segmantics.scripts.train_2d_model",
    "volume_segmantics.scripts.predict_2d_model",
]

# Exception names that, when caught around an import, mark it as optional.
_IMPORT_GUARD_EXCEPTIONS = {"ImportError", "ModuleNotFoundError", "Exception"}


def _source_path(module_name):
    spec = importlib.util.find_spec(module_name)
    assert spec is not None and spec.origin, f"cannot locate source for {module_name}"
    return Path(spec.origin)


def _try_guards_imports(try_node):
    """True if any ``except`` of this ``try`` would swallow a failed import."""
    for handler in try_node.handlers:
        exc = handler.type
        if exc is None:  # bare ``except:`` guards everything
            return True
        names = []
        if isinstance(exc, ast.Tuple):
            names = [e.id for e in exc.elts if isinstance(e, ast.Name)]
        elif isinstance(exc, ast.Name):
            names = [exc.id]
        if any(n in _IMPORT_GUARD_EXCEPTIONS for n in names):
            return True
    return False


def _collect_imports(node, guarded, out):
    """Recursively gather ``(lineno, module, names)`` for every import.

    ``guarded`` propagates down ``try`` bodies whose ``except`` catches import
    errors, so optional imports are excluded.
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.Try):
            body_guarded = guarded or _try_guards_imports(child)
            for stmt in child.body:
                _collect_imports(stmt, body_guarded, out)
            for stmt in (*child.orelse, *child.finalbody):
                _collect_imports(stmt, guarded, out)
            for handler in child.handlers:
                for stmt in handler.body:
                    _collect_imports(stmt, guarded, out)
        elif isinstance(child, ast.Import):
            if not guarded:
                for alias in child.names:
                    out.append((child.lineno, alias.name, []))
        elif isinstance(child, ast.ImportFrom):
            if not guarded and not child.level:  # skip relative imports
                out.append(
                    (child.lineno, child.module, [a.name for a in child.names])
                )
        else:
            _collect_imports(child, guarded, out)


def _imports_in(module_name):
    tree = ast.parse(_source_path(module_name).read_text(encoding="utf-8"))
    collected = []
    _collect_imports(tree, False, collected)
    return collected


def _resolve(module, names):
    mod = importlib.import_module(module)
    for name in names:
        if hasattr(mod, name):
            continue
        # ``from pkg import submodule`` where the submodule isn't yet an
        # attribute of the package -- importing it explicitly is the real test.
        importlib.import_module(f"{module}.{name}")


@pytest.mark.parametrize("entry_module", ENTRY_POINT_MODULES)
def test_entry_point_imports_resolve(entry_module):
    """Every import (incl. lazy, function-local) in the CLI scripts resolves."""
    imports = _imports_in(entry_module)
    assert imports, f"no imports parsed from {entry_module}"

    failures = []
    for lineno, module, names in imports:
        if not module:
            continue
        try:
            _resolve(module, names)
        except Exception as exc:  # noqa: BLE001 - report all, not just the first
            target = module + (f" ({', '.join(names)})" if names else "")
            failures.append(f"{entry_module}:{lineno}: import {target!r} -> {exc!r}")

    assert not failures, "Unresolvable import(s) in CLI entry point:\n" + "\n".join(
        failures
    )
