"""Apply default_factory semantics to legacy Safety-Gymnasium dataclasses.

Safety-Gymnasium 1.0 is imported by OmniSafe even for MetaDrive. Its NumPy-array
defaults predate Python 3.11's dataclass validation. This patch only converts
unhashable defaults to copied factories; it does not modify any RL algorithm,
MetaDrive environment, reward, cost, or policy update.
"""

import importlib.util
from pathlib import Path

HELPER = '''from copy import deepcopy
from dataclasses import dataclass as _dataclass, field, Field

def dataclass(cls=None, **kwargs):
    def wrap(klass):
        for name in klass.__dict__.get("__annotations__", {}):
            if name not in klass.__dict__:
                continue
            value = klass.__dict__[name]
            if not isinstance(value, Field) and type(value).__hash__ is None:
                setattr(klass, name, field(default_factory=lambda value=value: deepcopy(value)))
        return _dataclass(klass, **kwargs)
    return wrap if cls is None else wrap(cls)
'''


if __name__ == "__main__":
    root = Path(importlib.util.find_spec("safety_gymnasium").origin).parent
    (root / "_python311_compat.py").write_text(HELPER)
    count = 0
    for path in root.rglob("*.py"):
        if path.name == "_python311_compat.py":
            continue
        source = path.read_text()
        updated = source.replace("from dataclasses import dataclass",
                                 "from safety_gymnasium._python311_compat import dataclass")
        if updated != source:
            path.write_text(updated)
            count += 1
    print(f"Updated legacy dataclass imports in {count} dependency files")
