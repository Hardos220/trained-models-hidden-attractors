"""Списки готового для --skip по всем локальным результатам: data/c_skip.txt (<режим>/<имя>) и data/h_skip.txt (N<N>/<имя>).

Готовой считается модель, у которой есть .json (файлы .failed.txt не считаются).
Запуск: python gpu/make_skip.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRS = [ROOT / "gpuq-results", ROOT / "local_results", ROOT / "node_results"]


def done(glob):
    out = set()
    for d in DIRS:
        for f in d.rglob(glob):
            if ".failed" not in f.name:
                out.add(f"{f.parent.name}/{f.stem}")
    return sorted(out)


for name, glob in (("c_skip.txt", "C/*/*.json"), ("h_skip.txt", "H/N*/*.json")):
    keys = done(glob)
    (ROOT / "data" / name).write_text("\n".join(keys) + "\n", encoding="utf-8")
    print(name, len(keys))
