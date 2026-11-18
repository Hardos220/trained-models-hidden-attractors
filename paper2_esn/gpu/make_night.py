"""Цепочки на ночь для А100 из gpu/a100_specs.txt: gpu/night_<буква>.sh, команды по порядку (готовое пропускается по skip-спискам).

Запуск: python gpu/make_night.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
specs = {}
for line in (ROOT / "gpu" / "a100_specs.txt").read_text(encoding="utf-8").splitlines():
    if line.strip():
        name, _, _, _, cmd = line.split("|", 4)
        specs[name] = cmd
CHAINS = {
    "A": ["esn-C1000-se3-H1-r0.9", "esn-C1000-se3-HE1-r0.6", "esn-C1000-se3-HE1-r0.9", "esn-H16-s5-10"],
    "B": ["esn-C1000-le1-H1-r0.6", "esn-C1000-le1-H1-r0.9", "esn-C1000-le1-HE1-r0.6", "esn-C1000-le1-HE1-r0.9"],
    "C": ["esn-H18-s5-7", "esn-H18-s8-10", "esn-H16-s11-15"],
    "D": ["esn-H18-s11-14", "esn-H20-s9-12"],
    "E": ["esn-H20-s3-5", "esn-H20-s6-8", "esn-H18-s15-17"],
}
assert sorted(sum(CHAINS.values(), [])) == sorted(specs), "каждая задача — ровно в одной цепочке"
for k, names in CHAINS.items():
    body = [f"#!/usr/bin/env bash", f"# Ночная цепочка {k}: " + ", ".join(names)]
    body += [specs[n] for n in names]
    (ROOT / "gpu" / f"night_{k}.sh").write_text("\n".join(body) + "\n", encoding="utf-8", newline="\n")
    print(k, names)
