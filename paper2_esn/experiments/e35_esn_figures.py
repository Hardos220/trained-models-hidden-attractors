"""Рисунки этапа 3 по резервуарам (эхо-сети N = 300, обученные на Лоренце).

fig50_esn_reconstruction — истинный и реконструированный аттракторы, прогноз и VPT, сводка по реализациям;
fig51_esn_spurious       — галерея паразитных режимов в физических координатах (3D);
fig52_esn_outcomes       — исходы из куба и после толчков по реализациям;
fig53_esn_fixed_points   — спектры неподвижных точек; копии равновесий Лоренца C±;
fig54_esn_survival       — выживание реконструированного «аттрактора» и пример обрушения;
fig55_esn_search         — P(N) поиска редкого паразитного режима (если есть результаты e34).
Входы: results/e32_esn_study_N300.json, results/e33_esn_verify_N300.json, results/e33_esn_attractors_N300.npz,
results/e34_esn_search_*.json.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from hsearch import viz  # noqa: E402
from hsearch.viz import ESCAPE_GREY, INK, INK2, MUTED, SERIES, save  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

RES = ROOT / "results"
FIG = RES / "figures"
LYAP = 0.906
H = 0.02
# неподвижные точки системы Лоренца (10, 28, 8/3): C± = (±√(β(ρ−1)), ±√(β(ρ−1)), ρ−1)
C_PM = np.array([[np.sqrt(8 / 3 * 27), np.sqrt(8 / 3 * 27), 27.0], [-np.sqrt(8 / 3 * 27), -np.sqrt(8 / 3 * 27), 27.0]])
RE_C = 0.0940     # Re собственного значения неустойчивой пары у C± при ρ = 28


def load():
    e32 = json.loads((RES / "e32_esn_study_N300.json").read_text(encoding="utf-8"))
    e33 = json.loads((RES / "e33_esn_verify_N300.json").read_text(encoding="utf-8"))
    z = np.load(RES / "e33_esn_attractors_N300.npz")
    return {r["seed"]: r for r in e32}, {r["seed"]: r for r in e33}, z


def scale_of(seed):
    """Масштаб нормировки выхода (СКО компонент Лоренца по обучающему ряду): u_phys = u * scale."""
    from hsearch.systems.reservoir import lorenz_data
    u = lorenz_data(20000)
    return u.std(axis=0)


def fate(rec33):
    """Итоговая классификация куба с учётом проверки времени жизни: прочие режимы, не прошедшие
    проверку, относятся к тому исходу, в который они перешли (по доле after_life)."""
    cube = dict(rec33["cube"])
    res = {"реконструированный аттрактор": cube.pop("lorenz", 0) / 1000.0, "паразитная устойчивая точка": 0.0,
           "паразитный регулярный режим": 0.0, "долгий переходный процесс": 0.0}
    for k, v in list(cube.items()):
        if k.startswith("fp:"):
            res["паразитная устойчивая точка"] += v / 1000.0
        elif k.startswith("c"):
            res["паразитный регулярный режим"] += v / 1000.0
    other = cube.get("other_chaos", 0) + cube.get("other_periodic", 0)
    if other:
        # куда перешли неподтверждённые кандидаты: распределение after_life каждого кластера
        # взвешивается числом его членов
        tot, wsum = {}, 0.0
        for c in rec33["candidates"]:
            n_al = sum(c["after_life"].values())
            if c["persist"] == 0 and n_al:
                for k, v in c["after_life"].items():
                    tot[k] = tot.get(k, 0.0) + c["n_members"] * v / n_al
                wsum += c["n_members"]
        if wsum:
            for k, v in tot.items():
                share = other / 1000.0 * v / wsum
                if k.startswith("fp:"):
                    res["паразитная устойчивая точка"] += share
                elif k == "lorenz":
                    res["реконструированный аттрактор"] += share
                else:
                    res["долгий переходный процесс"] += share
        else:
            res["долгий переходный процесс"] += other / 1000.0
    return res


def fig_reconstruction(e32, e33, seed=2):
    from hsearch.reservoir_analysis import ClosedLoopESN
    from hsearch.systems.reservoir import lorenz_data, train_esn
    u_all = lorenz_data(25000)
    esn = train_esn(u_all[:20000], n_res=300, seed=seed)
    m = ClosedLoopESN(esn)
    sc = esn["scale"]
    # реконструированный аттрактор: замкнутый прогон с конца обучения
    R = esn["R_train"][-1][None, :].copy()
    U = np.empty((6000, 3))
    for k in range(6000):
        R = m.step(R)
        U[k] = m.output(R)[0] * sc
    fig = plt.figure(figsize=(12, 8.2))
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    tr = u_all[5000:11000]
    ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=SERIES[0], lw=0.35)
    ax.set_title("Система Лоренца (10, 28, 8/3)", fontsize=10)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=18, azim=-60)
    ax = fig.add_subplot(2, 2, 2, projection="3d")
    ax.plot(U[1000:, 0], U[1000:, 1], U[1000:, 2], color=SERIES[1], lw=0.35)
    ax.set_title(f"Эхо-сеть N = 300 (реализация {seed}) в замкнутом режиме", fontsize=10)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=18, azim=-60)
    # прогноз против истины
    ax = fig.add_subplot(2, 2, 3)
    T = 900
    truth = u_all[20000:20000 + T]
    R = esn["R_train"][-1][None, :].copy()
    pred = np.empty((T, 3))
    for k in range(T):
        pred[k] = m.output(R)[0] * sc
        R = m.step(R)
    t = np.arange(T) * H * LYAP
    ax.plot(t, truth[:, 0], color=SERIES[0], lw=1.2, label="истинный ряд")
    ax.plot(t, pred[:, 0], color=SERIES[1], lw=1.0, ls="--", label="прогноз эхо-сети")
    v = e32[seed]["vpt_lyap"]
    ax.axvline(v, color=INK2, lw=1, ls=":")
    ax.text(v + 0.2, 17, f"VPT = {v:.1f} ляп. врем.", fontsize=8, color=INK2)
    ax.set_xlabel("время, ляпуновские времена (λ·t)"); ax.set_ylabel("x(t)")
    ax.set_title("Краткосрочный прогноз с конца обучающего ряда", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    # сводка по реализациям: VPT (столбцы), цвет — итоговая долгосрочная динамика, подпись — λ «климата»
    from matplotlib.patches import Patch
    ax = fig.add_subplot(2, 2, 4)
    seeds = sorted(e32)
    cats = {}
    for i, s in enumerate(seeds):
        c, lab = category(e32[s], e33.get(s))
        cats[lab] = c
        ax.barh(i, e32[s]["vpt_lyap"], color=c, height=0.62, edgecolor=viz.SURFACE)
        ax.text(e32[s]["vpt_lyap"] + 0.12, i, f"λ = {e32[s]['climate']['lam']:.2f}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(seeds))); ax.set_yticklabels([f"реализация {s}" for s in seeds], fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(0, 11)
    ax.set_xlabel("VPT, ляп. времена (справа — λ «климата»; у системы Лоренца 0.906)")
    ax.set_title("Восемь реализаций: краткосрочный прогноз и долгосрочная динамика", fontsize=10)
    ax.legend(handles=[Patch(color=c, label=l) for l, c in cats.items()], fontsize=7, loc="upper center",
              bbox_to_anchor=(0.5, -0.17), ncol=2)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, FIG / "fig50_esn_reconstruction")


def category(r32, r33):
    """Итоговая долгосрочная динамика реализации."""
    lam = r32["climate"]["lam"]
    if lam <= -0.5:
        return SERIES[7], "обрушение в неподвижную точку"
    if lam < 0.3:
        return SERIES[3], "периодический «климат»"
    if r33 is None:
        return MUTED, "нет данных"
    f = fate(r33)
    coll = r33["survival"]["fraction_collapsed"]
    if f["реконструированный аттрактор"] > 0.98 and coll < 0.01:
        return SERIES[2], "паразитных режимов не найдено"
    if coll > 0.3:
        return SERIES[1], "хаос переходный, утечка в паразитную точку"
    return SERIES[6], "сосуществуют паразитные аттракторы"


def fig_spurious(e32, e33, z, sc):
    seeds = [s for s in sorted(e33) if (e33[s]["spurious_attractors"] or e33[s]["stable_fp"]) and f"s{s}_lorenz_u" in z]
    n = len(seeds)
    if n == 0:
        return
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(4.3 * cols, 4.0 * rows))
    for k, s in enumerate(seeds):
        r = e33[s]
        ax = fig.add_subplot(rows, cols, k + 1, projection="3d")
        L = z[f"s{s}_lorenz_u"] * sc
        ax.plot(L[:, 0], L[:, 1], L[:, 2], color=ESCAPE_GREY, lw=0.3, alpha=0.9)
        for j, a in enumerate(r["spurious_attractors"]):
            U = z[f"s{s}_{a['name']}_u"] * sc
            ax.plot(U[:, 0], U[:, 1], U[:, 2], color=SERIES[(j + 1) % 8], lw=0.8,
                    label=f"паразитный {'хаос' if a['kind'] == 'chaos' else 'регулярный режим'} (куб {a['p_cube']:.0%})")
        for i, fp in enumerate(r["fixed_points"]):
            u = np.array(fp["u"]) * sc
            if fp["rho"] < 1:
                ax.scatter(*u, color=SERIES[7], s=60, edgecolors=INK, depthshade=False, zorder=5)
            else:
                ax.scatter(*u, color=INK, marker="x", s=18, depthshade=False)
        ax.scatter(*C_PM.T, color=SERIES[3], marker="+", s=60, depthshade=False)
        ax.scatter([], [], color=SERIES[7], s=40, edgecolors=INK, label="устойчивая точка сети")
        ax.scatter([], [], color=INK, marker="x", s=18, label="неустойчивые точки сети")
        ax.scatter([], [], color=SERIES[3], marker="+", s=50, label="C± системы Лоренца")
        coll = r["survival"]["fraction_collapsed"]
        ax.set_title(f"реализация {s}: обрушение с аттрактора {coll:.0%} за 1000 ед. вр.", fontsize=9)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.view_init(elev=16, azim=-58)
        ax.legend(fontsize=6, loc="upper left")
    fig.suptitle("Паразитные режимы обученных эхо-сетей в физических координатах выхода (серое — реконструированный «аттрактор»)", fontsize=10)
    fig.tight_layout()
    save(fig, FIG / "fig51_esn_spurious")


def fig_outcomes(e32, e33):
    seeds = sorted(e33)
    cats = ["реконструированный аттрактор", "паразитная устойчивая точка", "паразитный регулярный режим", "долгий переходный процесс"]
    colors = [SERIES[0], SERIES[7], SERIES[6], ESCAPE_GREY]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    for i, s in enumerate(seeds):
        f = fate(e33[s])
        left = 0.0
        for c, col in zip(cats, colors):
            v = f[c]
            if v > 0:
                ax.barh(i, v, left=left, color=col, height=0.62, edgecolor=viz.SURFACE, linewidth=1.5)
            left += v
    from matplotlib.patches import Patch
    ax.set_yticks(range(len(seeds))); ax.set_yticklabels([f"реализация {s}" for s in seeds], fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("доля 1000 равномерных состояний куба [−1, 1]³⁰⁰")
    ax.set_title("Исходы из куба (горизонт 2·10⁴ шагов + проверка 5·10⁴)", fontsize=10)
    ax.legend(handles=[Patch(color=col, label=c) for c, col in zip(cats, colors)], fontsize=7,
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    sigs = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
    for j, s in enumerate(sorted(e32)):
        oc = e32[s]["outcomes"]
        y = []
        for sg in sigs:
            d = oc[f"kick_{sg:g}"]
            y.append(1 - d.get("lorenz", 0) / max(1, sum(d.values())))
        ax.plot(sigs, np.maximum(y, 2e-3), "o-", color=SERIES[j % 8], ms=4, lw=1.4, label=f"реализация {s}")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1.5e-3, 1.6)
    ax.set_xlabel("амплитуда толчка σ от состояния на реконструированном аттракторе")
    ax.set_ylabel("доля исходов вне аттрактора")
    ax.set_title("Достижимость паразитных режимов толчком (горизонт 6000 шагов)", fontsize=10)
    ax.text(1.1e-3, 1.75e-3, "нижняя граница — ни одного исхода из 200", fontsize=7, color=INK2)
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    save(fig, FIG / "fig52_esn_outcomes")


def fp_type(fp, sc):
    u = np.array(fp["u"]) * sc
    if np.min(np.linalg.norm(C_PM - u, axis=1)) < 2.0 and fp["rho"] > 1:
        return "C"
    return "S" if fp["rho"] < 1 else "U"


def fig_fixed_points(e33, z, sc):
    from hsearch.systems.reservoir import lorenz_data
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 0.9, 1.15]})
    # (а) ранговые спектры |μ| в трёх типах точек одной реализации
    ax = axes[0]
    s = 0
    r = e33[s]
    picked = {}
    for i, fp in enumerate(r["fixed_points"]):
        t = fp_type(fp, sc)
        if t == "U" and fp["n_unstable"] < 3:
            continue
        picked.setdefault(t, i)
    styles = {"C": (SERIES[3], "копия C± системы Лоренца"), "S": (SERIES[7], "устойчивая паразитная точка"),
              "U": (SERIES[0], "неустойчивая точка у копии нуля")}
    for t in ("C", "S", "U"):
        if t not in picked:
            continue
        i = picked[t]
        w = np.sort(np.abs(z[f"s{s}_fp{i}_eig"]))[::-1][:24]
        col, lab = styles[t]
        ax.plot(np.arange(1, len(w) + 1), w, "o-", color=col, ms=4, lw=1.4,
                label=f"{lab}: ρ = {r['fixed_points'][i]['rho']:.4f}")
    ax.axhline(1.0, color=INK, lw=0.9, ls="--")
    ax.set_xlabel("номер собственного значения (по убыванию |μ|)"); ax.set_ylabel("|μ| якобиана отображения")
    ax.set_title(f"Спектры в неподвижных точках (реализация {s})", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    # (б) скорость неустойчивости копий C± против истинной
    ax = axes[1]
    rates = []
    for s, r in sorted(e33.items()):
        k = 0
        for fp in r["fixed_points"]:
            if fp_type(fp, sc) == "C":
                rate = np.log(fp["rho"]) / H
                rates.append(rate)
                ax.scatter(s + (k - 0.5) * 0.18, rate, color=SERIES[3], marker="D", s=36, edgecolors=INK, linewidths=0.4, zorder=3)
                k += 1
    ax.axhline(RE_C, color=INK, lw=1, ls="--")
    ax.text(-0.3, RE_C + 0.0012, "система Лоренца: Re λ = 0.0940", fontsize=8, color=INK2)
    ax.set_ylim(0.07, 0.13)
    ax.set_xlabel("реализация"); ax.set_ylabel("ln|μ_max| / h у копий C±")
    ax.set_title(f"Копии C±: {np.mean(rates):.3f} ± {np.std(rates):.3f} против 0.094", fontsize=10)
    # (в) положения всех неподвижных точек относительно аттрактора Лоренца
    ax = axes[2]
    L = lorenz_data(6000)
    ax.plot(L[:, 0], L[:, 2], color=ESCAPE_GREY, lw=0.3)
    for s, r in sorted(e33.items()):
        for fp in r["fixed_points"]:
            u = np.array(fp["u"]) * sc
            t = fp_type(fp, sc)
            if t == "S":
                ax.scatter(u[0], u[2], color=SERIES[7], s=46, edgecolors=INK, linewidths=0.5, zorder=4)
            elif t == "C":
                ax.scatter(u[0], u[2], color=SERIES[3], marker="D", s=26, edgecolors=INK, linewidths=0.4, zorder=3)
            else:
                ax.scatter(u[0], u[2], color=SERIES[0], marker="x", s=14, alpha=0.7, zorder=2)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.scatter([], [], color=SERIES[7], s=46, edgecolors=INK, label="устойчивые (паразитные)")
    ax.scatter([], [], color=SERIES[3], marker="D", s=26, edgecolors=INK, label="копии C±")
    ax.scatter([], [], color=SERIES[0], marker="x", s=14, label="прочие неустойчивые")
    ax.set_xlabel("x"); ax.set_ylabel("z")
    ax.set_title("Неподвижные точки восьми сетей в плоскости (x, z)", fontsize=10)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    save(fig, FIG / "fig53_esn_fixed_points")


def fig_survival(e33, z, sc):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    for j, (s, r) in enumerate(sorted(e33.items())):
        S = np.array(r["survival"]["S"]); t = np.array(r["survival"]["t_steps"]) * H * LYAP
        # опорные состояния брались с одной траектории; если она уже обрушилась, часть стартов
        # лежит в паразитной точке — нормируем на долю стартовавших на хаотическом множестве
        S0 = S[0] if S[0] > 0 else 0.0
        if S0 > 0.05 and S[-1] / S0 < 0.995:
            Sn = S / S0
            tau = None
            k = np.where((Sn < 0.95) & (Sn > 0.02))[0]
            if len(k) > 3:
                tau = -1.0 / np.polyfit(t[k], np.log(Sn[k]), 1)[0]
            ax.plot(t, np.maximum(Sn, 2.5e-3), color=SERIES[j % 8], lw=1.8,
                    label=f"реализация {s}" + (f": τ ≈ {tau:.0f} ляп. врем." if tau else ""))
    ax.set_yscale("log"); ax.set_ylim(5e-3, 1.3)
    ax.set_xlabel("время, ляпуновские времена"); ax.set_ylabel("доля траекторий, ещё не попавших в точку")
    ax.set_title("«Выживание» на реконструированном хаотическом множестве", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    ax = axes[1]
    pick = None
    for s in sorted(e33):
        if f"s{s}_collapse_u" in z:
            u = z[f"s{s}_collapse_u"]
            if pick is None or (len(u) > 1500 and len(pick[1]) < 1500):
                pick = (s, u)
    if pick:
        s, u = pick
        U = u * sc
        t = np.arange(len(U)) * 2 * H * LYAP
        ax.plot(t, U[:, 0], color=SERIES[0], lw=0.6, label="x(t)")
        ax.plot(t, U[:, 2], color=SERIES[1], lw=0.6, label="z(t)")
        ax.set_xlabel("время, ляпуновские времена"); ax.set_ylabel("выход сети (физические единицы)")
        ax.set_title(f"Реализация {s}: «климат Лоренца», затем обрушение в паразитную точку", fontsize=10)
        ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    save(fig, FIG / "fig54_esn_survival")


def fig_hidden():
    """Скрытый паразитный аттрактор эхо-сети N = 100 (реализация 0): фазовый портрет, сечение бассейнов,
    мера скрытости по окрестностям неподвижных точек."""
    pj, pz = RES / "e36_hidden_esn.json", RES / "e36_hidden_esn.npz"
    if not (pj.exists() and pz.exists()):
        return
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    d = json.loads(pj.read_text(encoding="utf-8"))
    z = np.load(pz)
    sc = z["scale"]
    fig = plt.figure(figsize=(15, 4.9))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    A = z["att_u"] * sc
    ax.scatter(A[:, 0], A[:, 1], A[:, 2], color=MUTED, s=0.6, alpha=0.6, depthshade=False)
    ax.scatter([], [], color=MUTED, s=6, label="реконструированный аттрактор")
    style = {"c0": (SERIES[1], "паразитный хаос (самовозбуждающийся)"), "c1": (SERIES[2], "скрытый регулярный аттрактор")}
    for k in ("c0", "c1"):
        if f"{k}_u" in z:
            U = z[f"{k}_u"] * sc
            ax.plot(U[:, 0], U[:, 1], U[:, 2], color=style[k][0], lw=0.6, label=style[k][1])
    for u, rho in zip(z["fp_u"] * sc, z["fp_rho"]):
        if rho < 1:
            ax.scatter(*u, color=SERIES[7], s=60, edgecolors=INK, depthshade=False, zorder=5)
        else:
            ax.scatter(*u, color=INK, marker="x", s=18, depthshade=False)
    ax.scatter(*C_PM.T, color=SERIES[3], marker="+", s=60, depthshade=False)
    ax.scatter([], [], color=SERIES[7], s=40, edgecolors=INK, label="устойчивая точка")
    ax.scatter([], [], color=INK, marker="x", s=18, label="неустойчивые точки")
    ax.scatter([], [], color=SERIES[3], marker="+", s=50, label="C± системы Лоренца")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z"); ax.view_init(elev=16, azim=-58)
    ax.set_title("Эхо-сеть N = 100: режимы в координатах выхода", fontsize=10)
    ax.legend(fontsize=6.5, loc="upper left")
    # сечение бассейнов
    ax = fig.add_subplot(1, 3, 2)
    labs = d["slice_labels"]
    colmap = {"lorenz": SERIES[0], "c0": SERIES[1], "c1": SERIES[2]}
    cols = []
    for l in labs:
        if l in colmap:
            cols.append(colmap[l])
        elif l.startswith("fp:"):
            cols.append(SERIES[7])
        else:
            cols.append(ESCAPE_GREY)
    img = z["slice"]
    a, b = z["slice_a"], z["slice_b"]
    ax.imshow(img, origin="lower", extent=[a.min(), a.max(), b.min(), b.max()], cmap=ListedColormap(cols),
              vmin=-0.5, vmax=len(labs) - 0.5, aspect="auto", interpolation="nearest")
    for (xa, xb, mk, lab) in ((0, 0, "o", "устойчивая точка"), (1, 0, "s", "точка скрытого аттрактора"),
                              (0, 1, "^", "точка реконструированного аттрактора")):
        ax.scatter(xa, xb, marker=mk, s=60, color="white", edgecolors=INK, zorder=4)
    nice = {"lorenz": "реконструированный аттрактор", "c0": "паразитный хаос", "c1": "скрытый аттрактор"}
    ax.legend(handles=[Patch(color=c, label=nice.get(l, "устойчивая точка" if l.startswith("fp:") else l))
                       for l, c in zip(labs, cols)], fontsize=7, loc="upper right", frameon=True)
    ax.set_xlabel("α (направление к скрытому аттрактору)"); ax.set_ylabel("β (направление к аттрактору)")
    ax.set_title("Сечение бассейнов плоскостью в ℝ¹⁰⁰", fontsize=10)
    # мера скрытости
    ax = fig.add_subplot(1, 3, 3)
    eps = sorted(d["h_eps"], key=float)
    x = [float(e) for e in eps]
    for key, col, lab in (("lorenz", SERIES[0], "реконструированный аттрактор"), ("c0", SERIES[1], "паразитный хаос"),
                          ("c1", SERIES[2], "скрытый аттрактор")):
        y = [d["h_eps"][e].get(key, 0) / d["h_eps"][e]["n"] for e in eps]
        ax.plot(x, np.maximum(y, 1e-4), "o-", color=col, lw=2.2 if key == "c1" else 1.5, label=lab)
    yfp = [sum(v for k, v in d["h_eps"][e].items() if k.startswith("fp:")) / d["h_eps"][e]["n"] for e in eps]
    ax.plot(x, np.maximum(yfp, 1e-4), "s:", color=SERIES[7], lw=1.2, ms=4, label="устойчивая точка")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(5e-5, 1.5)
    ax.text(x[0], 1.4e-4, f"скрытый аттрактор и устойчивая точка: 0 из {d['h_eps'][eps[0]]['n']} при каждом ε",
            fontsize=7, color=INK2)
    ax.set_xlabel("ε — радиус окрестности неустойчивых неподвижных точек"); ax.set_ylabel("доля траекторий h_ε")
    ax.set_title("Мера скрытости по всем неустойчивым точкам", fontsize=10)
    ax.legend(fontsize=7, loc="center right")
    fig.tight_layout()
    ax2 = fig.axes[1]
    pos = ax2.get_position()
    ax2.set_position([pos.x0 + 0.03, pos.y0, pos.width - 0.03, pos.height])
    save(fig, FIG / "fig56_esn_hidden")


def esn_search_stats(d):
    """N_0.9 стратегий и выигрыш относительно перебора. Если у перебора часть запусков не нашла цель
    (цензурирование по бюджету), N⁰_0.9 = ln 10 / p̂, где p̂ — оценка максимального правдоподобия
    геометрического распределения по цензурированной выборке: p̂ = (число попаданий) / Σ min(T_i, бюджет)."""
    from hsearch.search.harness import n_delta_from_runs
    B = d["_meta"]["budget"]
    fa = d["uniform"]["found_at"]
    k = sum(1 for x in fa if x is not None)
    tot = sum(x if x is not None else B for x in fa)
    p_hat = k / tot
    n0 = np.log(10) / p_hat
    res = {"p_hat": p_hat, "N0_theory": n0, "uniform_success": k / len(fa)}
    for s in ("uniform", "sus", "portfolio_no_sus", "portfolio"):
        if s not in d:
            continue
        q, lo, hi = n_delta_from_runs(d[s]["found_at"])
        succ = np.mean([x is not None for x in d[s]["found_at"]])
        med = float(np.median([x if x is not None else np.inf for x in d[s]["found_at"]]))
        res[s] = {"N09": q, "ci": (lo, hi), "G": (n0 / q) if np.isfinite(q) and q > 0 else float("nan"),
                  "success": succ, "median": med}
    return res


def fig_search():
    files = sorted(glob.glob(str(RES / "e34_esn_search_*.json")))
    if not files:
        return
    from hsearch.search.harness import pn_curve
    fig, axes = plt.subplots(1, len(files), figsize=(5.6 * len(files), 4.0), squeeze=False)
    lab = {"uniform": "равномерный перебор", "sus": "расщепление", "portfolio_no_sus": "активное обучение + 20 % равн.",
           "portfolio": "портфель"}
    col = {"uniform": MUTED, "sus": SERIES[1], "portfolio_no_sus": SERIES[2], "portfolio": SERIES[0]}
    for ax, f in zip(axes[0], files):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        meta = d["_meta"]
        st = esn_search_stats(d)
        for k in ("uniform", "sus", "portfolio_no_sus", "portfolio"):
            if k in d:
                g, P = pn_curve(d[k]["found_at"], meta["budget"])
                G = st[k]["G"]
                ax.step(g, P, where="post", color=col[k], lw=2.6 if k == "portfolio" else 1.6,
                        label=f"{lab[k]}" + (f" ($G_{{0.9}}$ = {G:.2f})" if G and np.isfinite(G) else ""))
        nn = np.logspace(0, np.log10(meta["budget"]), 200)
        ax.plot(nn, 1 - (1 - st["p_hat"]) ** nn, color=MUTED, lw=1, ls="--", label=f"перебор, теория: p̂ = {st['p_hat']:.1e}")
        ax.axhline(0.9, color=MUTED, lw=0.8, ls=":")
        ax.set_xscale("log"); ax.set_xlabel("N, оценок (прогонов сети)"); ax.set_ylabel("P(N)")
        ax.set_title(f"Сеть {meta['seed']} (N = 300): поиск редкой паразитной точки, R = {meta['R']}", fontsize=10)
        ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    save(fig, FIG / "fig55_esn_search")


def main():
    viz.setup()
    e32, e33, z = load()
    sc = scale_of(0)
    for f, args in ((fig_reconstruction, (e32, e33)), (fig_spurious, (e32, e33, z, sc)), (fig_outcomes, (e32, e33)),
                    (fig_fixed_points, (e33, z, sc)), (fig_survival, (e33, z, sc)), (fig_hidden, ()), (fig_search, ())):
        try:
            f(*args)
            print("ok:", f.__name__, flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print("skip:", f.__name__, repr(e), flush=True)


if __name__ == "__main__":
    main()
