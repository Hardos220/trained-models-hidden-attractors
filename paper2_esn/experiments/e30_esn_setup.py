"""Этап 3, подготовка: обучение ESN на Лоренце, проверка замкнутого режима (воспроизведение
«климата»: λ и статистики выхода), регистрация системы, первая выборка начальных условий
в кубе [-1, 1]^N и в окрестности обучающих состояний. Выход: results/e30_esn_setup.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from hsearch.systems.reservoir import make_esn_system  # noqa: E402
from hsearch.identify import LAB_ESCAPE, LAB_UNKNOWN, build_reference_set, calibrate_reference, label_fractions, match  # noqa: E402
from hsearch.integrate import F_FP, F_LAM, F_RMAX, run_batch  # noqa: E402


def main(n_res=300, seed=0):
    t0 = time.time()
    s = make_esn_system(n_res=n_res, seed=seed)
    esn = s.esn
    print(f"ESN N={n_res}: RMSE обучения {esn['train_rmse']:.3g}, время {time.time() - t0:.1f} с")
    # замкнутый прогон с последнего обучающего состояния
    rng = np.random.default_rng(30)
    r0 = s.known["LORENZ"]
    t0 = time.time()
    feats, finals = run_batch(s, r0[None, :])
    print(f"замкнутый цикл из обучающего состояния: λ = {feats[0, F_LAM]:.4f} (на шаг h = 0.02 -> {feats[0, F_LAM] / 0.02:.3f} на единицу времени), "
          f"FP = {feats[0, F_FP]}, rmax = {feats[0, F_RMAX]:.2f}  ({time.time() - t0:.1f} с)")
    # выход u = W_out r на окне: сравнение статистик с обучающими данными
    tr = s.traj(r0, n=s.n_tr + s.n_obs)[s.n_tr:]
    u_hat = tr @ esn["W_out"].T
    un = esn["u_norm"]
    print("статистики выхода (норм.): mean", np.round(u_hat.mean(0), 3), "std", np.round(u_hat.std(0), 3),
          "| обучение: mean", np.round(un.mean(0), 3), "std", np.round(un.std(0), 3))
    out = {"n_res": n_res, "seed": seed, "train_rmse": esn["train_rmse"], "lam_closed_loop_per_step": float(feats[0, F_LAM]),
           "u_hat_mean": u_hat.mean(0).tolist(), "u_hat_std": u_hat.std(0).tolist(),
           "u_train_mean": un.mean(0).tolist(), "u_train_std": un.std(0).tolist()}
    # эталон: реконструированный аттрактор; выборка в кубе и в окрестности обучающих состояний
    ref = calibrate_reference(s, "LORENZ", r0, rng, n_rep=32)
    print("эталон LORENZ:", ref.kind, "λ =", round(ref.lam, 4))
    for region, n in (("cube", 400), ("near_train", 400)):
        t0 = time.time()
        if region == "cube":
            ics = s.sample_box(n, rng)
        else:
            R = esn["R_train"]
            picks = R[rng.integers(0, len(R), size=n)]
            ics = np.clip(picks + 0.05 * rng.normal(size=picks.shape), -1, 1)
        f, fin = run_batch(s, ics)
        labels, z = match(s, f, fin, [ref])
        kinds = {"escape": int((f[:, 0] > 0.5).sum()), "fp": int((f[:, F_FP] > 0.5).sum()),
                 "chaotic(λ>0.02/шаг)": int((f[:, F_LAM] > 0.02).sum()), "lorenz_match": int((labels == 0).sum()),
                 "unknown": int((labels == LAB_UNKNOWN).sum())}
        out[region] = kinds
        print(f"{region}: {kinds}  ({time.time() - t0:.1f} с)")
    (ROOT / "results" / "e30_esn_setup.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
