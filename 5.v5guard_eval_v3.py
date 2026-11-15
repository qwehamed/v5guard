#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V5Guard evaluation harness v3  (Colab-ready, no GPU, no network)
==================================================================
Builds on v2. New in v3 (each block answers one reviewer point):

 R1  B4 packet-cap sweep  (exp "cap")
       max_packet_size in {256,512,1K,2K,4K,8K} for B0 / B4 / V5G against A2, A2fit, A4_enf.
       Reports availability AND legit false-rejection (benign run) per cap, plus a verdict
       table (worst-case availability over attacks, false rejection on benign).
       A2fit = NEW attacker that fills whatever packet cap is configured with tiny (8 B)
       user properties. Without it a cap sweep against the fixed A2 (400x16 B) is trivial:
       any cap < ~6.6 KB rejects A2 by size alone. A4_enf also adapts to the cap now
       (identical to v2 at the default 8192 B).

 R2  Fair detector comparison (exp "confirm")
       + V5G+B2 (enforcer + Isolation Forest), V5G+B3 (enforcer + RF, leave-one-family-out)
       Confirmation window k in {1,2,3,5} for ALL detectors. Every threshold is re-calibrated
       to the same client-level FPR (alpha) for each k, so k changes speed, not the FPR target.
       Reports availability, TTD (first attacker) and TTD90 (90% of attackers), TPR, FPR,
       paired difference against the no-detector reference (B0 / V5G).

 R3  A4_bud + mitigator ablation + sensitivity
       A4_bud: attacker that reads the calibrated per-feature budget of the +bud profiler and
               stays at `a4_margin` (0.7) of it on every feature, at the enforcer ceilings,
               with many identities (Sybil sweep up to 1280 nodes).
       Load-aware vs fixed threshold:
               V5G+z   (tau(t), original)  vs  V5G+zfix (tau = tau0)
               V5G+bud+LA (threshold * tau(t)/tau0)  vs  V5G+bud (fixed)
       One-at-a-time sensitivity on w_alpha (=alpha of Eq.1, beta = 1-alpha), ewma_lam (lambda),
       tau0, rho, gamma for V5G+z and V5G+bud+LA. (The FPR calibration level is called
       `alpha` in the code; the footprint weight is `w_alpha` to avoid the name clash.)

HONEST CAVEATS
 * Still a simulation. Absolute numbers depend on P (capacity, costs).
 * v2 had no EWMA and a hard-coded 0.5/0.5 footprint weight. v3 adds both; defaults
   (ewma_lam=1.0, w_alpha=0.5) reproduce v2 behaviour, so old tables stay comparable.
 * "Load-aware" for the calibrated +bud profiler is MY operationalisation:
   thr_eff = thr * max(tau_min, tau0 - gamma*load/rho) / tau0. The paper only defines
   tau(t) for the z-score. Say so in the paper.
 * A4_bud uses Poisson arrivals like everyone else, so its realised rate feature is noisy;
   a_margin=0.7 is a tuning choice, not a theorem.

USAGE (Colab):
    !python v5guard_eval_v3.py --seeds 5  --out results      # first real run
    !python v5guard_eval_v3.py --seeds 10 --out results      # full
    !python v5guard_eval_v3.py --quick --out results_quick   # smoke test
    !python v5guard_eval_v3.py --seeds 5 --exps cap,confirm  # only some blocks
"""
import argparse, math, os, sys, time, json
from dataclasses import dataclass, asdict, replace
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest, RandomForestClassifier

FEATS = ["rate", "props", "bytes", "queue", "reconn"]
K = len(FEATS)
FAMS = ["A1", "A2", "A3", "A4_B1", "A4_enf"]      # families B3 trains on / holds out
EXTRA = ["A2fit", "A4_bud"]                        # adaptive attackers no learner trains on
MATRIX_SCN = FAMS + EXTRA
SCNS = ["benign"] + FAMS + EXTRA                   # order fixes RNG streams (append only!)
PROF = ("z_orig", "rz", "bud")                     # detectors that use EWMA smoothing

DEFS_ALL = ["B0", "B1", "B2", "B2_c10", "B3_seen", "B3_lofo", "B4",
            "V5G", "V5G+z", "V5G+zfix", "V5G+rz", "V5G+bud", "V5G+bud+LA", "V5G+B2", "V5G+B3"]
DEFS_SWEEP = ["B0", "B1", "B2", "B3_lofo", "B4", "V5G", "V5G+rz", "V5G+bud",
              "V5G+bud+LA", "V5G+B2", "V5G+B3"]
DEFS_CONF_DET = ["B2", "B3_lofo", "V5G+B2", "V5G+B3", "V5G+z", "V5G+zfix",
                 "V5G+rz", "V5G+bud", "V5G+bud+LA"]
# name -> (detector, load_aware, leave_one_family_out)
SPEC = {"V5G": (None, False, False), "V5G+z": ("z_orig", True, False),
        "V5G+zfix": ("z_orig", False, False), "V5G+rz": ("rz", False, False),
        "V5G+bud": ("bud", False, False), "V5G+bud+LA": ("bud", True, False),
        "V5G+B2": ("if", False, False), "V5G+B3": ("rf", False, True)}


# --------------------------------------------------------------------------
# Parameters (everything that affects a result lives here -> params.csv)
# --------------------------------------------------------------------------
@dataclass
class P:
    # timing
    dt: float = 0.1
    T: float = 60.0
    win: float = 1.0
    # legit workload
    n_legit: int = 50
    legit_rate_mean: float = 2.0
    legit_rate_sigma: float = 0.4
    legit_props_mean: float = 2.0
    legit_props_sigma: float = 0.3
    legit_prop_bytes: float = 30.0
    legit_recon: float = 0.05
    base_bytes: float = 200.0
    # broker capacity / cost model (arbitrary work units)
    cap: float = 5000.0
    buffer_s: float = 2.0
    c0: float = 1.0
    c_prop: float = 0.02
    q_quad: float = 0.0002
    c_hdr: float = 0.2
    c_conn: float = 5.0
    c_q: float = 0.05
    parse_mode: str = "partial"
    # memory model
    mem_budget_MB: float = 64.0
    k_mem: float = 2.0
    conn_bytes: float = 10_000.0
    sess_bytes: float = 20_000.0
    msg_bytes: float = 1000.0
    # defense costs (CHARGED to the broker)
    c_def_pkt: float = 0.02
    c_ml_client: float = 0.1
    # ceilings (operator-set from legit workload)
    ceil_margin: float = 1.5
    packet_cap: float = 8192.0
    expiry_cap: float = 10.0
    queue_cap: float = 100.0
    expiry_default: float = 3600.0
    queue_default: float = 1e6
    # attacks
    a2_props: int = 400
    a2_prop_bytes: float = 16.0
    a2_rate: float = 5.0
    a1_recon: float = 1.0
    a3_qrate: float = 20.0
    pb_fit: float = 8.0               # A2fit / A4_bud bytes per user property (tiny)
    a4_margin: float = 0.7            # A4_bud stays at this fraction of the budget
    a4_ref_consec: int = 3            # A4_bud targets the budget calibrated at THIS k and lambda=1,
                                      # so the attacker is identical across the k / lambda sweeps
    # detection
    alpha: float = 0.01               # target client-level FPR of calibrated detectors
    consec: int = 3                   # confirmation windows
    tau0: float = 3.0
    tau_min: float = 1.5
    gamma: float = 1.0
    rho: float = 0.7
    w_alpha: float = 0.5              # alpha of Eq.(1): memory weight; beta = 1 - w_alpha
    ewma_lam: float = 1.0             # lambda of Eq.(2); 1.0 = no smoothing (= v2)
    deadline: float = 0.2
    # protocol
    B_matrix: int = 40
    B_hi: int = 320                   # "enforcer is not enough" regime for A4_enf
    B_syb: int = 1000                 # Sybil regime for A4_bud
    B_sweep: tuple = (0, 5, 10, 20, 40, 80, 160, 320)
    B_sybil: tuple = (0, 40, 160, 320, 640, 1000, 1280, 2560)
    warm_B: tuple = (10, 40, 100)
    nb_train: int = 4
    nb_cal: int = 6
    equiv_margin_pct: float = 5.0
    # R1 / R2 / R3 grids
    cap_sweep: tuple = (256, 512, 1024, 2048, 4096, 8192)
    cap_B: tuple = (40, 320)
    consec_set: tuple = (1, 2, 3, 5)
    sens: tuple = (("w_alpha", (0.0, 0.25, 0.5, 0.75, 1.0)),
                   ("ewma_lam", (0.1, 0.2, 0.5, 1.0)),
                   ("tau0", (2.0, 3.0, 4.0, 6.0)),
                   ("rho", (0.3, 0.5, 0.7, 0.9)),
                   ("gamma", (0.0, 0.5, 1.0, 2.0)))
    ok_avail: float = 95.0            # verdict: worst-case on-time availability >= this
    ok_fr: float = 1.0                # verdict: benign false rejection <= this (%)


def lam_set(p):
    return tuple(sorted(set(dict(p.sens).get("ewma_lam", ())) | {1.0, float(p.ewma_lam)}))


# --------------------------------------------------------------------------
# Derived ceilings and defense configuration
# --------------------------------------------------------------------------
def derive_limits(p):
    rng = np.random.default_rng(12345)
    lam = p.legit_props_mean * rng.lognormal(0, p.legit_props_sigma, 200_000)
    props = rng.poisson(lam)
    rates = p.legit_rate_mean * rng.lognormal(0, p.legit_rate_sigma, 200_000)
    return dict(max_props=int(math.ceil(p.ceil_margin * np.quantile(props, 0.999))),
                r_lim=float(p.ceil_margin * np.quantile(rates, 0.999)),
                packet_cap=p.packet_cap, expiry_cap=p.expiry_cap, queue_cap=p.queue_cap)


def dconf(d):
    c = dict(b1=False, native=False, enf=False, v5=False, det=None, lofo=False, la=False)
    if d == "B1": c["b1"] = True
    elif d == "B2": c["det"] = "if"
    elif d == "B2_c10": c["det"] = "c10"
    elif d == "B3_seen": c["det"] = "rf"
    elif d == "B3_lofo": c["det"] = "rf"; c["lofo"] = True
    elif d == "B4": c["native"] = True
    elif d in SPEC:
        det, la, lofo = SPEC[d]
        c.update(native=True, enf=True, v5=True, det=det, la=la, lofo=lofo)
    elif d != "B0":
        raise ValueError(d)
    return c


def det_key(det, held):
    return f"rf_{held}" if det == "rf" else det


def tkey(det, held, k, lam):
    l = lam if det in PROF else 1.0
    return f"{det_key(det, held)}@{int(k)}@{float(l):.3f}"


def attackers(scn, B, p, lim, jit, M=None):
    z = np.zeros(B)
    a = dict(rate=z.copy(), props=np.ones(B), pb=np.full(B, p.legit_prop_bytes),
             recon=z.copy(), expiry=np.full(B, p.expiry_default), qrate=z.copy(),
             qlim=np.full(B, np.inf))
    if B == 0 or scn == "benign":
        return a
    fit_legit = max(1, math.floor((p.packet_cap - p.base_bytes) / p.legit_prop_bytes))
    if scn == "A1":
        a["recon"][:] = p.a1_recon * jit
    elif scn == "A2":
        a["rate"][:] = p.a2_rate * jit; a["props"][:] = p.a2_props; a["pb"][:] = p.a2_prop_bytes
    elif scn == "A2fit":      # fills whatever packet cap is configured, with tiny properties
        a["rate"][:] = p.a2_rate * jit
        a["props"][:] = max(1, math.floor((p.packet_cap - p.base_bytes) / p.pb_fit))
        a["pb"][:] = p.pb_fit
    elif scn == "A3":
        a["qrate"][:] = p.a3_qrate * jit
    elif scn == "A4_B1":
        a["rate"][:] = 0.9 * lim["r_lim"]; a["props"][:] = p.a2_props; a["pb"][:] = p.a2_prop_bytes
        a["recon"][:] = p.a1_recon; a["qrate"][:] = p.a3_qrate
    elif scn == "A4_enf":     # exactly at enforcer ceilings (and at the packet cap if that is tighter)
        a["rate"][:] = p.a2_rate; a["props"][:] = min(lim["max_props"], fit_legit)
        a["recon"][:] = p.a1_recon; a["expiry"][:] = lim["expiry_cap"]
        a["qrate"][:] = lim["queue_cap"]
    elif scn == "A4_bud":     # reads the calibrated budget of V5G+bud, stays a4_margin below it
        thr = M["thr"][tkey("bud", None, p.a4_ref_consec, 1.0)]
        capf = p.a4_margin * thr * M["ref"]["bud"]          # allowed value per feature
        props = max(1, math.floor(min(capf[1], (capf[2] - p.base_bytes) / p.pb_fit,
                                      lim["max_props"], (p.packet_cap - p.base_bytes) / p.pb_fit)))
        a["rate"][:] = capf[0]; a["props"][:] = props; a["pb"][:] = p.pb_fit
        a["recon"][:] = capf[4]; a["expiry"][:] = lim["expiry_cap"]
        a["qrate"][:] = lim["queue_cap"]; a["qlim"][:] = max(0.0, capf[3])
    return a


# --------------------------------------------------------------------------
# Detector scores (X: active clients x K)
# --------------------------------------------------------------------------
def score(det, X, M, held=None):
    if det == "z_orig":
        return ((X - X.mean(0)) / (X.std(0) + 1e-6)).max(1)
    if det == "rz":
        med = np.median(X, 0)
        mad = np.maximum(1.4826 * np.median(np.abs(X - med), 0), M["ref"]["floor"])
        return ((X - med) / mad).max(1)
    if det == "bud":
        return (X / M["ref"]["bud"]).max(1)
    if det in ("if", "c10"):
        return -M["if"].score_samples(X)
    if det == "rf":
        return M["rf"][held].predict_proba(X)[:, 1]
    raise ValueError(det)


# --------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------
def run_sim(scn, B, defense, seed, p, lim, M=None, mode="run", jit=1.0, trace=False):
    """mode: 'run' (defense active) | 'collect' (no blocking, returns window features)."""
    nl = p.n_legit; n = nl + B
    rng = np.random.default_rng([seed, SCNS.index(scn), B])
    dc = dconf(defense)
    lam = p.legit_props_mean * rng.lognormal(0, p.legit_props_sigma, nl)
    rate_l = p.legit_rate_mean * rng.lognormal(0, p.legit_rate_sigma, nl)
    recon_l = p.legit_recon * rng.lognormal(0, 0.3, nl)
    at = attackers(scn, B, p, lim, jit, M)
    is_att = np.r_[np.zeros(nl, bool), np.ones(B, bool)]
    rate = np.r_[rate_l, at["rate"]]; props = np.r_[lam, at["props"]]
    pb = np.r_[np.full(nl, p.legit_prop_bytes), at["pb"]]
    recon = np.r_[recon_l, at["recon"]]
    expiry_req = np.r_[np.full(nl, 5.0), at["expiry"]]
    qrate = np.r_[np.zeros(nl), at["qrate"]]
    qlim = np.r_[np.full(nl, np.inf), at["qlim"]]
    m2 = np.r_[lam + lam ** 2, props[nl:] ** 2]
    cf = p.c0 + p.c_prop * props + p.q_quad * m2

    native, enf = dc["native"], dc["enf"]
    exp_eff = np.minimum(expiry_req, lim["expiry_cap"]) if native else expiry_req
    qcap = lim["queue_cap"] if native else p.queue_default
    size_lim = np.maximum(0.0, np.floor((p.packet_cap - p.base_bytes) / pb))
    lim_p = np.full(n, np.inf)
    if native: lim_p = size_lim.copy()
    if enf: lim_p = np.minimum(lim_p, lim["max_props"])
    prej = np.r_[stats.poisson.sf(lim_p[:nl], lam), (props[nl:] > lim_p[nl:]).astype(float)]
    reason_size = bool(native) & (bool(not enf) | (size_lim <= lim["max_props"]))
    mp = lim["max_props"]
    if p.parse_mode == "full": pm = cf
    elif p.parse_mode == "header": pm = np.full(n, p.c_hdr)
    else: pm = np.full(n, p.c0 + p.c_prop * mp + p.q_quad * mp ** 2)
    rc = np.where(reason_size, p.c_hdr, pm)
    c_def = p.c_def_pkt if dc["v5"] else 0.0
    burst = lim["r_lim"]; tokens = np.full(n, burst)

    det = dc["det"]; held = None; thr = None
    if det == "rf": held = scn if (dc["lofo"] and scn in FAMS) else None
    if det is not None and det != "z_orig" and mode == "run":
        thr = M["thr"][tkey(det, held, p.consec, p.ewma_lam)]
    lam_s = p.ewma_lam if det in PROF else 1.0
    Xs_prev = None

    active = np.ones(n, bool); cnt_exc = np.zeros(n, int)
    t_act = np.zeros(n); q = np.zeros(n); blocked_t = np.full(n, np.nan)
    buffer = p.cap * p.buffer_s; backlog = 0.0; cpu_total = 0.0; mem_peak = 0.0
    sw = int(round(p.win / p.dt)); nsteps = int(round(p.T / p.dt))
    w_cnt = np.zeros(n); w_props = np.zeros(n); w_proc = 0.0; w_mem = 0.0
    F = []; tr = []
    S = dict(sent=0.0, blocked=0.0, b1=0.0, rej=0.0, dropbuf=0.0, deliv=0.0, ontime=0.0)
    lat_s, lat_w = [], []

    for s in range(nsteps):
        t = s * p.dt
        c = rng.poisson(rate * p.dt).astype(float)
        pp_l = rng.poisson(lam * c[:nl]).astype(float)
        offered = c * active
        if dc["b1"]:
            tokens = np.minimum(burst, tokens + lim["r_lim"] * p.dt)
            allowed = np.minimum(offered, np.floor(tokens)); tokens -= allowed
        else:
            allowed = offered
        dropb1 = offered - allowed
        rej = allowed * prej; acc = allowed - rej
        work = acc * cf + rej * rc + dropb1 * p.c_hdr + offered * c_def
        rec_ev = recon * p.dt * active
        work = work + rec_ev * p.c_conn
        dq = np.minimum(qrate * p.dt, np.maximum(0, np.minimum(qcap, qlim) - q)) * active
        q += dq; work = work + dq * p.c_q
        t_act += p.dt * active
        mem = (p.conn_bytes * active.sum() + q.sum() * p.msg_bytes
               + (recon * np.minimum(exp_eff, t_act + p.dt) * p.sess_bytes).sum()) / 1e6
        mem_peak = max(mem_peak, mem)
        cap_eff = p.cap / (1.0 + p.k_mem * max(0.0, mem / p.mem_budget_MB - 1.0))
        arrived = work.sum()
        f = 1.0 if arrived <= 0 else min(1.0, (max(0.0, buffer - backlog) + cap_eff * p.dt) / arrived)
        lat = (backlog + 0.5 * f * arrived) / cap_eff
        done = min(backlog + f * arrived, cap_eff * p.dt)
        backlog = max(0.0, backlog + f * arrived - cap_eff * p.dt)
        cpu_total += done; w_proc += done / (p.cap * p.dt); w_mem += mem
        sent = c[:nl].sum(); S["sent"] += sent
        S["blocked"] += (c[:nl] * np.logical_not(active[:nl])).sum(); S["b1"] += dropb1[:nl].sum()
        S["rej"] += rej[:nl].sum()
        a_l = acc[:nl].sum(); dl = a_l * f
        S["dropbuf"] += a_l - dl; S["deliv"] += dl
        if lat <= p.deadline: S["ontime"] += dl
        if dl > 0: lat_s.append(lat); lat_w.append(dl)
        w_cnt += offered
        w_props[:nl] += pp_l * active[:nl]; w_props[nl:] += offered[nl:] * props[nl:]
        if (s + 1) % sw == 0:
            with np.errstate(invalid="ignore", divide="ignore"):
                pm_ = np.where(w_cnt > 0, w_props / w_cnt, 0.0)
            by_ = np.where(w_cnt > 0, p.base_bytes + pm_ * pb, 0.0)
            ql = np.r_[np.maximum(0, 1 + rng.normal(0, 0.3, nl)), q[nl:]]
            X = np.c_[w_cnt / p.win, pm_, by_, ql, recon * active]
            if mode == "collect":
                F.append(X.copy())
            elif det is not None:
                if lam_s < 1.0:                                   # EWMA, Eq.(2)
                    Xs_prev = X if Xs_prev is None else lam_s * X + (1 - lam_s) * Xs_prev
                    Xu = Xs_prev
                else:
                    Xu = X
                ai = np.flatnonzero(active)
                if len(ai) > 2:
                    Xa = Xu[ai]
                    if p.c_ml_client and det in ("if", "c10", "rf"):
                        backlog += len(ai) * p.c_ml_client; cpu_total += len(ai) * p.c_ml_client
                    sc = score(det, Xa, M, held)
                    load = p.w_alpha * (w_mem / sw) / p.mem_budget_MB + (1 - p.w_alpha) * (w_proc / sw)
                    fac = max(p.tau_min, p.tau0 - p.gamma * load / p.rho)      # tau(t), Eq.(4)
                    if det == "z_orig":
                        tau = fac if dc["la"] else p.tau0
                    else:
                        tau = thr * fac / p.tau0 if dc["la"] else thr
                    if trace and det == "z_orig":
                        am = is_att[ai]
                        Zf = (Xa - Xa.mean(0)) / (Xa.std(0) + 1e-6)
                        zg = float(Zf[am].mean(0).max()) if am.any() else np.nan
                        tr.append(dict(t=t + p.dt, zgroup=zg, maxz_att=float(sc[am].max()) if am.any() else np.nan,
                                       maxz_leg=float(sc[~am].max()) if (~am).any() else np.nan,
                                       p_att=float(am.mean()), tau=float(tau)))
                    ex = sc > tau
                    cnt_exc[ai] = np.where(ex, cnt_exc[ai] + 1, 0)
                    blk = ai[cnt_exc[ai] >= p.consec]
                    if len(blk):
                        active[blk] = False; blocked_t[blk] = t + p.dt
            w_cnt[:] = 0; w_props[:] = 0; w_proc = 0.0; w_mem = 0.0

    if mode == "collect":
        return np.array(F), is_att
    ls = np.array(lat_s); lw = np.array(lat_w)
    def wq(qq):
        if len(ls) == 0: return np.nan
        o = np.argsort(ls); cs = np.cumsum(lw[o]); return float(ls[o][np.searchsorted(cs, qq * cs[-1])])
    sent = max(S["sent"], 1e-9)
    bl = ~np.isnan(blocked_t)
    ttd90 = float(np.quantile(np.where(bl[nl:], blocked_t[nl:], p.T), 0.9)) if B else np.nan
    return dict(avail_deadline=100 * S["ontime"] / sent, avail_any=100 * S["deliv"] / sent,
                fail_blocked=100 * S["blocked"] / sent, fail_b1=100 * S["b1"] / sent,
                fail_reject=100 * S["rej"] / sent, fail_dropbuf=100 * S["dropbuf"] / sent,
                fail_late=100 * (S["deliv"] - S["ontime"]) / sent,
                lat_p50_ms=1000 * wq(0.5), lat_p99_ms=1000 * wq(0.99),
                cpu_total=cpu_total, cpu_util=cpu_total / (p.cap * p.T), mem_peak_MB=mem_peak,
                tpr=100 * bl[nl:].mean() if B else np.nan, fpr=100 * bl[:nl].mean(),
                ttd=float(np.nanmin(blocked_t[nl:])) if B and bl[nl:].any() else p.T,
                ttd90=ttd90, trace=tr if trace else None)


# --------------------------------------------------------------------------
# Models: warm-up data, training, calibration (all per seed, disjoint seeds)
# --------------------------------------------------------------------------
def run_scores(S, k):
    W = S.shape[0]
    if W < k: return S.max(0)
    return np.min(np.stack([S[i:W - k + 1 + i] for i in range(k)]), 0).max(0)


def smooth_seq(Fw, lam):
    out = np.empty_like(Fw); out[0] = Fw[0]
    for w in range(1, Fw.shape[0]):
        out[w] = lam * Fw[w] + (1 - lam) * out[w - 1]
    return out


def build_models(p, seed, lim):
    base = 10_000 + seed * 100
    rng = np.random.default_rng(base)
    ben = [run_sim("benign", 0, "B0", base + i, p, lim, mode="collect")[0]
           for i in range(p.nb_train + p.nb_cal)]
    tr_b, cal = ben[:p.nb_train], ben[p.nb_train:]
    Xb = np.vstack([f.reshape(-1, K) for f in tr_b])
    ref = dict(floor=np.maximum(0.25 * Xb.std(0), 1e-3), bud=np.maximum(np.quantile(Xb, 0.99, axis=0), 1e-3))
    sub = Xb[rng.choice(len(Xb), min(20_000, len(Xb)), replace=False)]
    iso = IsolationForest(n_estimators=50, random_state=seed, n_jobs=1).fit(sub)
    M = dict(ref=ref, **{"if": iso})
    rows, ys, fam = [Xb], [np.zeros(len(Xb))], [np.array(["benign"] * len(Xb))]
    for j, fm in enumerate(FAMS):
        for k, Bw in enumerate(p.warm_B):
            Fw, ia = run_sim(fm, Bw, "B0", base + 50 + 10 * j + k, p, lim, mode="collect",
                             jit=float(rng.lognormal(0, 0.25)))
            Xw = Fw.reshape(-1, K); yw = np.tile(ia.astype(float), Fw.shape[0])
            rows.append(Xw); ys.append(yw); fam.append(np.array([fm] * len(Xw)))
    Xa, ya, fa = np.vstack(rows), np.concatenate(ys), np.concatenate(fam)
    M["rf"] = {}
    for held in [None] + FAMS:
        m = fa != held if held else np.ones(len(fa), bool)
        idx = np.flatnonzero(m); idx = rng.choice(idx, min(40_000, len(idx)), replace=False)
        M["rf"][held] = RandomForestClassifier(50, max_depth=12, class_weight="balanced",
                                               n_jobs=1, random_state=seed).fit(Xa[idx], ya[idx])
    # calibration: every detector, every confirmation window k, every lambda, at client-level FPR alpha
    thr = {}
    def wscores(det, Fw, held, lm):
        Fs = smooth_seq(Fw, lm) if (det in PROF and lm < 1.0) else Fw
        W, N, _ = Fs.shape
        if det in ("if", "rf", "bud"):
            return score(det, Fs.reshape(-1, K), M, held).reshape(W, N)
        return np.array([score(det, Fs[w], M, held) for w in range(W)])
    def calibrate(det, held, lams):
        for lm in lams:
            Ss = [wscores(det, Fw, held, lm) for Fw in cal]
            for k in p.consec_set:
                thr[tkey(det, held, k, lm)] = float(np.quantile(
                    np.concatenate([run_scores(S, k) for S in Ss]), 1 - p.alpha))
    calibrate("if", None, (1.0,))
    calibrate("rz", None, lam_set(p))
    calibrate("bud", None, lam_set(p))
    for held in [None] + FAMS: calibrate("rf", held, (1.0,))
    c10 = float(np.quantile(-iso.score_samples(sub), 0.90))
    for k in p.consec_set: thr[tkey("c10", None, k, 1.0)] = c10
    M["thr"] = thr
    return M


# --------------------------------------------------------------------------
# Jobs / worker
# --------------------------------------------------------------------------
def build_jobs(p, exps):
    """job = (scn, B, defense, exp, knob, val, p_overrides)"""
    J = []
    if "matrix" in exps:
        J += [("benign", 0, d, "matrix", "", 0, {}) for d in DEFS_ALL]
        J += [(s, p.B_matrix, d, "matrix", "", 0, {}) for s in MATRIX_SCN for d in DEFS_ALL]
    if "sweep" in exps:
        for s, Bs in (("A2", p.B_sweep), ("A4_enf", p.B_sweep), ("A4_bud", p.B_sybil)):
            J += [(s, b, d, "sweep", "B", b, {}) for b in Bs for d in DEFS_SWEEP]
    if "cap" in exps:      # R1
        cs = [("benign", 0)] + [(s, b) for s in ("A2", "A2fit", "A4_enf") for b in p.cap_B]
        for s, b in cs:
            for cap in p.cap_sweep:
                for d in ("B0", "B4", "V5G"):
                    J.append((s, b, d, "cap", "packet_cap", float(cap), {"packet_cap": float(cap)}))
    if "confirm" in exps:  # R2
        cs = [("benign", 0), ("A1", p.B_matrix), ("A2", p.B_matrix),
              ("A4_enf", p.B_hi), ("A4_bud", p.B_syb)]
        for s, b in cs:
            for d in ("B0", "V5G"):                       # references, default consec
                J.append((s, b, d, "confirm", "consec", float(p.consec), {}))
            for d in DEFS_CONF_DET:
                for k in p.consec_set:
                    J.append((s, b, d, "confirm", "consec", float(k), {"consec": int(k)}))
    if "sens" in exps:     # R3
        cs = [("benign", 0), ("A4_enf", p.B_hi), ("A4_bud", p.B_syb)]
        for knob, vals in p.sens:
            for v in vals:
                for s, b in cs:
                    for d in ("V5G+z", "V5G+bud+LA"):
                        J.append((s, b, d, "sens", knob, float(v), {knob: float(v)}))
    return J


def seed_worker(args):
    seed, pd_, lim, exps = args
    p = P(**pd_)
    M = build_models(p, seed, lim)
    out = []
    for scn, B, d, exp, knob, val, pov in build_jobs(p, exps):
        pj = replace(p, **pov)
        r = run_sim(scn, B, d, seed, pj, lim, M)
        r.pop("trace")
        r.update(seed=seed, scn=scn, B=B, defense=d, exp=exp, knob=knob, val=val,
                 consec=pj.consec, packet_cap=pj.packet_cap, tau0=pj.tau0, rho=pj.rho,
                 gamma=pj.gamma, w_alpha=pj.w_alpha, ewma_lam=pj.ewma_lam)
        out.append(r)
    return out


# --------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------
def ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2: return (np.nan, np.nan)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    return x.mean(), h


def fmt(x, d=1):
    m, h = ci(x); return "nan" if np.isnan(m) else f"{m:.{d}f}+-{h:.{d}f}"


def sel(df, col, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items(): m &= (df[k] == v).values
    return df[m].sort_values("seed").set_index("seed")[col]


def paired(df, col, scn, B, d, ref="B0", rel=True):
    a = df[(df.scn == scn) & (df.B == B) & (df.defense == d)].sort_values("seed")[col].values
    b = df[(df.scn == scn) & (df.B == B) & (df.defense == ref)].sort_values("seed")[col].values
    diff = 100 * (a - b) / b if rel else a - b
    return ci(diff)


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------
def report_matrix(df, p, out):
    m = df[df.exp == "matrix"]
    if m.empty: return
    rows = []
    b0b = m[(m.scn == "benign") & (m.defense == "B0")]
    for (s, d), g in m.groupby(["scn", "defense"]):
        rows.append(dict(scenario=s, defense=d, B=g.B.iloc[0],
                         avail_deadline=fmt(g.avail_deadline), avail_any=fmt(g.avail_any),
                         lat_p50_ms=fmt(g.lat_p50_ms), lat_p99_ms=fmt(g.lat_p99_ms),
                         cpu_util=fmt(100 * g.cpu_util), mem_peak_MB=fmt(g.mem_peak_MB),
                         tpr=fmt(g.tpr), fpr=fmt(g.fpr), legit_rejected=fmt(g.fail_reject, 2),
                         ttd=fmt(g.ttd), ttd90=fmt(g.ttd90),
                         cpu_x_benign=round(g.cpu_total.mean() / b0b.cpu_total.mean(), 1),
                         mem_x_benign=round(g.mem_peak_MB.mean() / b0b.mem_peak_MB.mean(), 1)))
    T = pd.DataFrame(rows); T.to_csv(f"{out}/table_matrix.csv", index=False)
    print("\n=== ATTACK MATRIX (mean+-95%CI over seeds; deadline availability) ===")
    print(T[["scenario", "defense", "avail_deadline", "lat_p99_ms", "cpu_x_benign", "mem_x_benign", "tpr", "fpr"]].to_string(index=False))
    pr = []
    for s in MATRIX_SCN:
        for d in DEFS_ALL:
            if d == "B0": continue
            ca = paired(m, "cpu_total", s, p.B_matrix, d)
            cb = paired(m, "avail_deadline", s, p.B_matrix, d, rel=False)
            la = paired(m, "lat_p99_ms", s, p.B_matrix, d)
            pr.append(dict(scenario=s, defense=d, d_cpu_pct=f"{ca[0]:.1f}+-{ca[1]:.1f}",
                           d_avail_pp=f"{cb[0]:.1f}+-{cb[1]:.1f}", d_p99_pct=f"{la[0]:.1f}+-{la[1]:.1f}"))
    pd.DataFrame(pr).to_csv(f"{out}/table_paired_vs_B0.csv", index=False)
    ov = []
    for d in DEFS_ALL:
        if d == "B0": continue
        for col in ("cpu_total", "mem_peak_MB"):
            mu, h = paired(m, col, "benign", 0, d)
            ov.append(dict(defense=d, metric=col, overhead_pct=f"{mu:.2f}", ci95=f"[{mu-h:.2f},{mu+h:.2f}]",
                           equivalent_within_pm=f"{p.equiv_margin_pct}%: {'YES' if (mu-h > -p.equiv_margin_pct and mu+h < p.equiv_margin_pct) else 'NO / inconclusive'}"))
    pd.DataFrame(ov).to_csv(f"{out}/table_benign_overhead.csv", index=False)
    print("\n=== BENIGN OVERHEAD vs B0 (paired, 95% CI, equivalence margin) ===")
    print(pd.DataFrame(ov).to_string(index=False))
    nz = m.groupby(["scn", "defense"]).agg(cv_cpu_pct=("cpu_total", lambda x: 100 * x.std(ddof=1) / x.mean()),
                                           cv_mem_pct=("mem_peak_MB", lambda x: 100 * x.std(ddof=1) / x.mean())).round(1)
    nz.to_csv(f"{out}/table_variability_same_scenario.csv")


def report_cap(df, p, out):
    """R1: packet-cap sweep."""
    c = df[df.exp == "cap"]
    if c.empty: return
    rows = []
    for (s, B, d, cap), g in c.groupby(["scn", "B", "defense", "packet_cap"]):
        rows.append(dict(scenario=s, B=int(B), defense=d, packet_cap=int(cap),
                         avail_deadline=fmt(g.avail_deadline), legit_false_reject_pct=fmt(g.fail_reject, 2),
                         lat_p99_ms=fmt(g.lat_p99_ms), cpu_util_pct=fmt(100 * g.cpu_util),
                         avail_mean=g.avail_deadline.mean(), fr_mean=g.fail_reject.mean()))
    T = pd.DataFrame(rows)
    T.drop(columns=["avail_mean", "fr_mean"]).to_csv(f"{out}/table_cap_sweep.csv", index=False)
    vr = []
    for (d, cap), g in T.groupby(["defense", "packet_cap"]):
        atk = g[g.scenario != "benign"]; ben = g[g.scenario == "benign"]
        w = atk.loc[atk.avail_mean.idxmin()]
        fr = float(ben.fr_mean.iloc[0]) if len(ben) else np.nan
        vr.append(dict(defense=d, packet_cap=cap, worst_attack_avail=round(w.avail_mean, 1),
                       worst_case=f"{w.scenario}@B{w.B}", benign_false_reject_pct=round(fr, 2),
                       ok=bool(w.avail_mean >= p.ok_avail and fr <= p.ok_fr)))
    V = pd.DataFrame(vr); V.to_csv(f"{out}/table_cap_verdict.csv", index=False)
    print("\n=== R1: PACKET-CAP SWEEP (worst-case over A2, A2fit, A4_enf and B; false reject from benign run) ===")
    print(V.to_string(index=False))
    for d in ("B4", "V5G"):
        okc = V[(V.defense == d) & V.ok].packet_cap.tolist()
        print(f"{d}: meets worst-case avail >= {p.ok_avail}% and benign false-reject <= {p.ok_fr}% at caps: {okc or 'none'}")
    b4 = V[(V.defense == "B4") & V.ok].packet_cap.tolist()
    print("=> " + ("a tight packet cap alone (no user-property cap) is sufficient in this simulator: the claim 'the property "
                   "cap does the decisive work' must be restated as 'a tight size bound does'." if b4 else
                   "B4 never meets the criteria: the claim survives in this simulator."))
    # figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    Bs = sorted(set(T[T.scenario != "benign"].B)); fig, ax = plt.subplots(1, len(Bs) + 1, figsize=(5 * (len(Bs) + 1), 3.8))
    cols = {"A2": "C0", "A2fit": "C1", "A4_enf": "C2"}; ls = {"B0": ":", "B4": "-", "V5G": "--"}
    for a, B in zip(ax, Bs):
        for s in cols:
            for d in ls:
                g = T[(T.scenario == s) & (T.B == B) & (T.defense == d)].sort_values("packet_cap")
                a.plot(g.packet_cap, g.avail_mean, ls[d], color=cols[s], marker="o", ms=3, label=f"{s} {d}")
        a.set_xscale("log", base=2); a.set_title(f"On-time availability, B={B} (%)", fontsize=9); a.set_xlabel("max_packet_size (B)")
    ax[0].legend(fontsize=5)
    for d in ("B4", "V5G"):
        g = T[(T.scenario == "benign") & (T.defense == d)].sort_values("packet_cap")
        ax[-1].plot(g.packet_cap, g.fr_mean, marker="o", label=d)
    ax[-1].set_xscale("log", base=2); ax[-1].set_title("Legit false rejection, benign (%)", fontsize=9)
    ax[-1].set_xlabel("max_packet_size (B)"); ax[-1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{out}/fig_cap_sweep.png", dpi=150); plt.close(fig)


def report_confirm(df, p, out):
    """R2: confirmation-window sweep for all detectors, with paired loss vs the no-detector reference."""
    c = df[df.exp == "confirm"]
    if c.empty: return
    rows = []
    for (s, B, d, k), g in c.groupby(["scn", "B", "defense", "val"]):
        ref = "V5G" if d.startswith("V5G") else "B0"
        a = sel(c, "avail_deadline", scn=s, B=B, defense=d, val=k)
        r = sel(c, "avail_deadline", scn=s, B=B, defense=ref, val=float(p.consec))
        dm, dh = ci((a - r).values) if d not in ("B0", "V5G") else (np.nan, np.nan)
        rows.append(dict(scenario=s, B=int(B), defense=d, consec=int(k), avail_deadline=fmt(g.avail_deadline),
                         d_avail_vs_ref_pp=("" if np.isnan(dm) else f"{dm:.1f}+-{dh:.1f}"), ref=ref,
                         lost_late_pct=fmt(g.fail_late), ttd_s=fmt(g.ttd), ttd90_s=fmt(g.ttd90),
                         tpr=fmt(g.tpr), fpr=fmt(g.fpr), lat_p99_ms=fmt(g.lat_p99_ms),
                         avail_mean=g.avail_deadline.mean()))
    T = pd.DataFrame(rows); T.drop(columns="avail_mean").to_csv(f"{out}/table_confirm_sweep.csv", index=False)
    print("\n=== R2: CONFIRMATION-WINDOW SWEEP (on-time availability %, mean over seeds) ===")
    for (s, B), g in T.groupby(["scenario", "B"]):
        pv = g.pivot_table(index="defense", columns="consec", values="avail_mean").round(1)
        print(f"\n-- {s} B={B} --"); print(pv.to_string())
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    scs = list(dict.fromkeys(zip(T.scenario, T.B)))
    fig, axs = plt.subplots(1, len(scs), figsize=(4.2 * len(scs), 3.8))
    for ax, (s, B) in zip(np.atleast_1d(axs), scs):
        for d in DEFS_CONF_DET:
            g = T[(T.scenario == s) & (T.B == B) & (T.defense == d)].sort_values("consec")
            ax.plot(g.consec, g.avail_mean, marker="o", ms=3, label=d)
        for d, ls_ in (("B0", ":"), ("V5G", "--")):
            g = T[(T.scenario == s) & (T.B == B) & (T.defense == d)]
            if len(g): ax.axhline(g.avail_mean.iloc[0], ls=ls_, color="k", lw=0.8, label=f"{d} (ref)")
        ax.set_title(f"{s}, B={B}", fontsize=9); ax.set_xlabel("confirmation windows k"); ax.set_xticks(list(p.consec_set))
    np.atleast_1d(axs)[0].set_ylabel("on-time availability (%)"); np.atleast_1d(axs)[0].legend(fontsize=5)
    fig.tight_layout(); fig.savefig(f"{out}/fig_confirm_sweep.png", dpi=150); plt.close(fig)


def report_ablation(df, p, out):
    """R3: load-aware tau(t) vs fixed threshold (paired, per seed)."""
    pairs = [("V5G+z", "V5G+zfix"), ("V5G+bud+LA", "V5G+bud")]
    cases = [("matrix", "benign", 0)] + [("matrix", s, p.B_matrix) for s in MATRIX_SCN]
    cases += [("sweep", s, b) for s, Bs in (("A4_enf", p.B_sweep), ("A4_bud", p.B_sybil)) for b in Bs if b > 0]
    rows = []
    for exp, s, b in cases:
        e = df[df.exp == exp]
        for la, fx in pairs:
            a_ = sel(e, "avail_deadline", scn=s, B=b, defense=la); f_ = sel(e, "avail_deadline", scn=s, B=b, defense=fx)
            if len(a_) == 0 or len(f_) == 0: continue
            r = dict(exp=exp, scenario=s, B=b, load_aware=la, fixed=fx)
            for col, nm in (("avail_deadline", "d_avail_pp"), ("fpr", "d_fpr_pp"), ("tpr", "d_tpr_pp"), ("cpu_total", "d_cpu_pct")):
                x = sel(e, col, scn=s, B=b, defense=la); y = sel(e, col, scn=s, B=b, defense=fx)
                dd = (x - y) if col != "cpu_total" else 100 * (x - y) / y
                m_, h = ci(dd.values); r[nm] = "nan" if np.isnan(m_) else f"{m_:.2f}+-{h:.2f}"
            rows.append(r)
    if not rows: return
    T = pd.DataFrame(rows); T.to_csv(f"{out}/table_ablation_loadaware.csv", index=False)
    print("\n=== R3: LOAD-AWARE tau(t) minus FIXED tau (paired; + means load-aware is better for avail) ===")
    print(T.to_string(index=False))


def report_sens(df, p, out):
    """R3: one-at-a-time sensitivity."""
    s = df[df.exp == "sens"]
    if s.empty: return
    rows = []
    for (k, v, d, sc, B), g in s.groupby(["knob", "val", "defense", "scn", "B"]):
        rows.append(dict(knob=k, val=v, defense=d, scenario=sc, B=int(B), avail_deadline=fmt(g.avail_deadline),
                         tpr=fmt(g.tpr), fpr=fmt(g.fpr), legit_blocked=fmt(g.fail_blocked),
                         avail_mean=g.avail_deadline.mean(), hw=ci(g.avail_deadline)[1]))
    T = pd.DataFrame(rows)
    T.drop(columns=["avail_mean", "hw"]).to_csv(f"{out}/table_sensitivity.csv", index=False)
    rk = T.groupby(["knob", "defense", "scenario"]).agg(avail_range_pp=("avail_mean", lambda x: x.max() - x.min()),
                                                        max_ci_hw=("hw", "max")).reset_index()
    rk["above_noise"] = rk.avail_range_pp > 2 * rk.max_ci_hw
    rk = rk.sort_values("avail_range_pp", ascending=False).round(2)
    rk.to_csv(f"{out}/table_sensitivity_ranking.csv", index=False)
    print("\n=== R3: SENSITIVITY RANKING (range of mean availability over the grid; above_noise = range > 2x CI half-width) ===")
    print(rk.to_string(index=False))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    knobs = [k for k, _ in p.sens]; fig, ax = plt.subplots(1, len(knobs), figsize=(3.6 * len(knobs), 3.6))
    for a, k in zip(np.atleast_1d(ax), knobs):
        for d in ("V5G+z", "V5G+bud+LA"):
            for sc in ("benign", "A4_enf", "A4_bud"):
                g = T[(T.knob == k) & (T.defense == d) & (T.scenario == sc)].sort_values("val")
                a.errorbar(g.val, g.avail_mean, yerr=g.hw, marker="o", ms=3, capsize=2, label=f"{d} {sc}")
        a.set_title(k, fontsize=9); a.set_xlabel(k)
    np.atleast_1d(ax)[0].set_ylabel("on-time availability (%)"); np.atleast_1d(ax)[0].legend(fontsize=5)
    fig.tight_layout(); fig.savefig(f"{out}/fig_sensitivity.png", dpi=150); plt.close(fig)


def plots(df, p, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sw = df[df.exp == "sweep"]
    if sw.empty: return
    for scn in ("A2", "A4_enf", "A4_bud"):
        fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
        for d in DEFS_SWEEP:
            g = sw[(sw.scn == scn) & (sw.defense == d)].groupby("B")
            for a, col, sc in zip(ax, ["avail_deadline", "lat_p99_ms", "cpu_util", "mem_peak_MB"], [1, 1, 100, 1]):
                mu = g[col].mean() * sc; h = g[col].apply(lambda x: ci(x)[1]) * sc
                a.errorbar(mu.index, mu.values, yerr=h.values, marker="o", ms=3, capsize=2, label=d)
        for a, t in zip(ax, ["Legit availability within deadline (%)", "Legit p99 latency (ms)", "Broker CPU util (%)", "Peak memory (MB)"]):
            a.set_title(t, fontsize=9); a.set_xlabel("attacker nodes B"); a.set_xscale("symlog", linthresh=5)
        ax[1].set_yscale("log"); ax[0].legend(fontsize=5)
        fig.suptitle(f"Scenario {scn}: mean +- 95% CI over {df.seed.nunique()} seeds"); fig.tight_layout()
        fig.savefig(f"{out}/fig_sweep_{scn}.png", dpi=150); plt.close(fig)
    T = sw.groupby(["scn", "defense", "B"]).agg(avail=("avail_deadline", "mean"), tpr=("tpr", "mean"),
                                                fpr=("fpr", "mean"), cpu_util=("cpu_util", "mean")).round(2).reset_index()
    T.to_csv(f"{out}/table_sweep.csv", index=False)


def bound_plot(p, lim, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    M = None
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8)); rows = []
    for k, scn in enumerate(("A2", "A4_enf")):
        for B in (5, 20, 80, 160, 320):
            r = run_sim(scn, B, "V5G+z", 0, p, lim, M, trace=True)
            tr = pd.DataFrame(r["trace"]); tr["B"] = B
            pm = B / (B + p.n_legit); bound = math.sqrt((1 - pm) / pm)
            ax[k].plot(tr.t, tr.zgroup, label=f"B={B} group-mean z (bound {bound:.2f})")
            ax[k].plot(tr.t, tr.maxz_att, ls=":", color=ax[k].lines[-1].get_color())
            rows.append(dict(scn=scn, B=B, p=round(pm, 3), bound=round(bound, 2),
                             max_group_mean_z=round(tr.zgroup.max(), 2), max_single_attacker_z=round(tr.maxz_att.max(), 2),
                             tau_min_seen=round(tr.tau.min(), 2), tpr=r["tpr"],
                             group_bound_holds=bool((tr.zgroup.dropna() <= np.sqrt((1 - tr.p_att) / tr.p_att)[tr.zgroup.notna()] + 1e-6).all())))
        ax[k].axhline(p.tau0, color="k", ls="--", lw=1, label="tau0"); ax[k].set_title(f"{scn}: z-score vs bound")
        ax[k].set_xlabel("time (s)"); ax[k].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(f"{out}/fig_bound_validation.png", dpi=150); plt.close(fig)
    T = pd.DataFrame(rows); T.to_csv(f"{out}/table_bound_validation.csv", index=False)
    print("\n=== BOUND VALIDATION ===")
    print(T.to_string(index=False))


def microbench(n=200_000):
    import random
    N, Kf, lam = 200, K, 0.2
    xb = [[0.0] * Kf for _ in range(N)]; s1 = [0.0] * Kf; s2 = [0.0] * Kf
    cl = [random.randrange(N) for _ in range(n)]; vals = [random.random() for _ in range(n)]
    t0 = time.perf_counter()
    for i in range(n):
        c = cl[i]; row = xb[c]; v = vals[i]
        for k in range(Kf):
            old = row[k]; new = lam * v + (1 - lam) * old; row[k] = new
            s1[k] += new - old; s2[k] += new * new - old * old
    return 1e9 * (time.perf_counter() - t0) / n


def write_params(p, lim, out):
    rows = [dict(parameter=k, value=str(v)) for k, v in asdict(p).items()]
    rows += [dict(parameter=f"derived_{k}", value=str(v)) for k, v in lim.items()]
    pd.DataFrame(rows).to_csv(f"{out}/params.csv", index=False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--quick", action="store_true", help="2 seeds, T=20s, reduced grids")
    ap.add_argument("--exps", default="matrix,sweep,cap,confirm,sens")
    ap.add_argument("--parse-mode", choices=["full", "partial", "header"], default="partial")
    a, _ = ap.parse_known_args(argv)
    p = P(parse_mode=a.parse_mode)
    if a.quick:
        p.T = 20.0; p.B_sweep = (0, 20, 80, 320); p.B_sybil = (0, 320, 1000, 1280)
        p.warm_B = (20, 80); p.nb_train, p.nb_cal = 3, 4
        p.cap_sweep = (256, 1024, 4096, 8192); p.cap_B = (40,); p.consec_set = (1, 3, 5)
        p.sens = (("w_alpha", (0.0, 1.0)), ("ewma_lam", (0.2, 1.0)), ("tau0", (2.0, 4.0)),
                  ("rho", (0.3, 0.9)), ("gamma", (0.0, 2.0)))
    seeds = a.seeds or (2 if a.quick else 10)
    exps = set(a.exps.split(","))
    assert p.consec in p.consec_set and p.a4_ref_consec in p.consec_set
    os.makedirs(a.out, exist_ok=True)
    lim = derive_limits(p); write_params(p, lim, a.out)
    print("derived ceilings:", lim)
    print(f"jobs per seed: {len(build_jobs(p, exps))}  seeds: {seeds}  workers: {a.workers}")
    t0 = time.time(); pdict = asdict(p)
    tasks = [(s, pdict, lim, exps) for s in range(seeds)]
    if a.workers > 1 and seeds > 1:
        with ProcessPoolExecutor(min(a.workers, seeds)) as ex: res = list(ex.map(seed_worker, tasks))
    else:
        res = [seed_worker(t) for t in tasks]
    df = pd.DataFrame([r for rr in res for r in rr]); df.to_csv(f"{a.out}/raw_runs.csv", index=False)
    print(f"{len(df)} runs in {time.time()-t0:.0f}s")
    report_matrix(df, p, a.out); report_cap(df, p, a.out); report_confirm(df, p, a.out)
    report_ablation(df, p, a.out); report_sens(df, p, a.out); plots(df, p, a.out)
    bound_plot(p, lim, a.out)
    ns = microbench(); print(f"\nPure-Python profiler update: {ns:.0f} ns/packet (Python-level; not a broker measurement)")
    json.dump(dict(profiler_py_ns_per_packet=ns, seeds=seeds), open(f"{a.out}/misc.json", "w"))
    print("outputs in", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
