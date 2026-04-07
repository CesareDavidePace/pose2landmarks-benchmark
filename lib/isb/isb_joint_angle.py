"""
ISB joint‑angle calculator + gait‑cycle extraction & aggregation
-----------------------------------------------------------------

Main features added (v2):
1. **detect_gait_cycles(df, side='R', prominence=8, min_separation=40)**
   ‑ trova gli indici dei potenziali *heel‑strike* localizzando i minimi di
     ginocchio flessione/estensione (colonna `knee_flexext`).
2. **slice_and_normalise(df, events)**
   ‑ da ogni coppia di eventi consecutivi ricava un ciclo, lo interpola a 101
     punti (0‑100 % del passo) restituendo una matrice (n_cycle, 101, n_cols).
3. **aggregate_cycles(cycles)**
   ‑ calcola media, deviazione standard e li restituisce come `DataFrame` a
     101 campioni.
4. **plot_cycles_mean_sd(agg_df, joint='knee')**
   ‑ grafico con banda ±1 SD.
5. **process_dataset(trc_paths, side='R', save_csv=None, plot=True)**
   ‑ pipeline completa su una lista di file .trc; opzionalmente salva la
     matrice normalizzata di tutti i cicli.

La parte precedente (costruzione CS, angoli, CLI) resta invariata.
"""

from __future__ import annotations

import argparse
import os
import pathlib
from typing import Dict, List, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from lib.utils.trc_parser import read_trc_file
from ..utils.markers_names_move4d import markers_names


###############################################################################
# Helper maths utilities                                                       #
###############################################################################

def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


def _orthonormalise(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt


def _rot_from_axes(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    R = np.stack([x, y, z], axis=-1)
    return _orthonormalise(R)


###############################################################################
# Euler helpers                                                                #
###############################################################################

def _euler_zyx(R: np.ndarray) -> np.ndarray:  # Z‑Y‑X
    beta = np.arcsin(-R[..., 2, 0])
    alpha = np.arctan2(R[..., 2, 1], R[..., 2, 2])
    gamma = np.arctan2(R[..., 1, 0], R[..., 0, 0])
    return np.rad2deg(np.stack([alpha, beta, gamma], axis=-1))


def _euler_xyz(R: np.ndarray) -> np.ndarray:  # X‑Y‑Z
    beta = np.arcsin(-R[..., 0, 2])
    alpha = np.arctan2(R[..., 1, 2], R[..., 2, 2])
    gamma = np.arctan2(R[..., 0, 1], R[..., 0, 0])
    return np.rad2deg(np.stack([alpha, beta, gamma], axis=-1))


###############################################################################
# ISBJointAngleCalculator (unchanged interface, internals refactored)          #
###############################################################################

class ISBJointAngleCalculator:
    def __init__(self, marker_data: Dict[str, np.ndarray], side: str = "R"):
        self.m = marker_data
        self.side = side.upper()
        assert self.side in {"R", "L"}
        self.nf = next(iter(marker_data.values())).shape[0]

    # ---------------- pelvis --------------------------------------------------

    def _pelvis_cs(self):
        la, ra = self.m["Lt_ASIS"], self.m["Rt_ASIS"]
        lps, rps = self.m["Lt_PSIS"], self.m["Rt_PSIS"]
        origin = (la + ra) / 2
        z = _unit(ra - la)
        ps_mid = (lps + rps) / 2
        x_tmp = ps_mid - origin
        x = _unit(np.cross(x_tmp, z))
        y = _unit(np.cross(z, x))
        R = _rot_from_axes(x, y, z)
        return origin, R

    # --------------- hip centres (Davis) -------------------------------------
    def _hip_joint_centres(self, O, R):
        la, ra = self.m["Lt_ASIS"], self.m["Rt_ASIS"]
        width = np.linalg.norm(ra - la, axis=1)
        C = np.array([-0.24, 0.30, 0.33])
        offs = np.array([-9.9, -10.9, 7.3])
        v = np.empty((self.nf, 3))
        for i in range(3):
            v[:, i] = C[i] * width + offs[i]
        vR = np.stack([-v[:, 0], -v[:, 1], -v[:, 2]], axis=-1)
        vL = np.stack([v[:, 0], -v[:, 1], -v[:, 2]], axis=-1)
        Rt = np.transpose(R, (0, 2, 1))
        hjcR = O + (Rt @ vR[..., None]).squeeze(-1)
        hjcL = O + (Rt @ vL[..., None]).squeeze(-1)
        return hjcL, hjcR

    # --------------- segment CS helpers --------------------------------------
    def _thigh_cs(self, hip):
        side = "Rt" if self.side == "R" else "Lt"
        lat, med = self.m[f"{side}_Femoral_Lateral_Epicn"], self.m[f"{side}_Femoral_Medial_Epicn"]
        knee = (lat + med) / 2
        y = _unit(hip - knee)
        z = _unit(lat - med)
        if self.side == "L":
            z = -z
        x = _unit(np.cross(y, z))
        R = _rot_from_axes(x, y, z)
        return knee, R

    def _shank_cs(self, knee):
        side = "Rt" if self.side == "R" else "Lt"
        lat, med = self.m[f"{side}_Lateral_Malleolus"], self.m[f"{side}_Medial_Malleolus"]
        ankle = (lat + med) / 2
        y = _unit(knee - ankle)
        z = _unit(lat - med)
        if self.side == "L":
            z = -z
        x = _unit(np.cross(y, z))
        R = _rot_from_axes(x, y, z)
        return ankle, R

    def _foot_cs(self, ankle):
        side = "Rt" if self.side == "R" else "Lt"
        heel = self.m[f"{side}_Calcaneous_Post"]
        mt5 = self.m[f"{side}_Metatarsal_Phal_V"]
        mt1 = self.m[f"{side}_Metatarsal_Phal_I"]
        x = _unit(((mt1 + mt5) / 2) - heel)
        y = _unit(np.cross(x, (mt5 - mt1)))
        z = _unit(np.cross(x, y))
        if self.side == "L":
            z = -z
        R = _rot_from_axes(x, y, z)
        return ankle, R

    def _knee_flex_simple(self, thigh_y: np.ndarray, shank_y: np.ndarray) -> float:
        """
        Flex(+) / Ext(–) from the angle between the long axes (Y) of thigh & shank.

        Accepts either:
          • 1‑D vectors of length 3  → returns a scalar
          • 2‑D array  (n, 3)       → returns a 1‑D array length n
        """
        thigh_y = np.asarray(thigh_y)
        shank_y = np.asarray(shank_y)

        if thigh_y.ndim == 2:  # vectorised case
            dot = np.einsum("ij,ij->i", thigh_y, shank_y)
        else:  # single frame
            dot = np.dot(thigh_y, shank_y)

        dot = np.clip(dot, -1.0, 1.0)  # numerical safety
        return np.degrees(np.arccos(dot))

    # ---------------- main compute ------------------------------------------
    def compute(self) -> pd.DataFrame:
        ang = {k: [] for k in [
            "hip_flexext", "hip_abdad", "hip_intext",
            "knee_flexext", "knee_abdad", "knee_intext",
            "ankle_pfdf", "ankle_inv_ev", "ankle_intext"]}

        Pelvis_O, Pelvis_R = self._pelvis_cs()
        hjcL, hjcR = self._hip_joint_centres(Pelvis_O, Pelvis_R)
        hip = hjcR if self.side == "R" else hjcL

        for f in range(self.nf):
            knee_O, Thigh_R = self._thigh_cs(hip[f])
            ankle_O, Shank_R = self._shank_cs(knee_O[f])
            _, Foot_R = self._foot_cs(ankle_O[f])

            R_hp = Pelvis_R[f].T @ Thigh_R[f]
            hip_e = _euler_zyx(R_hp)
            ang["hip_flexext"].append(hip_e[0])
            ang["hip_abdad"].append(hip_e[1])
            ang["hip_intext"].append(hip_e[2])

            # --- KNEE: angle between long axes  ------------------------------------
            thigh_y = Thigh_R[f][:, 1]  # (3,)
            shank_y = Shank_R[f][:, 1]  # (3,)
            ang["knee_flexext"].append(self._knee_flex_simple(thigh_y, shank_y))
            # keep abduction / rotation from Euler if you need them
            R_kn = Thigh_R[f].T @ Shank_R[f]
            kn_e = _euler_xyz(R_kn)
            ang["knee_abdad"].append(kn_e[1])
            ang["knee_intext"].append(kn_e[2])

            R_an = Shank_R[f].T @ Foot_R[f]
            an_e = _euler_xyz(R_an)
            ang["ankle_pfdf"].append(an_e[0])
            ang["ankle_inv_ev"].append(an_e[1])
            ang["ankle_intext"].append(an_e[2])

        return pd.DataFrame(ang)


###############################################################################
# Gait‑cycle detection + aggregation                                           #
###############################################################################

def detect_gait_cycles(df: pd.DataFrame, side: str = "R", prominence: float = 8.0,
                       min_separation: int = 30) -> List[int]:
    """Return list of *heel‑strike* frame indices (knee flexion minima).

    Se il trial non contiene passi restituirà [].
    """
    knee = df["knee_flexext"].to_numpy()
    # Minimi → invertiamo segno e cerchiamo picchi
    peaks, _ = find_peaks(-knee, prominence=prominence, distance=min_separation)
    return peaks.tolist()


def slice_and_normalise(df: pd.DataFrame, events: Sequence[int]) -> np.ndarray:
    """Interpola ogni ciclo (tra eventi[i]‑eventi[i+1]) a 101 punti.

    Restituisce array (n_cycle, 101, n_angles).
    """
    cycles = []
    cols = df.columns
    for a, b in zip(events[:-1], events[1:]):
        if b - a < 30:  # troppo corto (es. rumore)
            continue
        frac = np.linspace(0, 1, 101)
        segment = np.empty((101, len(cols)))
        for j, c in enumerate(cols):
            y = df[c].to_numpy()[a:b + 1]
            x = np.linspace(0, 1, y.size)
            segment[:, j] = np.interp(frac, x, y)
        cycles.append(segment)
    return np.asarray(cycles)  # (n,101, n_angles)


def aggregate_cycles(cycles: np.ndarray, columns: Sequence[str]) -> pd.DataFrame:
    mean = cycles.mean(0)
    std = cycles.std(0)
    dfm = pd.DataFrame(mean, columns=columns)
    dfs = pd.DataFrame(std, columns=[f"{c}_sd" for c in columns])
    return pd.concat([dfm, dfs], axis=1)


def plot_cycles_mean_sd(agg: pd.DataFrame, joint: str = "knee") -> None:
    map_cols = {
        "hip": ["hip_flexext", "hip_abdad", "hip_intext"],
        "knee": ["knee_flexext", "knee_abdad", "knee_intext"],
        "ankle": ["ankle_pfdf", "ankle_inv_ev", "ankle_intext"],
    }
    cols = map_cols[joint]
    x = np.linspace(0, 100, 101)
    plt.figure(figsize=(8, 5))
    for c in cols:
        m = agg[c].to_numpy()
        s = agg[f"{c}_sd"].to_numpy()
        plt.plot(x, m, label=c)
        plt.fill_between(x, m - s, m + s, alpha=0.2)
    plt.xlabel("% ciclo passo")
    plt.ylabel("angolo (deg)")
    plt.title(f"{joint.capitalize()} – media ±1 SD su {len(agg)} punti")
    plt.legend()
    plt.tight_layout()


def plot_cycles_mean_sd(
        agg: pd.DataFrame,
        joint: str = "knee",
        *,
        dpi: int = 100,
        show: bool = True,
):
    """
    Crea plot separati (media ± SD) per ciascun angolo del giunto.

    Parameters
    ----------
    agg : pd.DataFrame
        DataFrame prodotto da `aggregate_cycles`, con colonne `xxx` e `xxx_sd`.
    joint : {'hip','knee','ankle'}
        Quale giunto visualizzare.

    dpi : int
        Risoluzione dei PNG.
    show : bool
        Chiama `plt.show()` alla fine.
    """
    map_cols = {
        "hip": ["hip_flexext", "hip_abdad", "hip_intext"],
        "knee": ["knee_flexext", "knee_abdad", "knee_intext"],
        "ankle": ["ankle_pfdf", "ankle_inv_ev", "ankle_intext"],
    }
    cols = map_cols[joint]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]  # palette corrente
    x = np.linspace(0, 100, 101)

    figs = []
    for i, c in enumerate(cols):
        fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
        m = agg[c].to_numpy()
        s = agg[f"{c}_sd"].to_numpy()
        color = colors[i % len(colors)]

        ax.plot(x, m, color=color, label=c)
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.2)
        ax.set(xlabel="% ciclo passo", ylabel="angolo (deg)",
               title=f"{c.replace('_', ' ').title()} – media ±1 SD")
        ax.legend()
        ax.grid(True)
        figs.append(fig)

    if show:
        plt.show()

    return figs  # utile se vuoi manipolarle altrove


###############################################################################
# Dataset‑level pipeline                                                       #
###############################################################################

def compute_joint_angles_from_trc(trc_path: str, side: str = "R") -> pd.DataFrame:
    _, marker_names, _, _, markers_data = read_trc_file(trc_path)
    mdict = {n: markers_data[:, i, :] for i, n in enumerate(marker_names)}

    return ISBJointAngleCalculator(mdict, side).compute()


def process_dataset(dataset_path: Union[str, pathlib.Path], side: str = "R", *,
                    save_csv: Union[None, str, pathlib.Path] = None, plot: bool = True):
    dataset_path = pathlib.Path(dataset_path)  # ⇦ converto subito
    trc_files = load_paths(dataset_path)  # nome più esplicito

    print(f"Found {len(trc_files)} trials in {dataset_path}")

    all_cycles = []
    for p in trc_files:
        df = compute_joint_angles_from_trc(str(p), side)
        hs = detect_gait_cycles(df)
        if len(hs) < 2:
            continue
        cyc = slice_and_normalise(df, hs)
        if cyc.size > 0:
            all_cycles.append(cyc)
    if not all_cycles:
        raise RuntimeError("Nessun ciclo valido trovato nel dataset")

    for p in trc_files[:10]:  # solo un esempio
        df = compute_joint_angles_from_trc(str(p), side)
        hs = detect_gait_cycles(df)
        if len(hs) < 2:
            continue

        # Plot once, for the first valid file
        plot_knee_with_events(p, side=side)

    all_cycles = np.vstack(all_cycles)  # (n_tot,101,n_angles)
    agg = aggregate_cycles(all_cycles, df.columns)
    if save_csv:
        agg.to_csv(save_csv, index=False)
    if plot:
        for joint in ["hip", "knee", "ankle"]:
            plot_cycles_mean_sd(agg, joint)
            plt.show()
    return agg



def load_paths(dataset_path: pathlib.Path) -> List[str]:
    """Load all .trc files from a given directory."""
    # get list of subject dir from dataset_path
    subject_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]

    print(f"Found {len(subject_dirs)} subjects in {dataset_path}")

    trc_files = []
    for subject_dir in subject_dirs:
        # get list of all .trc files in the subject dir

        for file in subject_dir.glob("*.trc"):
            if str(file).endswith("GAIT__AL_mt.trc"):
                trc_files.append(file)

    return trc_files

def plot_knee_with_events(trc_path, side="R", prominence=8, min_sep=40):
    # ---- compute angles & events ----
    df = compute_joint_angles_from_trc(trc_path, side)
    events = detect_gait_cycles(df, side, prominence, min_sep)

    # ---- raw signal ----
    knee = df["knee_flexext"].to_numpy()
    frames = np.arange(len(knee))

    plt.figure(figsize=(12,4))
    plt.plot(frames, knee, label="knee flex/ext")
    plt.scatter(events, knee[events], c="red", zorder=5, label="detected HS")
    for e in events:
        plt.axvline(e, color="red", ls="--", alpha=0.3)
    plt.xlabel("Frame");  plt.ylabel("deg")
    plt.title(f"Knee flex/ext with HS candidates – {trc_path.name}")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()



###############################################################################
# CLI                                                                         #
###############################################################################

if __name__ == "__main__":
    ap = argparse.ArgumentParser("Gait‑angle batch processor")
    ap.add_argument("--trc", type=pathlib.Path)
    ap.add_argument("--side", choices=["R", "L"], default="R")
    ap.add_argument("--save_csv", type=pathlib.Path)
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()

    process_dataset(args.trc, side=args.side, save_csv=args.save_csv, plot=not args.no_plot)
