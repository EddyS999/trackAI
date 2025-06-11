from pygbx import Gbx, GbxType
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

"""
Fonctions utilitaires pour charger les données d'un replay Trackmania 

Issue #1 :
Il est impossible d'identifier une carte via son replay.
La méthode la plus efficace que j'ai trouvée est de charger les positions du ghost
et de les comparer à un VCP (Virtual Checkpoint) de la carte. 
si la distance moyenne entre les positions du ghost et celles du VCP est inférieure
à une certaine valeur, on considère que le replay est valide pour cette carte.

"""

def load_ghost_positions(replay_path: str | Path) -> np.ndarray:
    """
    Charge les positions du ghost à partir d'un .Replay.Gbx.
    Renvoie un np.array de shape (N,3) . Si aucun sample, retourne un array vide (0,3).
    """
    gbx   = Gbx(str(Path(replay_path)))
    ghost = gbx.get_class_by_id(GbxType.CTN_GHOST)
    if ghost is None:
        raise ValueError(f"No ghost block in {replay_path!r}")

    # pygbx stocke les échantillons dans `ghost.records`
    recs = getattr(ghost, "records", None)
    if not recs:
        # parfois on a control_entries au lieu de records
        recs = getattr(ghost, "control_entries", None)

    if not recs:
        # pas de samples du tout
        return np.empty((0, 3), dtype=np.float32)

    # extraire (x,y,z) pour chaque sample
    return np.array(
        [[r.position.x, r.position.y, r.position.z] for r in recs],
        dtype=np.float32
    )

def check_replay_vs_map(replay_path: str, vcp_path: str, max_avg_dist: float = 5.0) -> bool:
    """
    Compare les positions du ghost dans un replay avec les positions d'un VCP (Virtual Checkpoint) d'une carte.
    :param replay_path: Chemin vers le fichier de replay.
    :param vcp_path: Chemin vers le VCP de la carte.
    :param max_avg_dist: Distance maximale moyenne autorisée entre les positions du ghost et celles du VCP.
    :return: True si la distance moyenne est inférieure ou égale à max_avg_dist, False sinon.
    """
    positions = load_ghost_positions(replay_path)  # (N,3) ou (0,3)
    if positions.size == 0:
        # pas d'échantillons → on ne peut pas vérifier
        return False

    vcp  = np.load(vcp_path)      # (M,3)
    tree = cKDTree(vcp)
    dists, _ = tree.query(positions, k=1)
    avg_dist = float(np.mean(dists))
    #print('distance moyenne: ',avg_dist)
    return avg_dist <= max_avg_dist


def get_replay_and_map_data(
    replay_path: str,
    map_gbx_path: str = None
) -> dict:
    """
    Extrait les infos du replay et, si on fournit map_gbx_path,
    les infos de la carte challenge correspondante.
    """
    # ---- 1) Ghost du replay ----
    gbx_replay = Gbx(str(Path(replay_path)))
    ghost = gbx_replay.get_class_by_id(GbxType.CTN_GHOST)
    if not ghost:
        raise ValueError("Pas de ghost dans ce fichier replay.")
    
    info = {
        "race_time_s":   ghost.race_time,
        
        "num_respawns":  getattr(ghost, "num_respawns", None),
        "game_version":  getattr(ghost, "game_version", None),
        "cp_times_ms":   getattr(ghost, "cp_times", None),
        "stunts_score":  getattr(ghost, "stunts_score", None),
        "replay_uid":    getattr(ghost, "uid", None),
        "map_uid":       None,      # à remplir plus bas si possible
        "map_name":      None,
        "map_author":    None,
    }

    # ---- 2) Si on nous donne la carte .Challenge.Gbx, on l'ouvre ----
    if map_gbx_path:
        map_gbx = Gbx(str(Path(map_gbx_path)))
        challenge = map_gbx.get_class_by_id(GbxType.CHALLENGE)
        if challenge:
            info["map_uid"] = getattr(challenge, "map_uid", None)
            info["map_name"] = getattr(challenge, "map_name", None)
            info["map_author"] = getattr(challenge, "map_author", None)

    return info



# replay_file = "C:/Users/User/Documents/TmForever/Tracks/Replays/A05-Race_akira_1728.Replay.Gbx"
# vcp_file = "D:/python/trackAI/maps/A01_0.5m_.npy"
# # How to get map name from a replay file ? 

# ok = check_replay_vs_map(replay_file, vcp_file, max_avg_dist=100.0)
# print("Replay correspond à la map ?" , ok)



# if __name__ == "__main__":
#     replay_path = "C:/Users/User/Documents/TmForever/Tracks/Replays/A01-Race_akira(00'31''00).Replay.Gbx"
#     replay_data = get_replay_and_map_data(replay_path)
#     print(f"Replay Data from {replay_path}:")
#     for k, v in replay_data.items():
#         print(f"{k:15s} → {v}")