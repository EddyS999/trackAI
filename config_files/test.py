import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from config_files import config_copy 
# print(cfg.trackmania_base_path)
from pathlib import Path
from pygbx import Gbx, GbxType

import re 

def _strip_colors(s: str) -> str:
    return re.sub(r"\$[0-9A-Fa-f]{3}", "", s)

def map_name_from_map_path(map_path: str) -> str:
    full_path = (
        config_copy.trackmania_base_path
        / "Tracks" / "Challenges"
        / Path(map_path.strip("'\""))
    )

    gbx = Gbx(str(full_path))
    ch  = gbx.get_class_by_id(GbxType.CHALLENGE)

    if ch and getattr(ch, "map_name", None):
        # Ex : "$fffESL$000-$666Hockolicious" → "ESL-Hockolicious"
        return _strip_colors(ch.map_name)

    # -- Fallback : on renvoie le nom de fichier sans extension
    return _strip_colors(full_path.stem)


# map_path = "'/home/user/Trackmania/Tracks/Challenges/MyMap.gbx'"
# gbx = Gbx(str(cfg.trackmania_base_path / "Tracks" / "Challenges" / Path(map_path.strip("'\""))))
# gbx_challenge = gbx.get_class_by_id(GbxType.CHALLENGE)
# print(gbx_challenge.map_name)


from trackmania_rl.map_loader import map_name_from_map_path
print(map_name_from_map_path(r'"ESL-Hockolicious.Challenge.Gbx"'))
