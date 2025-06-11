# ------------------------------------------------------------------
#  Sauvegarde des hyperparamètres du run dans un fichier texte/json
# ------------------------------------------------------------------
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import json, inspect, datetime
from config_files import config_copy
import shutil
from pathlib import Path

def copy_configuration_file():
    base_dir = Path(__file__).resolve().parents[1]
    shutil.copyfile(
        base_dir / "config_files" / "config.py",
        base_dir / "config_files" / "config_copy.py",
    )


def save_hyperparams(config_mod, save_dir):
    # Récupère tous les attributs publics (pas de __ ou fonctions)
    hyperparams = {
        k: v
        for k, v in inspect.getmembers(config_mod)
        if not k.startswith("_") and not inspect.ismodule(v) and not inspect.isfunction(v)
    }
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    outfile = save_dir / f"hyperparams_{ts}.json"
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(hyperparams, f, indent=2, default=str)
    print(f"Hyper-paramètres sauvegardés dans : {outfile}")

# juste après l’appel copy_configuration_file() ou après import config_copy
base_dir = Path(__file__).resolve().parents[1]
save_dir = base_dir / "save" / config_copy.run_name
save_dir.mkdir(parents=True, exist_ok=True)

save_hyperparams(config_copy, save_dir)
# ------------------------------------------------------------------