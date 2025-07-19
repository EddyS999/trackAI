
"""
Ce fichier contient un script pour convertir des fichiers GBX en listes de positions brutes.
Il utilise la bibliothèque `trackmania_rl` pour extraire les positions et les distances des checkpoints
et les enregistre dans un format exploitable.
Il est conçu pour être exécuté en ligne de commande avec un argument spécifiant le chemin du fichier GBX.
Il est utile pour préparer des données de cartes Trackmania pour l'entraînement d'agents d'apprentissage automatique.
Il extrait également les intervalles de distance des checkpoints pour une utilisation ultérieure.
"""

import argparse
from pathlib import Path
from trackmania_rl.geometry import extract_cp_distance_interval
from trackmania_rl.map_loader import gbx_to_raw_pos_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gbx_path", type=Path)
    args = parser.parse_args()
    base_dir = Path(__file__).resolve().parents[1]

    raw_positions_list = gbx_to_raw_pos_list(args.gbx_path)
    _ = extract_cp_distance_interval(raw_positions_list, 0.5, base_dir)


if __name__ == "__main__":
    main()