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