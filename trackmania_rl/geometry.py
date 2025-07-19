from pathlib import Path
from typing import List

import numpy as np
import numpy.typing as npt
from scipy.interpolate import make_interp_spline


"""
un ensemble de fonctions utilitaires pour la gestion des chemins et des zones dans Trackmania 
ce code vient de # https://gist.github.com/TimSC/8c25ca941d614bf48ebba6b473747d72
"""

def line_plane_collision_point(plane_normal, plane_point, ray_direction, ray_point, epsilon=1e-6):

    ndotu = plane_normal.dot(ray_direction)

    if abs(ndotu) < epsilon:
        raise RuntimeError("no intersection or line is within plane")

    w = ray_point - plane_point
    si = -plane_normal.dot(w) / ndotu
    intersection_point = ray_point + si * ray_direction
    return intersection_point


def fraction_time_spent_in_current_zone(
    current_zone_center: npt.NDArray, next_zone_center: npt.NDArray, current_pos: npt.NDArray, next_pos: npt.NDArray
) -> float:

    plane_normal = next_zone_center - current_zone_center
    si = -plane_normal.dot(current_pos - (next_zone_center + current_zone_center) / 2) / plane_normal.dot(next_pos - current_pos)
    return max(0, min(1, si))


def extract_cp_distance_interval(raw_position_list: List, target_distance_between_cp_m: float, base_dir: Path):

    interpolation_function = make_interp_spline(x=range(len(raw_position_list)), y=raw_position_list, k=1)
    raw_position_list = interpolation_function(np.arange(0, len(raw_position_list) - 1 + 1e-6, 0.01))
    a = np.array(raw_position_list)
    b = np.linalg.norm(a[:-1] - a[1:], axis=1) 
    c = np.pad(b.cumsum(), (1, 0))  
    number_zones = round(c[-1] / target_distance_between_cp_m - 0.5) + 0.5   
    zone_length = c[-1] / number_zones
    index_first_pos_in_new_zone = np.unique(c // zone_length, return_index=True)[1][1:]
    index_last_pos_in_current_zone = index_first_pos_in_new_zone - 1
    w1 = 1 - (c[index_last_pos_in_current_zone] % zone_length) / zone_length
    w2 = (c[index_first_pos_in_new_zone] % zone_length) / zone_length
    zone_centers = a[index_last_pos_in_current_zone] + (a[index_first_pos_in_new_zone] - a[index_last_pos_in_current_zone]) * (
        w1 / (1e-4 + w1 + w2)
    ).reshape((-1, 1))
    zone_centers = np.vstack(
        (
            raw_position_list[0][None, :],
            zone_centers,
            (2 * raw_position_list[-1] - zone_centers[-1])[None, :],
        )
    )
    np.save(base_dir / "maps" / "map.npy", np.array(zone_centers).round(4))
    return zone_centers
