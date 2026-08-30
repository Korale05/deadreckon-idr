"""
map_matching.py
===============
STEP 6 of the plan: snap the EKF's estimated trajectory onto the road network.

Uses OpenStreetMap road data (via osmnx) to "snap" each estimated position
to the nearest road segment. This is the final refinement stage — it takes
the EKF's already-fused estimate and constrains it to physically plausible
locations (i.e., on roads).

If osmnx is not available (e.g., no internet or missing dependency), the
module gracefully degrades by returning the input trajectory unchanged.
"""

import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

try:
    import osmnx as ox
    import networkx as nx
    from scipy.spatial import cKDTree
    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False


def download_road_network(center_lat: float, center_lon: float,
                          dist_m: float = 2000) -> dict:
    """
    Download the drivable road network from OpenStreetMap around a center point.

    Parameters
    ----------
    center_lat, center_lon : float
        Center of the area to download.
    dist_m : float
        Radius in meters around the center point.

    Returns
    -------
    dict with 'nodes_xy' (N, 2 array of [lon, lat]), 'tree' (cKDTree for
    fast nearest-neighbor lookup), and 'graph' (the osmnx graph object).
    """
    if not OSMNX_AVAILABLE:
        return None

    try:
        G = ox.graph_from_point((center_lat, center_lon), dist=dist_m,
                                 network_type="drive", simplify=True)
        nodes, edges = ox.graph_to_gdfs(G)

        # Build array of node coordinates for fast spatial lookup
        node_coords = np.column_stack([
            nodes.geometry.x.values,  # longitude
            nodes.geometry.y.values,  # latitude
        ])

        tree = cKDTree(node_coords)

        return {
            "nodes_xy": node_coords,
            "tree": tree,
            "graph": G,
            "nodes_gdf": nodes,
        }
    except Exception as e:
        print(f"[map_matching] Failed to download road network: {e}")
        return None


def snap_to_road(trajectory_latlon: np.ndarray,
                 road_network: dict) -> np.ndarray:
    """
    Snap each point in a trajectory to the nearest road network node.

    Parameters
    ----------
    trajectory_latlon : np.ndarray
        (N, 2) array of [latitude, longitude] positions.
    road_network : dict
        Output of download_road_network().

    Returns
    -------
    np.ndarray (N, 2) of [latitude, longitude] snapped to roads.
    """
    if road_network is None:
        return trajectory_latlon

    tree = road_network["tree"]
    nodes_xy = road_network["nodes_xy"]  # [lon, lat]

    # Query points: [lon, lat] to match tree's coordinate order
    query = np.column_stack([
        trajectory_latlon[:, 1],  # lon
        trajectory_latlon[:, 0],  # lat
    ])

    _, indices = tree.query(query)

    # Get snapped coordinates [lon, lat] -> return as [lat, lon]
    snapped_xy = nodes_xy[indices]
    snapped_latlon = np.column_stack([
        snapped_xy[:, 1],  # lat
        snapped_xy[:, 0],  # lon
    ])

    return snapped_latlon


def enu_to_latlon(east: np.ndarray, north: np.ndarray,
                  lat0: float, lon0: float) -> np.ndarray:
    """
    Convert ENU meters back to lat/lon (inverse of latlon_to_enu).

    Returns (N, 2) array of [latitude, longitude].
    """
    lat0_rad = np.radians(lat0)
    lat = lat0 + np.degrees(north / config.EARTH_RADIUS_M)
    lon = lon0 + np.degrees(east / (config.EARTH_RADIUS_M * np.cos(lat0_rad)))
    return np.column_stack([lat, lon])


def run_map_matching(est_pos_east: np.ndarray, est_pos_north: np.ndarray,
                     ref_lat0: float, ref_lon0: float,
                     road_network: dict = None) -> dict:
    """
    Run map matching on an estimated trajectory (in ENU meters).

    Parameters
    ----------
    est_pos_east, est_pos_north : np.ndarray
        Estimated positions in ENU meters from session start.
    ref_lat0, ref_lon0 : float
        Reference lat/lon (session start point) for ENU<->latlon conversion.
    road_network : dict, optional
        Pre-downloaded road network. If None, downloads automatically.

    Returns
    -------
    dict with:
        'lat', 'lon' : original trajectory in lat/lon
        'matched_lat', 'matched_lon' : map-matched trajectory
        'road_network' : the road network object (for reuse)
    """
    # Convert ENU -> lat/lon
    traj_latlon = enu_to_latlon(est_pos_east, est_pos_north, ref_lat0, ref_lon0)

    if not OSMNX_AVAILABLE:
        print("[map_matching] osmnx not installed. Returning raw trajectory.")
        return {
            "lat": traj_latlon[:, 0],
            "lon": traj_latlon[:, 1],
            "matched_lat": traj_latlon[:, 0],
            "matched_lon": traj_latlon[:, 1],
            "road_network": None,
        }

    # Download road network if not provided
    if road_network is None:
        center_lat = np.mean(traj_latlon[:, 0])
        center_lon = np.mean(traj_latlon[:, 1])
        road_network = download_road_network(center_lat, center_lon, dist_m=3000)

    if road_network is None:
        return {
            "lat": traj_latlon[:, 0],
            "lon": traj_latlon[:, 1],
            "matched_lat": traj_latlon[:, 0],
            "matched_lon": traj_latlon[:, 1],
            "road_network": None,
        }

    # Snap to roads
    matched = snap_to_road(traj_latlon, road_network)

    return {
        "lat": traj_latlon[:, 0],
        "lon": traj_latlon[:, 1],
        "matched_lat": matched[:, 0],
        "matched_lon": matched[:, 1],
        "road_network": road_network,
    }


if __name__ == "__main__":
    if OSMNX_AVAILABLE:
        print("osmnx is available. Map matching will work.")
    else:
        print("osmnx is NOT installed. Map matching will be skipped.")
        print("Install with: pip install osmnx")

    # Quick test with the first session's ground truth
    from src.discover_sessions import discover_all_sessions
    from src.io_utils import load_v_file, latlon_to_enu

    sessions = discover_all_sessions()
    if sessions:
        gt = load_v_file(sessions[0]["v_path"])
        lat = gt["lat"].to_numpy()
        lon = gt["lon"].to_numpy()
        east, north = latlon_to_enu(lat, lon, lat[0], lon[0])

        result = run_map_matching(east, north, lat[0], lon[0])
        print(f"Trajectory points: {len(east)}")
        print(f"Map matching {'succeeded' if result['road_network'] else 'skipped'}")
