import numpy as np

from sklearn.cluster import DBSCAN, OPTICS, KMeans, AgglomerativeClustering, HDBSCAN

from sklearn.mixture import GaussianMixture

from sklearn.metrics.pairwise import haversine_distances

from scipy.sparse.csgraph import minimum_spanning_tree

from sklearn.neighbors import BallTree

import math

from django.db.models import QuerySet


from collections import defaultdict
from statistics import mean
# --- Haversine Distance --- #

def haversine_distance(lat1, lon1, lat2, lon2):

    R = 6371.0  # Earth radius in km

    phi1, phi2 = math.radians(lat1), math.radians(lat2)

    dphi = math.radians(lat2 - lat1)

    dlambda = math.radians(lon2 - lon1)


    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2

    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))



# --- Cluster Algorithm with Effective Radius --- #

def cluster_buildings_by_radius(

    buildings: QuerySet,

    numeric_field: str,       # e.g. "heating_demand"

    selection_mode: str,      # "max", "min", "mean"

    radius_km: float          # e.g. 0.5 km

):
    buildings_list = list(buildings)

    clusters = []


    while buildings_list:

        values = [getattr(b, numeric_field) or 0 for b in buildings_list]


        # --- Select center --- #

        if selection_mode == "max":

            center = buildings_list[values.index(max(values))]

        elif selection_mode == "min":

            center = buildings_list[values.index(min(values))]
            pass

        elif selection_mode == "mean":

            mean_val = sum(values) / len(values)

            center = buildings_list[min(range(len(values)), key=lambda i: abs(values[i] - mean_val))]

        else:

            raise ValueError("selection_mode must be 'max', 'min', or 'mean'")


        # --- Form cluster --- #

        cluster = []

        remaining = []

        for b in buildings_list:

            dist = haversine_distance(center.latitude, center.longitude, b.latitude, b.longitude)

            if dist <= radius_km:

                cluster.append(b)

            else:

                remaining.append(b)


        # --- Compute effective radius --- #

        distances = [

            haversine_distance(center.latitude, center.longitude, b.latitude, b.longitude)
            for b in cluster

        ]

        effective_radius = max(distances) if distances else 0

        mean_distance = sum(distances) / len(distances) if distances else 0


        clusters.append({

            "center": center,

            "buildings": cluster,

            "center_value": getattr(center, numeric_field),

            "effective_radius_km": round(effective_radius, 4),  # Max distance from center

            "mean_distance_km": round(mean_distance, 4),       # Optional insight

            "count": len(cluster),

        })

        buildings_list = remaining

    return clusters


# -----------------------------

# Helper

# -----------------------------

def _get_coords(buildings):

    """Extract coordinates in radians for haversine distance."""

    return np.radians(np.array([(b.latitude, b.longitude) for b in buildings]))

def radius_clusters(buildings, radius_km):

    """

    coords: array of [lat, lon] in radians

    radius_km: maximum radius for each cluster

    """
    c = _get_coords(buildings)


    tree = BallTree(c, metric='haversine')

    visited = np.zeros(len(c), dtype=bool)

    clusters = []
    

    for i in range(len(c)):

        if not visited[i]:

            # find all neighbors within radius

            ind = tree.query_radius([c[i]], r=radius_km/6371.0)[0]  # Earth radius = 6371 km

            clusters.append(ind.tolist())

            visited[ind] = True

    result = []

    for item in clusters:

        bs = [buildings[i] for i in item]

        result.append(bs)

        
    return result


def cluster_hdbscan(buildings, min_cluster_size=3):

    """

    Cluster buildings using HDBSCAN.


    Parameters
    ----------

    buildings : list of Building

        Each building must have latitude and longitude attributes.

    min_cluster_size : int

        Minimum number of buildings to form a cluster.


    Returns
    -------

    clusters : list of lists

        Each inner list contains buildings belonging to one cluster.

        Noise points are ignored.

    """

    if not buildings:

        return []


    # Convert lat/lon to radians

    coords = np.radians([[b.latitude, b.longitude] for b in buildings])


    # Initialize HDBSCAN

    clusterer = HDBSCAN(

        min_cluster_size=min_cluster_size,

        metric='haversine'
    )


    # Fit and predict cluster labels

    labels = clusterer.fit_predict(coords)


    # Group buildings by cluster label

    clusters = []

    unique_labels = set(labels)

    unique_labels.discard(-1)  # -1 = noise, ignore


    for label in unique_labels:

        cluster_buildings = [b for b, l in zip(buildings, labels) if l == label]

        clusters.append(cluster_buildings)

    return clusters


# -----------------------------

# DBSCAN

# -----------------------------

def cluster_dbscan(buildings, eps=0.5, min_samples=2):

    if not buildings:

        return []
    coords = _get_coords(buildings)

    kms_per_radian = 6371.0088

    eps_rad = radius_km / kms_per_radian

    db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine")

    labels = db.fit_predict(coords)

    return _labels_to_clusters(buildings, labels)




def cluster_buildings_with_dbscan(
    buildings: QuerySet,
    eps: float,              # Epsilon (eps): The max distance in km for points to be considered neighbors.
    min_samples: int = 2,    # The number of samples in a neighborhood for a point to be a core point.
    return_noise: bool = False # If True, returns a tuple (clusters, noise_points).
):
    """
    Clusters buildings using the DBSCAN algorithm with several improvements.

    Args:
        buildings (QuerySet): A Django QuerySet of building objects.
        eps (float): The clustering radius in kilometers (DBSCAN's epsilon).
        min_samples (int): The minimum number of buildings to form a dense cluster.
        return_noise (bool): If True, the function also returns a list of noise points.

    Returns:
        list[dict] or tuple[list[dict], list]:
        - By default, a list of cluster dictionaries.
        - If return_noise is True, a tuple containing (list_of_clusters, list_of_noise_buildings).
    """
    buildings_list = list(buildings)

    # 1. --- Prepare and Validate Data ---
    # Filter out buildings that don't have valid coordinates
    valid_buildings = [
        b for b in buildings_list
        if b.latitude is not None and b.longitude is not None
    ]

    if not valid_buildings:
        return ([], []) if return_noise else []

    # Extract coordinates and convert to radians for the haversine metric
    coords = np.array([[b.latitude, b.longitude] for b in valid_buildings])
    coords_rad = np.radians(coords)

    # 2. --- Run DBSCAN Algorithm ---
    # Convert radius in km to radians for the haversine distance metric
    earth_radius_km = 6371.0
    eps_rad = eps / earth_radius_km

    # Use 'ball_tree' for efficient geographic searches to prevent memory errors
    db = DBSCAN(
        eps=eps_rad,
        min_samples=min_samples,
        metric='haversine',
        algorithm='ball_tree'
    ).fit(coords_rad)
    labels = db.labels_

    # 3. --- Group buildings by their assigned cluster label ---
    clusters_map = defaultdict(list)
    noise_points = []
    for i, label in enumerate(labels):
        if label == -1:
            # This point is noise
            noise_points.append(valid_buildings[i])
        else:
            # Add the building to its corresponding cluster
            clusters_map[label].append(valid_buildings[i])

    # 4. --- Format clusters into the desired list of dictionaries ---
    result_clusters = []
    for label, cluster_buildings in clusters_map.items():
        result_clusters.append({
            # DBSCAN does not have a "center", so these fields are None.
            "center": None,
            "center_value": None,
            "effective_radius_km": None,
            "mean_distance_km": None,

            # These fields are direct outputs of the clustering result.
            "buildings": cluster_buildings,
            "count": len(cluster_buildings),
        })

    # 5. --- Return results based on the 'return_noise' flag ---
    if return_noise:
        return result_clusters, noise_points
    else:
        return result_clusters
# -----------------------------

# OPTICS

# -----------------------------

def cluster_optics(buildings, max_radius_km=0.5, min_samples=2):

    if not buildings:

        return []
    coords = _get_coords(buildings)

    kms_per_radian = 6371.0088

    eps = max_radius_km / kms_per_radian

    optics = OPTICS(max_eps=eps, min_samples=min_samples, metric="haversine")

    labels = optics.fit_predict(coords)

    return _labels_to_clusters(buildings, labels)


# -----------------------------

# K-Means

# -----------------------------

def cluster_kmeans(buildings, n_clusters=5):

    if not buildings:

        return []
    coords = _get_coords(buildings)

    km = KMeans(n_clusters=n_clusters, random_state=42)

    labels = km.fit_predict(coords)

    return _labels_to_clusters(buildings, labels)


# -----------------------------

# Hierarchical (Agglomerative)

# -----------------------------

def cluster_hierarchical(buildings, n_clusters=5, linkage="average"):

    if not buildings:

        return []
    coords = _get_coords(buildings)

    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, metric="euclidean")

    labels = model.fit_predict(coords)

    return _labels_to_clusters(buildings, labels)


# -----------------------------

# Gaussian Mixture Model (GMM)

# -----------------------------

def cluster_gmm(buildings, n_clusters=5):

    if not buildings:

        return []
    coords = _get_coords(buildings)

    gmm = GaussianMixture(n_components=n_clusters, random_state=42)

    labels = gmm.fit_predict(coords)

    return _labels_to_clusters(buildings, labels)


# -----------------------------

# Spectral Clustering

# -----------------------------

from sklearn.cluster import SpectralClustering


def cluster_spectral(buildings, n_clusters=5):

    if not buildings:

        return []
    coords = _get_coords(buildings)

    sc = SpectralClustering(n_clusters=n_clusters, affinity="nearest_neighbors", random_state=42)

    labels = sc.fit_predict(coords)

    return _labels_to_clusters(buildings, labels)


# -----------------------------

# MST Partitioning

# -----------------------------

def cluster_mst(buildings, n_clusters=5):

    if not buildings:

        return []
    coords = _get_coords(buildings)

    dist_matrix = haversine_distances(coords) * 6371.0088

    mst = minimum_spanning_tree(dist_matrix).toarray()


    # Collect edges

    edges = [(i, j, mst[i, j]) for i in range(len(coords)) for j in range(len(coords)) if mst[i, j] > 0]

    edges = sorted(edges, key=lambda x: -x[2])  # sort descending by weight


    # Union-Find
    parent = list(range(len(coords)))

    def find(x):

        while parent[x] != x:

            parent[x] = parent[parent[x]]

            x = parent[x]

        return x

    def union(x, y):

        parent[find(x)] = find(y)


    for i in range(len(coords)):

        parent[i] = i

    for (u, v, w) in edges[n_clusters-1:]:  # cut longest edges

        union(u, v)


    labels = np.array([find(i) for i in range(len(coords))])

    return _labels_to_clusters(buildings, labels)


# -----------------------------

# Utility: Convert labels to clusters

# -----------------------------

def _labels_to_clusters(buildings, labels):

    clusters = []

    for cluster_id in set(labels):

        if cluster_id == -1:

            continue  # skip noise

        cluster = [b for b, lbl in zip(buildings, labels) if lbl == cluster_id]

        clusters.append(cluster)
    return clusters


