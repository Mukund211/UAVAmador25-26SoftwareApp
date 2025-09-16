#!/usr/bin/env python3
"""
cluster_centers.py
Reads input file with N lat lon points, clusters into 5 groups,
filters outliers, and writes 5 centroid lat lon pairs to output.

Usage:
    python src/cluster_centers.py data/no_outlier.txt centers.out
"""

import sys
import math
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN

R_EARTH = 6371000.0  # meters

def read_points(path):
    """Read the specified input format:
       First line: N
       Next N lines: 'latitude longitude' (space separated)
    Returns Nx2 numpy array: [[lat, lon], ...]
    """
    with open(path, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        raise ValueError("Empty input file.")
    try:
        n = int(lines[0])
    except ValueError:
        # If first line isn't count, assume every line is a point
        n = len(lines)
        pts_lines = lines
    else:
        pts_lines = lines[1: 1 + n]
    pts = []
    for line in pts_lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        lat, lon = float(parts[0]), float(parts[1])
        pts.append((lat, lon))
    if not pts:
        raise ValueError("No coordinates found in input.")
    return np.array(pts)  # shape (N, 2)


def latlon_to_xy(lat, lon, lat0, lon0):
    """Convert arrays of lat/lon (deg) to local x,y in meters around lat0/lon0.
       Uses simple equirectangular projection which is fine for small areas.
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    lat0_rad = math.radians(lat0)
    x = R_EARTH * (np.radians(lon) - math.radians(lon0)) * math.cos(lat0_rad)
    y = R_EARTH * (np.radians(lat) - math.radians(lat0))
    return np.column_stack((x, y))


def xy_to_latlon(x, y, lat0, lon0):
    """Convert x,y (meters) back to lat/lon (deg) using the same center."""
    lat = lat0 + (y / R_EARTH) * (180.0 / math.pi)
    lon = lon0 + (x / (R_EARTH * math.cos(math.radians(lat0)))) * (180.0 / math.pi)
    return lat, lon


def robust_refine_centroids(xy_points, labels, kmeans_centers, mad_multiplier=3.0):
    """For each cluster label (0..k-1), remove points far from center using median+MAD
       and recompute centroid using only kept points. Returns refined centroids in xy.
    """
    refined = []
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        mask = labels == lbl
        cluster_xy = xy_points[mask]
        if cluster_xy.shape[0] == 0:
            # fallback: use the original kmeans center
            refined.append(kmeans_centers[lbl])
            continue
        center = kmeans_centers[lbl]
        dists = np.linalg.norm(cluster_xy - center, axis=1)
        med = np.median(dists)
        mad = np.median(np.abs(dists - med))
        if mad == 0:
            thr = med + mad_multiplier * np.std(dists)
        else:
            thr = med + mad_multiplier * mad
        keep_mask = dists <= thr
        if np.count_nonzero(keep_mask) < 3:
            # If too few kept points, don't filter aggressively
            refined_center = cluster_xy.mean(axis=0)
        else:
            refined_center = cluster_xy[keep_mask].mean(axis=0)
        refined.append(refined_center)
    return np.array(refined)


def main(infile, outfile, plot=False):
    pts = read_points(infile)  # Nx2 array lat,lon
    lat0 = float(np.mean(pts[:, 0]))
    lon0 = float(np.mean(pts[:, 1]))

    # project to meters
    xy = latlon_to_xy(pts[:,0], pts[:,1], lat0, lon0)

    # 1) DBSCAN to remove very sparse outliers (noise)
    # eps in meters: tuneable; 15m is a reasonable starting point for dense cluster detection
    db = DBSCAN(eps=15.0, min_samples=4).fit(xy)
    mask_core = db.labels_ != -1
    if np.count_nonzero(mask_core) < 20:
        # if DBSCAN removed too many points (or everything), fallback to no-filter
        xy_filtered = xy
        pts_filtered = pts
    else:
        xy_filtered = xy[mask_core]
        pts_filtered = pts[mask_core]

    # 2) KMeans to form 5 clusters
    k = 5
    if xy_filtered.shape[0] < k:
        raise ValueError("Not enough points after filtering to form 5 clusters.")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(xy_filtered)
    centers_xy = kmeans.cluster_centers_

    # 3) Refine centers by removing internal outliers per-cluster then recompute
    refined_xy = robust_refine_centroids(xy_filtered, labels, centers_xy, mad_multiplier=3.0)

    # convert refined xy back to lat/lon (we must preserve mapping of which refined center belongs to which cluster)
    # Note: kmeans.labels_ indexes clusters by 0..k-1 — robust_refine_centroids preserved that order.
    out_coords = []
    for cxy in refined_xy:
        lat_c, lon_c = xy_to_latlon(cxy[0], cxy[1], lat0, lon0)
        out_coords.append((lat_c, lon_c))

    # sort by latitude ascending
    out_coords_sorted = sorted(out_coords, key=lambda x: x[0])

    # write 5 lines with lat lon rounded to 5 decimals
    with open(outfile, 'w') as f:
        for lat_c, lon_c in out_coords_sorted:
            f.write(f"{lat_c:.5f} {lon_c:.5f}\n")

    print(f"Wrote {len(out_coords_sorted)} centers to {outfile}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/cluster_centers.py <infile> <outfile>")
        sys.exit(1)
    infile = sys.argv[1]
    outfile = sys.argv[2]
    main(infile, outfile)
