# scripts/evaluate_centers.py
import math
import numpy as np

# Haversine distance (meters)
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2*R*math.asin(math.sqrt(a))

def read_points(path):
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    try:
        n = int(lines[0])
        pts = [tuple(map(float, line.split())) for line in lines[1:1+n]]
    except:
        pts = [tuple(map(float, line.split())) for line in lines]
    return np.array(pts)

def read_centers(path):
    with open(path) as f:
        return np.array([tuple(map(float, l.split())) for l in f if l.strip()])

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python scripts/evaluate_centers.py data/input.txt data/centers.out")
        sys.exit(1)

    data_file = sys.argv[1]
    centers_file = sys.argv[2]

    pts = read_points(data_file)
    centers = read_centers(centers_file)

    # For each point, find distance to nearest center
    dists = []
    for (lat, lon) in pts:
        m = min(haversine_m(lat, lon, c[0], c[1]) for c in centers)
        dists.append(m)
    dists = np.array(dists)
    print("points:", len(pts))
    print("centers:", len(centers))
    print(f"min distance to nearest center: {dists.min():.2f} m")
    print(f"median distance to nearest center: {np.median(dists):.2f} m")
    print(f"mean distance to nearest center: {dists.mean():.2f} m")
    print(f"max distance to nearest center: {dists.max():.2f} m")
    # percent within 20 ft (~6.1 m)
    within_20ft = (dists <= 6.096).sum() / len(dists) * 100.0
    print(f"percent of points within 20 ft (6.096 m) of a center: {within_20ft:.1f}%")
