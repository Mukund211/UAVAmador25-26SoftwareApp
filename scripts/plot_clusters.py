# scripts/plot_clusters.py
import numpy as np
import matplotlib.pyplot as plt

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
        print("Usage: python scripts/plot_clusters.py data/input.txt data/centers.out")
        sys.exit(1)

    pts = read_points(sys.argv[1])
    centers = read_centers(sys.argv[2])

    plt.figure(figsize=(8,6))
    plt.scatter(pts[:,1], pts[:,0], s=20, alpha=0.6, label="inferences (lon,lat)")
    plt.scatter(centers[:,1], centers[:,0], s=120, marker='X', c='red', label="centers (lon,lat)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.title("Inferences and computed centers (lon,lat)")
    plt.gca().invert_yaxis()  # optional — lat increases up, choose whichever you prefer
    plt.show()
