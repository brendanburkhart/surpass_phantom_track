import argparse
import csv
import math
import numpy as np

def create_setpoints(angle, N=7):
    scale = 0.2
    angle = angle * math.pi/180.0   

    T = scale * np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])

    points = []

    for i in range(N):
        for j in range(N):
            if i + j >= N:
                continue

            x = j - (N//2)
            y = i - (N//2)

            p = (T @ np.array([x, y])) * math.pi/180.0
            points.append(p.tolist())

    with open("galvo_setpoints.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["galvo_x", "galvo_y"])
        for p in points:
            x, y = p[0], p[1]
            writer.writerow([x, y])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("angle", help="angle in degrees")
    args = parser.parse_args()
    angle = float(args.angle)
    print(f"Creating setpoints for angle {angle}")
    create_setpoints(angle)
    print("Done!")


if __name__ == "__main__":
    main()
