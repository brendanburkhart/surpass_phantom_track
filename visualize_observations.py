import csv
import matplotlib.pyplot as plt

def load_data(f):
    points = []

    with open(f, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            lx = float(row["left_x"])
            ly = float(row["left_y"])
            rx = float(row["right_x"])
            ry = float(row["right_y"])

            points.append((lx, ly, rx, ry))

    return points

d = load_data("targets_90deg_flat.csv")
plt.scatter([p[0] for p in d], [p[1] for p in d], label="left")
#plt.scatter([p[2] for p in d], [p[3] for p in d], label="left")
plt.show()
plt.pause()

