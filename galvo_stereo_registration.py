import csv
import cv2 as cv
import numpy as np
from surpass_stereo import StereoCalibration, SeparateStereo

class Tracker:
    def __init__(self):
        self.last_left_image = None
        self.last_right_image = None

        self.left_targets = []
        self.right_targets = []
        self.targets = []

    def add_detection(self, left_point, right_point):
        y_error = abs(left_point[1] - right_point[1])
        if y_error > 2:
            return

        closest_distance = np.inf
        closest_index = -1
        for idx, t in enumerate(self.targets):
            left_mean = np.mean(t[0], axis=0)
            right_mean = np.mean(t[1], axis=0)

            dist = min(np.linalg.norm(left_mean - left_point), np.linalg.norm(right_mean - right_point))
            if dist < closest_distance:
                closest_distance = dist
                closest_index = idx

        # Classify as existing point or add new point
        if closest_distance > 5:
            self.targets.append( ( [ left_point ], [ right_point ] ) )
        else:
            self.targets[closest_index][0].append(left_point)
            self.targets[closest_index][1].append(right_point)

    def update(self, left_image, right_image):
        left_detection = self.find_targets(left_image, self.last_left_image, "left thresh")
        right_detection = self.find_targets(right_image, self.last_right_image, "right thresh")

        if left_detection is not None and right_detection is not None:
            self.add_detection(left_detection, right_detection)

        # Display all detected targets
        for target in self.targets:
            left = np.mean(target[0], axis=0)
            right = np.mean(target[1], axis=0)
            cv.circle(left_image, (int(left[0]), int(left[1])), 4, (0,255,0), -1)
            cv.circle(right_image, (int(right[0]), int(right[1])), 4, (0,255,0), -1)

        self.last_left_image = left_image
        self.last_right_image = right_image

    def find_targets(self, image, last_image, name):
        if last_image is None:
            return None

        # Only look for laser pulse in the green channel
        delta = np.float32(image[:, :, 1]) - np.float32(last_image[:, :, 1])
        _, thresh = cv.threshold(image[:, :, 1], 150.0, 255.0, cv.THRESH_BINARY)
        thresh = np.uint8(thresh)

        cv.imshow(name, thresh)

        contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            target = max(contours, key = cv.contourArea)
            M = cv.moments(target)
            area = M["m00"]
            if area > 10:
                cx = int(M["m10"]/area)
                cy = int(M["m01"]/area)

                return np.array([cx, cy])

        return None

def main():
    stereo = SeparateStereo.openDecklink(0, 1)
    if stereo is None:
        print("Failed to open stereo camera")

    calibration = StereoCalibration.fromJSON("./share/csr_30deg_calibration.json")
    stereo.addCalibration(calibration)

    tracker = Tracker()

    cv.namedWindow("Tracking", cv.WINDOW_NORMAL)

    while True:
        ok = stereo.capture()
        left = np.copy(stereo.leftRectified())
        right = np.copy(stereo.rightRectified())
        tracker.update(left, right)
        left = left[270:-200, 540:-540]
        right = right[270:-200, 540:-540]
        image = np.hstack((left, right))
        cv.putText(image, f"{len(tracker.targets)}", (100, 150), cv.FONT_HERSHEY_SIMPLEX, 4.0, (255, 255, 255), 4, cv.FILLED)
        cv.imshow("Tracking", image)
        key = cv.waitKey(20) & 0xFF
        if key == 27 or key == ord('q'):
            print("Quitting...")
            break

    with open("targets.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow([ "left_x", "left_y", "right_x", "right_y"])
        for t in tracker.targets:
            left_mean = np.mean(t[0], axis=0)
            right_mean = np.mean(t[1], axis=0)
            writer.writerow([left_mean[0], left_mean[1], right_mean[0], right_mean[1]])

if __name__ == '__main__':
    main()
