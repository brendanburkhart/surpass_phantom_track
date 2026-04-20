import csv
import cv2 as cv
import json
import numpy as np
from surpass_stereo import StereoCalibration, SeparateStereo
import time

class Tracker:
    def __init__(self, Q):
        self.last_left_image = None
        self.last_right_image = None

        self.left_targets = []
        self.right_targets = []
        self.targets = []

        self.Q = Q

    def _to_3d(self, left_samples, right_samples):
        left_target = np.mean(left_samples, axis=0)
        right_target = np.mean(right_samples, axis=0)

        y_disp = abs(left_target[1] - right_target[1])
        if y_disp > 3:
            print("y-error:", abs(left_target[1] - right_target[1]))

        disparity = left_target[0] - right_target[0]
        p = np.array([left_target[0], left_target[1], disparity, 1.0])
        P = self.Q @ p
        P = P / P[3]

        return np.array([P[0], P[1], P[2]], dtype=np.float32)
    
    def coalesce_detection(self, existing_detection, new_detection, threshold=0.25):
        # Have new target, use it
        if new_detection[1] is not None:
            return new_detection
        
        # No new target, but existing target is recent - use it
        age = new_detection[0] - existing_detection[0]
        if age < threshold:
            return existing_detection
        
        return (time.time(), None)
    
    def add_detection(self, left, right):
        left_ts, left_point = left
        right_ts, right_point = right
        y_error = abs(left_point[1] - right_point[1])
        if y_error > 5:
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

        if closest_distance > 15:
            self.targets.append( ( [ left_point ], [ right_point ] ) )
        else:
            self.targets[closest_index][0].append(left_point)
            self.targets[closest_index][1].append(right_point)

        # Make sure we don't reuse these detections
        self.left_detection = (time.time(), None)
        self.right_detection = (time.time(), None)

    def update(self, left_image, right_image):
        left_detection = self.find_targets(left_image, self.last_left_image, "left thresh")
        right_detection = self.find_targets(right_image, self.last_right_image, "right thresh")

        self.left_detection = self.coalesce_detection(self.left_detection, left_detection)
        self.right_detection = self.coalesce_detection(self.right_detection, right_detection)

        if self.left_detection[1] is not None and self.right_detection[1] is not None:
            self.add_detection(self.left_detection, self.right_detection)

        for target in self.targets:
            left = np.mean(target[0], axis=0)
            right = np.mean(target[1], axis=0)
            cv.circle(left_image, (int(left[0]), int(left[1])), 4, (255,0,255), -1)
            cv.circle(right_image, (int(right[0]), int(right[1])), 4, (255,0,255), -1)

        self.last_left_image = left_image
        self.last_right_image = right_image

    def find_targets(self, image, last_image, name):
        if last_image is None:
            return (time.time(), None)

        # Only look for laser pulse in the red channel
        delta = np.float32(image[:, :, 2]) - np.float32(last_image[:, :, 2])
        _, thresh = cv.threshold(delta, 10.0, 255.0, cv.THRESH_BINARY)
        thresh = np.uint8(thresh)

        cv.imshow(name, thresh)

        contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            target = max(contours, key = cv.contourArea)
            M = cv.moments(target)
            area = M["m00"]
            if area > 25:
                cx = int(M["m10"]/area)
                cy = int(M["m01"]/area)

                return (time.time(), np.array([cx, cy]))

        return (time.time(), None)

def main():
    stereo = SeparateStereo.openDecklink(0, 1)
    if stereo is None:
        print("Failed to open stereo camera")
        
    calibration = StereoCalibration.fromJSON("./share/csr_30deg_calibration.json")
    stereo.addCalibration(calibration)

    # Todo: get from calibration
    disparity_to_depth = np.eye(4, dtype=np.float64)
    f = 1288.4442228002379
    T = -5.484535539944649 / f
    disparity_to_depth[0, 3] = -1057.8859329223633
    disparity_to_depth[1, 3] = -578.4940795898438
    disparity_to_depth[2, 3] = f
    disparity_to_depth[3, 2] = -1/T
    disparity_to_depth[3, 3] = 0.0 # zero disparity

    tracker = Tracker(disparity_to_depth)

    cv.namedWindow("Tracking", cv.WINDOW_NORMAL)

    while True:
        ok = stereo.capture()
        left = stereo.leftRectified()
        right = stereo.rightRectified()
        tracker.update(left, right)
        image = np.hstack((left, right))
        cv.putText(image, f"{len(tracker.targets)}", (100, 150), cv.FONT_HERSHEY_SIMPLEX, 4.0, (255, 0, 0), 4, cv.FILLED)
        display = cv.resize(image, (0,0), fx=0.25, fy=0.25)
        cv.imshow("Tracking", display)
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
