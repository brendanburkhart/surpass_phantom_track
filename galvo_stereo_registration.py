import cv2 as cv
import json
import numpy as np
from surpass_stereo import SurpassStereo

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

    def update(self, left_image, right_image):
        self._update(left_image, self.last_left_image, self.left_targets)
        self._update(right_image, self.last_right_image, self.right_targets)
        self.last_left_image = left_image
        self.last_right_image = right_image

        print(len(self.left_targets), len(self.right_targets))

        self.left_targets = sorted(self.left_targets, key=lambda t: (np.mean(t, axis=0))[1])
        self.right_targets = sorted(self.right_targets, key=lambda t: (np.mean(t, axis=0))[1])

        self.targets = [
            self._to_3d(lt, rt) for lt, rt in zip(self.left_targets, self.right_targets)
        ]

    def _update(self, image, last_image, targets):
        if last_image is None:
            thresh = np.zeros(image.shape[:2], np.uint8)
            return thresh

        delta = cv.absdiff(last_image, image)
        delta_gray = cv.cvtColor(delta, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(delta_gray, 10, 255, cv.THRESH_BINARY)

        contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            target = max(contours, key = cv.contourArea)
            M = cv.moments(target)
            if M["m00"] > 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])

                closest_distance = np.inf
                closest_index = -1
                for idx, t in enumerate(targets):
                    t_mean = np.mean(t, axis=0)
                    dist = (t_mean[0]-cx)**2 + (t_mean[1]-cy)**2
                    if dist < closest_distance:
                        closest_distance = dist
                        closest_index = idx

                if closest_distance > 35:
                    targets.append( [ (cx, cy) ] )
                else:
                    targets[closest_index].append((cx, cy))

        for target in targets:
            t_mean = np.mean(target, axis=0)
            cv.circle(image, (int(t_mean[0]), int(t_mean[1])), 4, (255,0,255), -1)

        return thresh

def find_target(image):
    hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)

    lower = np.array([100, 100, 40], np.uint8)
    upper = np.array([140, 255, 255], np.uint8)
    mask = cv.inRange(hsv, lower, upper)

    morph_size = 5
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*morph_size + 1, 2*morph_size+1), (morph_size, morph_size))
    clean_mask = cv.morphologyEx(mask, cv.MORPH_OPEN, element)

    contours, _ = cv.findContours(clean_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(image, contours, -1, (0,255,0), 3)

    if len(contours) > 0:
        target = max(contours, key = cv.contourArea)
        cv.drawContours(image, [target], -1, (0,255,255), 5)

    cv.imshow("target mask", clean_mask)
    cv.imshow("target contours", image)

def main():
    config_file = "./share/diy_calibration.json"
    with open(config_file, "r") as f:
        config = json.load(f)

    stereo = SurpassStereo.DIY(config)
    stereo.set_exposure(25)

    tracker = Tracker(stereo.disparity_to_depth)

    cv.namedWindow("Tracking", cv.WND_PROP_FULLSCREEN)

    while True:
        ok, left, right = stereo.read()
        tracker.update(left, right)
        image = np.hstack((left, right))
        cv.imshow("Tracking", image)
        key = cv.waitKey(20) & 0xFF
        if key == 27 or key == ord('q'):
            print("Quitting...")
            print()
            print(json.dumps(np.array(tracker.targets).tolist()))
            break

if __name__ == '__main__':
    main()
