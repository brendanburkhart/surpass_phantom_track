import cv2 as cv
import math
import numpy as np
import json

class SurpassStereo:
    def DIY(config):
        return SurpassStereo(config)

    def __init__(self, config):
        self.left_camera = cv.VideoCapture(0, cv.CAP_V4L2)
        self.right_camera = cv.VideoCapture(2, cv.CAP_V4L2)

        default_exposure = 500
        self.left_camera.set(cv.CAP_PROP_EXPOSURE, default_exposure)
        self.right_camera.set(cv.CAP_PROP_EXPOSURE, default_exposure)

        self.left_camera.set(cv.CAP_PROP_BUFFERSIZE, 1)
        self.right_camera.set(cv.CAP_PROP_BUFFERSIZE, 1)

        width = 1920
        height = 1080

        downsample = 1

        self.left_camera.set(cv.CAP_PROP_FRAME_WIDTH, width//downsample)
        self.left_camera.set(cv.CAP_PROP_FRAME_HEIGHT, height//downsample)
        self.right_camera.set(cv.CAP_PROP_FRAME_WIDTH, width//downsample)
        self.right_camera.set(cv.CAP_PROP_FRAME_HEIGHT, height//downsample)

        self.has_rectification = False
        if config is not None:
            self.has_rectification = True
            self.left_intrinsic = np.array(config["left_camera"]["intrinsic"]) / downsample
            self.right_intrinsic = np.array(config["right_camera"]["intrinsic"]) / downsample

            self.left_distortion = np.array(config["left_camera"]["distortion"])
            self.right_distortion = np.array(config["right_camera"]["distortion"])

            original_image_size = (1920//downsample, 1080//downsample)

            R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(
                self.left_intrinsic,
                self.left_distortion,
                self.right_intrinsic,
                self.right_distortion,
                original_image_size,
                np.array(config["rotation"]),
                np.array(config["translation"]),
                alpha=0
            )

            self.left_rect = R1
            self.right_rect = R2
            self.left_proj = P1
            self.right_proj = P2
            self.disparity_to_depth = Q

            self.left_mapx, self.left_mapy = cv.initUndistortRectifyMap(
                self.left_intrinsic, self.left_distortion,
                self.left_rect, self.left_proj,
                original_image_size,
                cv.CV_32F
            )

            self.right_mapx, self.right_mapy = cv.initUndistortRectifyMap(
                self.right_intrinsic, self.right_distortion,
                self.right_rect, self.right_proj,
                original_image_size,
                cv.CV_32F
            )

            fx = self.left_proj[0,0]
            b = abs(np.array(config["translation"])[0])
            min_depth = 0.04
            max_disparity = math.ceil(fx*b/min_depth)
            max_disparity = max_disparity - (max_disparity % 16) # required to be multiple of 16

            block_size = 7
            block_area = block_size * block_size
            self.left_stereo_matcher = cv.StereoSGBM_create(
                0, max_disparity, block_size,
                8*3*block_area, 32*3*block_area,
                0, 0, 20,
                50, 1
            )

            self.right_stereo_matcher = cv.ximgproc.createRightMatcher(self.left_stereo_matcher)
            self.wls_filter = cv.ximgproc.createDisparityWLSFilter(self.left_stereo_matcher)
            self.wls_filter.setLambda(8000.0)
            self.wls_filter.setSigmaColor(0.6)

    def set_exposure(self, exposure):
        self.left_camera.set(cv.CAP_PROP_EXPOSURE, exposure)
        self.right_camera.set(cv.CAP_PROP_EXPOSURE, exposure)

    def compute_depth(self, left_image, right_image):
        left_gray = cv.cvtColor(left_image, cv.COLOR_RGB2GRAY)
        right_gray = cv.cvtColor(right_image, cv.COLOR_RGB2GRAY)
        left_disparity = self.left_stereo_matcher.compute(left_gray, right_gray)
        right_disparity = self.right_stereo_matcher.compute(right_gray, left_gray)

        disparity = self.wls_filter.filter(left_disparity, left_gray, disparity_map_right=right_disparity, right_view=right_gray)

        disparity_debug = cv.ximgproc.getDisparityVis(disparity)
        cv.imshow("disparity", disparity_debug)

        #disparity.convertTo(disparity, CV_32F, 1.0f/cv::StereoMatcher::DISP_SCALE, 0.0f)

    def read(self):
        self.left_camera.grab()
        self.right_camera.grab()

        ok, left_raw = self.left_camera.retrieve()
        if not ok:
            print("Failed to read left camera")
            return False, None, None

        ok, right_raw = self.right_camera.retrieve()
        if not ok:
            print("Failed to read right camera")
            return False, None, None

        if self.has_rectification:
            left_image = cv.remap(left_raw, self.left_mapx, self.left_mapy, cv.INTER_LINEAR)
            right_image = cv.remap(right_raw, self.right_mapx, self.right_mapy, cv.INTER_LINEAR)
        else:
            left_image = left_raw
            right_image = right_raw

        return True, left_image, right_image

    def run(self):
        cv.namedWindow("stereo", cv.WINDOW_NORMAL)
        while True:
            ok, left_image, right_image = self.read()
            if not ok:
                print("!! Failed to read from stereo camera !!")
                return

            combined = np.hstack((left_image, right_image))
            cv.imshow("stereo", combined)

            key = cv.waitKey(30)
            key = key & 0xFF # Upper bits are modifiers (control, alt, etc.)
            escape = 27
            if key == ord('q') or key == escape:
                break

def main():
    config_file = "./share/diy_calibration.json"
    with open(config_file, "r") as f:
        config = json.load(f)

    stereo = SurpassStereo.DIY(config)
    stereo.run()

if __name__ == '__main__':
    main()
