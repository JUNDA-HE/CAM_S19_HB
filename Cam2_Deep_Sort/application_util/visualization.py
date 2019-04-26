# vim: expandtab:ts=4:sw=4
import numpy as np
import colorsys
from .image_viewer import ImageViewer
from velocity_measure import compute_vel

def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255*r), int(255*g), int(255*b)


class NoVisualization(object):
    """
    A dummy visualization object that loops through all frames in a given
    sequence to update the tracker without performing any visualization.
    """

    def __init__(self, seq_info):
        self.frame_idx = seq_info["min_frame_idx"]
        self.last_idx = seq_info["max_frame_idx"]

    def set_image(self, image):
        pass

    def draw_groundtruth(self, track_ids, boxes):
        pass

    def draw_detections(self, detections):
        pass

    def draw_trackers(self, trackers):
        pass

    def run(self, frame_callback):
        while self.frame_idx <= self.last_idx:
            frame_callback(self, self.frame_idx)
            self.frame_idx += 1


class Visualization(object):
    """
    This class shows tracking output in an OpenCV image viewer.
    """

    def __init__(self, seq_info, update_ms):
        image_shape = seq_info["image_size"][::-1]
        aspect_ratio = float(image_shape[1]) / image_shape[0]
        image_shape = 1024, int(aspect_ratio * 1024)
        self.viewer = ImageViewer(
            update_ms, image_shape, "Figure %s" % seq_info["sequence_name"])
        self.viewer.thickness = 2
        self.frame_idx = seq_info["min_frame_idx"]
        self.last_idx = seq_info["max_frame_idx"]
        ImageViewer.enable_videowriter(self.viewer, 'test.mp4', "MJPG", 30)

    def run(self, frame_callback):
        self.viewer.run(lambda: self._update_fun(frame_callback))

    def _update_fun(self, frame_callback):
        if self.frame_idx > self.last_idx:
            return False  # Terminate
        frame_callback(self, self.frame_idx)
        self.frame_idx += 1
        return True

    def set_image(self, image):
        self.viewer.image = image

    def draw_groundtruth(self, track_ids, boxes):
        self.viewer.thickness = 2
        for track_id, box in zip(track_ids, boxes):
            self.viewer.color = create_unique_color_uchar(track_id)
            self.viewer.rectangle(*box.astype(np.int), label=str(track_id))

    def draw_detections(self, detections):
        self.viewer.thickness = 2
        self.viewer.color = 0, 0, 255
        for i, detection in enumerate(detections):
            self.viewer.rectangle(*detection.tlwh)

    def draw_trackers(self, frame_id, tracks):
        self.viewer.thickness = 2
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            self.viewer.color = create_unique_color_uchar(track.track_id)
            self.viewer.rectangle(
                *track.to_tlwh().astype(np.int), label=str(track.track_id))

            ##call compute_vel(box_vel, det, tr_frame, s_x, s_y, y_min, y_max, x_limit, H, location, V0, video_id)
            s_x = 3.2/5
            s_y = 4.0
            f1 = 905.53
            f2 = -71.2
            l0 = 0.2
            H = [[l0,-l0*(f1/f2),0.0],[0.0,1.0,0.0],[0.0,-(1/f2),1.0]]
            x_limit = 1920/2.0
            y_min = 300
            y_max = 2000.0
            V0 = 1

            bbox = track.to_tlbr()

            vel = compute_vel([track.mean[4],track.mean[5]],[bbox[0],bbox[1],bbox[2],bbox[3]],frame_id,s_x,s_y,y_min,y_max,x_limit,H,"Loc", V0 ,1)

            ##self.viewer.draw_velocities(track.mean[:2], label = f"{track.mean[4]:.2f},{track.mean[5]:.2f}")
            self.viewer.draw_velocities(track.mean[:2], label = f"{vel:.2f}")
