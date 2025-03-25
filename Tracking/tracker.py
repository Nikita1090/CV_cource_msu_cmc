import os

import numpy as np

from detection import detection_cast, draw_detections, extract_detections
from metrics import iou_score

class Tracker:
    """Generate detections and build tracklets."""
    def __init__(self, return_images=True, lookup_tail_size=80, labels=None):
        self.return_images = return_images  # Return images or detections?
        self.frame_index = 0
        self.labels = labels  # Tracker label list
        self.detection_history = []  # Saved detection list
        self.last_detected = {}
        self.tracklet_count = 0  # Counter to enumerate tracklets

        # We will search tracklet at last lookup_tail_size frames
        self.lookup_tail_size = lookup_tail_size

    def new_label(self):
        """Get new unique label."""
        self.tracklet_count += 1
        return self.tracklet_count - 1

    def init_tracklet(self, frame):
        """Get new unique label for every detection at frame and return it."""
        # Write code here
        # Use extract_detections and new_label
        detections = extract_detections(frame) #[label, x1, y1, x2, y2]
        detections[:, 0] = np.array([self.new_label() for i in range(detections.shape[0])])
        return detections

    @property
    def prev_detections(self):
        """Get detections at last lookup_tail_size frames from detection_history.

        One detection at one id.
        """
        detections = {}
        # Write code here
        for i in range(max(0, self.frame_index - self.lookup_tail_size), self.frame_index):
            for detection in self.detection_history[i]:
                detections[detection[0]] = detection
        detections = list(detections.values())
        return detection_cast(detections)

    def bind_tracklet(self, observations, threshold=0.7):
        """Assign tracklet IDs to the first column of observations.

        Match current observations with previous observations based on the highest IOU.

        observations: numpy int array Cx5 [[label_id, xmin, ymin, xmax, ymax]]
        return: matched observations numpy int array Cx5 [[tracklet_id, xmin, ymin, xmax, ymax]]
        """
        observations = observations.copy()
        previous_observations = self.prev_detections

        # Step 1: Compute pairwise IOU between current and previous observations
        matches = [
            [iou_score(observations[i][1:], previous_observations[j][1:]), i, j]
            for i in range(len(observations))
            for j in range(len(previous_observations))
        ]

        # Step 2: Sort matches by IOU in descending order
        matches.sort(key=lambda x: x[0], reverse=True)

        # Step 3: Assign tracklet IDs based on the best matches
        used_current, used_previous = [0] * observations.shape[0], [0] * previous_observations.shape[0]
        for match in matches:
            if not used_current[match[1]] and not used_previous[match[2]]:
                used_previous[match[2]] = 1
                used_current[match[1]] = 1
                observations[match[1]][0] = previous_observations[match[2]][0]

        # Step 4: Assign new tracklet IDs to unmatched observations
        for i, observation in enumerate(observations):
            if not used_current[i]:
                observation[0] = self.new_label()

        return detection_cast(observations)


    def save_detections(self, detections):
        """Save last detection frame number for each label."""
        for label in detections[:, 0]:
            self.last_detected[label] = self.frame_index

    def update_frame(self, frame):
        if not self.frame_index:
            # First frame should be processed with init_tracklet function
            detections = self.init_tracklet(frame)
        else:
            # Every Nth frame should be processed with CNN (very slow)
            # First, we extract detections
            detections = extract_detections(frame, labels=self.labels)
            # Then bind them with previous frames
            # Replacing label id to tracker id is performing in bind_tracklet function
            detections = self.bind_tracklet(detections)

        # After call CNN we save frame number for each detection
        self.save_detections(detections)
        # Save detections and frame to the history, increase frame counter
        self.detection_history.append(detections)
        self.frame_index += 1

        # Return image or raw detections
        # Image usefull to visualizing, raw detections to metric
        if self.return_images:
            return draw_detections(frame, detections)
        else:
            return detections


"""def main():
    from moviepy.editor import VideoFileClip # КАК ТЕБЯ УСТАНОВИТЬ

    dirname = os.path.dirname(__file__)
    input_clip = VideoFileClip(os.path.join(dirname, "data", "test.mp4"))

    tracker = Tracker()
    input_clip.fl_image(tracker.update_frame).preview()


if __name__ == "__main__":
    main()"""
