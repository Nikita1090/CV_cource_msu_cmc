import os

import numpy as np
import torch
from PIL import Image
from skimage import io
from skimage.transform import resize

from config import VOC_CLASSES, bbox_util, model
from utils import get_color


def detection_cast(detections):
    """Helper to cast any array to detections numpy array.
    Even empty.
    """
    return np.array(detections, dtype=np.int32).reshape((-1, 5))


def rectangle(shape, ll, rr, line_width=5):
    """Draw rectangle on numpy array.

    rr, cc = rectangle(frame.shape, (ymin, xmin), (ymax, xmax))
    frame[rr, cc] = [0, 255, 0] # Draw green bbox
    """
    ll = np.minimum(np.array(shape[:2], dtype=np.int32) - 1, np.maximum(ll, 0))
    rr = np.minimum(np.array(shape[:2], dtype=np.int32) - 1, np.maximum(rr, 0))
    result = []

    for c in range(line_width):
        for i in range(ll[0] + c, rr[0] - c + 1):
            result.append((i, ll[1] + c))
            result.append((i, rr[1] - c))
        for j in range(ll[1] + c + 1, rr[1] - c):
            result.append((ll[0] + c, j))
            result.append((rr[0] - c, j))

    return tuple(zip(*result))


IMAGENET_MEAN = np.array([103.939, 116.779, 123.68]).reshape(1, 1, 3)


def image2tensor(image):
    image = image.astype(np.float32)
    image = resize(image, (300, 300), anti_aliasing=True, preserve_range=True)
    image = image[..., ::-1]
    image -= IMAGENET_MEAN
    image = image.transpose([2, 0, 1])
    tensor = torch.tensor(image.copy()).unsqueeze(0)
    return tensor

@torch.no_grad()
def extract_detections(frame, min_confidence=0.6, labels=None):
    """Extract detections from frame.

    frame: numpy array WxHx3
    returns: numpy int array Cx5 [[label_id, xmin, ymin, xmax, ymax]]
    """
    input_tensor = image2tensor(frame)
    output = model(input_tensor)
    output_np = output.numpy()

    results = bbox_util.detection_out(output_np, confidence_threshold=min_confidence)
    

    if labels is not None:
        result_labels = results[:, 0].astype(np.int32)
        indices = [
            index
            for index, label in enumerate(result_labels)
            if VOC_CLASSES[label - 1] in labels
        ]
        results = results[indices]
    results = np.array(results).reshape(-1, 6)
    results = np.array(results[:, [0, 2, 3, 4, 5]]).reshape(-1, 5)
    h, w, _ = frame.shape
    results[:, 1:] = results[:, 1:] * np.array([w, h, w, h])
    return detection_cast(results)




def draw_detections(frame, detections):
    """Draw detections on frame.

    Hint: help(rectangle) would help you.
    Use get_color(label) to select color for detection.
    """
    frame = frame.copy()
    # у меня чето тут сломалось
    return frame


def main():
    dirname = os.path.dirname(__file__)
    frame = Image.open(os.path.join(dirname, "data", "test.png"))
    frame = np.array(frame)

    detections = extract_detections(frame)
    frame = draw_detections(frame, detections)

    io.imshow(frame)
    io.show()


if __name__ == "__main__":
    main()
