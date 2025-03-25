import numpy as np
def iou_score(bbox1, bbox2):
    """Jaccard index or Intersection over Union.

    https://en.wikipedia.org/wiki/Jaccard_index

    bbox: [xmin, ymin, xmax, ymax]
    """

    assert len(bbox1) == 4
    assert len(bbox2) == 4

    # Write code here
    xmin_inter = max(bbox1[0], bbox2[0])
    ymin_inter = max(bbox1[1], bbox2[1])
    xmax_inter = min(bbox1[2], bbox2[2])
    ymax_inter = min(bbox1[3], bbox2[3])

    if xmin_inter >= xmax_inter or ymin_inter >= ymax_inter:
        return 0.0  # В тестах иногда нет пересечения

    inter_area = (xmax_inter - xmin_inter) * (ymax_inter - ymin_inter) # это собственно его площадь

    area_bbox1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area_bbox2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    return inter_area / (area_bbox1 + area_bbox2 - inter_area)


def motp(obj, hyp, threshold=0.5):
    """Calculate MOTP

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0

    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Write code here

        # Step 1: Convert frame detections to dict with IDs as keys
        dict_1 = {x[0]: x[1:] for x in frame_obj}
        dict_2 = {x[0]: x[1:] for x in frame_hyp}
        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        for object_id, H_id in matches.items():
            if object_id in dict_1 and H_id in dict_2:
                iou = iou_score(dict_1[object_id], dict_2[H_id])
                if iou > threshold:
                    dist_sum += iou
                    match_count += 1
                    del dict_1[object_id]
                    del dict_2[H_id]
        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold
        pairwise_iou = {}
        for object_id, obj_bbox in dict_1.items():
            for H_id, hyp_bbox in dict_2.items():
                iou = iou_score(obj_bbox, hyp_bbox)
                if iou > threshold:
                    pairwise_iou[(object_id, H_id)] = iou
        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        sorted_iou = sorted(pairwise_iou.items(), key=lambda item: item[1], reverse=True)
        for (object_id, H_id), iou in sorted_iou:
            if object_id in dict_1 and H_id in dict_2:
                dist_sum += iou
                match_count += 1
                del dict_1[object_id]
                del dict_2[H_id]
                matches[object_id] = H_id
        # Step 5: Update matches with current matched IDs
        for object_id in dict_1.keys():
            if object_id in matches:
                del matches[object_id]

    # Step 6: Calculate MOTP
    MOTP = dist_sum / match_count if match_count > 0 else 0

    return MOTP


import numpy as np

def iou_score(bbox1, bbox2):
    """Jaccard index or Intersection over Union.

    https://en.wikipedia.org/wiki/Jaccard_index

    bbox: [xmin, ymin, xmax, ymax]
    """

    assert len(bbox1) == 4
    assert len(bbox2) == 4

    xmin_inter = max(bbox1[0], bbox2[0])
    ymin_inter = max(bbox1[1], bbox2[1])
    xmax_inter = min(bbox1[2], bbox2[2])
    ymax_inter = min(bbox1[3], bbox2[3])

    if xmin_inter >= xmax_inter or ymin_inter >= ymax_inter:
        return 0.0  # В тестах иногда нет пересечения

    inter_area = (xmax_inter - xmin_inter) * (ymax_inter - ymin_inter) # это собственно его площадь

    area_bbox1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area_bbox2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    return inter_area / (area_bbox1 + area_bbox2 - inter_area)

def motp_mota(obj, hyp, threshold=0.5): 
    """Calculate MOTP/MOTA

    obj: list
    Ground truth frame detections.
    detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
    Hypothetical frame detections.
    detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """
    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0
    missed_count = 0
    false_positive = 0
    mismatch_error = 0

    matches = {}  # matches between object IDs and hypothesis IDs

    for frame_obj, frame_hyp in zip(obj, hyp):
        # Step 1: Convert frame detections to dict with IDs as keys
        obj_dict = {d[0]: d[1:] for d in frame_obj}
        hyp_dict = {d[0]: d[1:] for d in frame_hyp}

        current_matches = {}

        # Step 2: Iterate over all previous matches
        for obj_id, hyp_id in matches.items():
            if obj_id in obj_dict and hyp_id in hyp_dict:
                iou = iou_score(obj_dict[obj_id], hyp_dict[hyp_id])
                if iou > threshold:
                    dist_sum += iou
                    match_count += 1
                    current_matches[obj_id] = hyp_id
                    del obj_dict[obj_id]
                    del hyp_dict[hyp_id]

        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        pairwise_iou = []
        for obj_id, obj_bbox in obj_dict.items():
            for hyp_id, hyp_bbox in hyp_dict.items():
                iou = iou_score(obj_bbox, hyp_bbox)
                if iou > threshold:
                    pairwise_iou.append((iou, obj_id, hyp_id))

        # Step 4: Iterate over sorted pairwise IOU
        pairwise_iou.sort(reverse=True, key=lambda x: x[0])
        for iou, obj_id, hyp_id in pairwise_iou:
            if obj_id in obj_dict and hyp_id in hyp_dict:
                dist_sum += iou
                match_count += 1
                current_matches[obj_id] = hyp_id
                del obj_dict[obj_id]
                del hyp_dict[hyp_id]

        # Step 5: If matched IDs contradict previous matched IDs - increase mismatch error
        for obj_id, hyp_id in current_matches.items():
            if obj_id in matches and matches[obj_id] != hyp_id:
                mismatch_error += 1

        # Step 6: Update matches with current matched IDs
        matches.update(current_matches)

        # Step 7: Errors
        false_positive += len(hyp_dict)  # All remaining hypotheses are false positives
        missed_count += len(obj_dict)  # All remaining objects are misses

    # Step 8: Calculate MOTP and MOTA
    MOTP = dist_sum / match_count if match_count > 0 else 0
    MOTA = 1 - (missed_count + false_positive + mismatch_error) / sum(len(f) for f in obj) if sum(len(f) for f in obj) > 0 else 0

    return MOTP, MOTA
