import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ============================== 1 Classifier model ============================
def get_cls_model(input_shape=(1, 40, 100)):
    """
    :param input_shape: tuple (n_rows, n_cols, n_channels)
            input shape of image for classification
    :return: nn model for classification
    """
    # your code here \/
    class ClassifierModel(nn.Module):
        def __init__(self):
            super(ClassifierModel, self).__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(32 * 10 * 25, 128)
            self.fc2 = nn.Linear(128, 2)

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = self.flatten(x)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    return ClassifierModel()
    # your code here /\


def fit_cls_model(X, y, fast_train=True):
    """
    :param X: 4-dim tensor with training images
    :param y: 1-dim tensor with labels for training
    :return: trained nn model
    """
    # your code here \/
    model = get_cls_model()
    # train model
    batch_size = 32 if fast_train else 10
    epochs = 5 if fast_train else 50
    lr = 0.001 if fast_train else 0.0001
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    return model
    # your code here /\


# ============================ 2 Classifier -> FCN =============================
def get_detection_model(cls_model):
    """
    :param cls_model: trained cls model
    :return: fully convolutional nn model with weights initialized from cls
             model
    """
    # your code here \/
    detection_model = nn.Sequential(cls_model.conv1, cls_model.pool, cls_model.conv2, cls_model.pool, nn.Conv2d(32, 128, kernel_size=(10, 25)), nn.Conv2d(128, 2, kernel_size=1))
    with torch.no_grad():
        detection_model[4].weight.copy_(cls_model.fc1.weight.view(128, 32, 10, 25))
        detection_model[4].bias.copy_(cls_model.fc1.bias)
        detection_model[5].weight.copy_(cls_model.fc2.weight.view(2, 128, 1, 1))
        detection_model[5].bias.copy_(cls_model.fc2.bias)
    return detection_model
    # your code here /\


# ============================ 3 Simple detector ===============================
def get_detections(detection_model, dictionary_of_images):
    """
    :param detection_model: trained fully convolutional detector model
    :param dictionary_of_images: dictionary of images in format
        {filename: ndarray}
    :return: detections in format {filename: detections}. detections is a N x 5
        array, where N is number of detections. Each detection is described
        using 5 numbers: [row, col, n_rows, n_cols, confidence].
    """
    # your code here \/
    #import matplotlib.pyplot as plt
    detections = {}
    threshold = 0.5
    for filename, image in dictionary_of_images.items():
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        img_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            out = detection_model(img_tensor)
            heatmap = torch.softmax(out, dim=1)[0, 1]
            detections[filename] = []
            for i in range(heatmap.shape[0]):
                for j in range(heatmap.shape[1]):
                    if heatmap[i, j] > threshold:
                        detections[filename].append([i, j, 40, 100, heatmap[i, j].item()])
    return detections
    # your code here /\


# =============================== 5 IoU ========================================
def calc_iou(first_bbox, second_bbox):
    """
    :param first bbox: bbox in format (row, col, n_rows, n_cols)
    :param second_bbox: bbox in format (row, col, n_rows, n_cols)
    :return: iou measure for two given bboxes
    """
    # your code here \/
    x2 = first_bbox[0] + first_bbox[2]
    y2 = first_bbox[1] + first_bbox[3]
    sx2 = second_bbox[0] + second_bbox[2]
    sy2 = second_bbox[1] + second_bbox[3]

    x1 = max(first_bbox[0], second_bbox[0])
    y1 = max(first_bbox[1], second_bbox[1])
    x2 = min(x2, sx2)
    y2 = min(y2, sy2)

    if x1 >= x2 or y1 >= y2:
        return 0.0

    intersection_area = (x2 - x1) * (y2 - y1)
    first_area = first_bbox[2] * first_bbox[3]
    second_area = second_bbox[2] * second_bbox[3]
    union_area = first_area + second_area - intersection_area
    iou = intersection_area / union_area
    return iou
    # your code here /\


# =============================== 6 AUC ========================================
def calc_auc(pred_bboxes, gt_bboxes):
    """
    Вычисляет метрику AUC для заданных детекций и ground truth боксов.

    :param pred_bboxes: словарь боксов в формате {filename: detections}
        detections — это массив N x 5, где N — количество детекций. Каждая
        детекция описывается 5 числами: [row, col, n_rows, n_cols, confidence].
    :param gt_bboxes: словарь боксов в формате {filename: bboxes}
        bboxes — список кортежей в формате (row, col, n_rows, n_cols)
    :return: значение AUC для заданных детекций и ground truth
    """
    # your code here \/
    iou_thr = 0.5
    tp, fp, all_confidences = [], [], []
    total_gt = sum(len(boxes) for boxes in gt_bboxes.values())
    for filename in pred_bboxes:
        predictions = list(pred_bboxes[filename])
        predictions.sort(key=lambda x: x[4], reverse=True)
        for pred in predictions:
            if not gt_bboxes.get(filename):
                fp.append(1)
                all_confidences.append(pred[4])
                continue
            ious = [calc_iou(tuple(pred[:-1]), gt_box) for gt_box in gt_bboxes[filename]]
            max_iou_index = np.argmax(ious)
            if ious[max_iou_index] >= iou_thr:
                tp.append(pred[4])
                gt_bboxes[filename].pop(max_iou_index)
            else:
                fp.append(1)
            all_confidences.append(pred[4])
    all_confidences.sort(reverse=True)
    tp.sort(reverse=True)
    all_confidences, tp = np.array(all_confidences), np.array(tp)
    recall_precision = np.zeros((len(all_confidences) + 1, 2))
    for i, conf in enumerate(all_confidences):
        count_above_conf = np.sum(all_confidences >= conf)
        if tp.size == 0:
            precision = recall = 0
        else:
            tp_above_conf = np.sum(tp >= conf)
            precision = tp_above_conf / count_above_conf
            recall = tp_above_conf / total_gt
        recall_precision[i + 1] = np.array([recall, precision])
    recall_precision[0] = np.array([0, 1])
    recall = recall_precision[:, 0]
    precision = recall_precision[:, 1]
    auc = np.trapz(precision, recall)
    return auc
    # your code here /\


# =============================== 7 NMS ========================================
def nms(detections_dictionary, iou_thr=1.0):
    """
    :param detections_dictionary: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param iou_thr: IoU threshold for nearby detections
    :return: dict in same format as detections_dictionary where close detections
        are deleted
    """
    # your code here \/
    nms_detections = {}
    for filename, detections in detections_dictionary.items():
        if len(detections) == 0:
            nms_detections[filename] = detections
            continue
        
        detections = np.array(detections)
        detections = detections[np.argsort(-detections[:, 4])]
        s_detections = []
        while len(detections) > 0:
            curr = detections[0]
            s_detections.append(curr)
            detections = detections[1:]
            detections = np.array([tmp for tmp in detections if calc_iou(curr[:4], tmp[:4]) < iou_thr])

        nms_detections[filename] = np.array(s_detections).tolist()

    return nms_detections
    # your code here /\

if __name__ == "__main__":
    data = np.load('train_data.npz')
    X = data['X'].reshape(-1, 1, 40, 100)
    y = data['y'].astype(np.int64)

    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()

    cls_model = fit_cls_model(X, y, fast_train=False)
    y_predicted = torch.argmax(cls_model(X), dim=1)
    torch.save(cls_model.state_dict(), 'classifier_model.pt')
