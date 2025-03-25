import random
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from albumentations import Compose, DualTransform, PadIfNeeded, RandomCrop
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from tqdm import tqdm

from utils.datasets import CocoLvisDataset
from utils.misc import draw_points, draw_probmap, save_checkpoint
from utils.points_sampler import MultiPointSampler
from torch.optim.lr_scheduler import StepLR


class DistMaps(torch.nn.Module):
    def __init__(self, norm_radius=5, spatial_scale=1.0, use_disks=True):
        super().__init__()
        self.spatial_scale = spatial_scale
        self.norm_radius = norm_radius
        self.use_disks = use_disks

    def compute_map(self, clicks, rows, cols, B):
        """
        Вычисляет карту расстояний для группы кликов.
        clicks: тензор размера (B, M, 3), где первые два элемента каждой тройки — координаты.
        Возвращает карту размера (B, 1, rows, cols).
        """
        # clicks: (B, M, 3)
        B, M, _ = clicks.shape
        clicks = clicks.reshape(-1, clicks.size(2))
        clicks_xy, _ = torch.split(clicks, [2, 1], dim=1)
        invalid = torch.max(clicks_xy, dim=1, keepdim=False)[0] < 0

        row_array = torch.arange(0, rows, dtype=torch.float32, device=clicks.device)
        col_array = torch.arange(0, cols, dtype=torch.float32, device=clicks.device)
        rr, cc = torch.meshgrid(row_array, col_array, indexing="ij")
        coords = torch.stack((rr, cc), dim=0)
        coords = coords.unsqueeze(0).repeat(clicks_xy.size(0), 1, 1, 1)

        add_xy = clicks_xy * self.spatial_scale
        add_xy = add_xy.reshape(clicks_xy.size(0), 2, 1, 1)
        coords = coords - add_xy
        coords = coords * coords
        coords[:, 0] += coords[:, 1]
        coords = coords[:, :1]

        coords[invalid, :, :, :] = 1e6

        coords = coords.reshape(B, M, 1, rows, cols)
        dmap = coords.min(dim=1)[0]

        if self.use_disks:
            dmap = (dmap <= (self.norm_radius * self.spatial_scale) ** 2).float()
        else:
            dmap = dmap.sqrt().mul(2).tanh()
        return dmap

    def forward(self, x, coords):
        """
        x: тензор изображения размера (B, 3, H, W)
        coords: тензор кликов размера (B, 2*M, 3), где первые M – положительные,
                а оставшиеся – отрицательные клики.
        Возвращает карту из 2 каналов: (B, 2, H, W)
        """
        B, twoM, _ = coords.shape
        M = twoM // 2
        pos_clicks = coords[:, :M, :]
        neg_clicks = coords[:, M:, :]

        rows, cols = x.shape[2], x.shape[3]
        pos_map = self.compute_map(pos_clicks, rows, cols, B)
        neg_map = self.compute_map(neg_clicks, rows, cols, B)
        return torch.cat([pos_map, neg_map], dim=1)



class BatchImageNormalize:
    def __init__(self, mean, std, dtype=torch.float):
        self.mean = torch.as_tensor(mean, dtype=dtype)[None, :, None, None]
        self.std = torch.as_tensor(std, dtype=dtype)[None, :, None, None]

    def __call__(self, tensor):
        tensor = tensor.clone()
        tensor.sub_(self.mean.to(tensor.device)).div_(self.std.to(tensor.device))
        return tensor


def get_next_points(pred, gt, points, click_indx, pred_thresh=0.5):
    """
    Симулируем постановку нового клика в область с наибольшей ошибкой.
    """
    assert click_indx > 0
    pred = pred.detach().cpu().numpy()[:, 0, :, :]
    gt = gt.cpu().detach().numpy()[:, 0, :, :] > 0.5

    fn_mask = gt & (pred < pred_thresh)
    fp_mask = ~gt & (pred > pred_thresh)

    fn_mask = np.pad(fn_mask, ((0, 0), (1, 1), (1, 1)), "constant").astype(np.uint8)
    fp_mask = np.pad(fp_mask, ((0, 0), (1, 1), (1, 1)), "constant").astype(np.uint8)
    num_points = points.size(1) // 2
    points = points.clone()

    for bindx in range(fn_mask.shape[0]):
        fn_mask_dt = cv2.distanceTransform(fn_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]
        fp_mask_dt = cv2.distanceTransform(fp_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist
        dt = fn_mask_dt if is_positive else fp_mask_dt
        inner_mask = dt > max(fn_max_dist, fp_max_dist) / 2.0
        indices = np.argwhere(inner_mask)
        if len(indices) > 0:
            coords = indices[np.random.randint(0, len(indices))]
            if is_positive:
                points[bindx, num_points - click_indx, 0] = float(coords[0])
                points[bindx, num_points - click_indx, 1] = float(coords[1])
                points[bindx, num_points - click_indx, 2] = float(click_indx)
            else:
                points[bindx, 2 * num_points - click_indx, 0] = float(coords[0])
                points[bindx, 2 * num_points - click_indx, 1] = float(coords[1])
                points[bindx, 2 * num_points - click_indx, 2] = float(click_indx)

    return points


class ISModel(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.normalization = BatchImageNormalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        )

        self.coord_feature_ch = 3
        self.dist_maps = DistMaps(norm_radius=5, spatial_scale=1.0, use_disks=True)

        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        self.feature_extractor = deeplabv3_mobilenet_v3_large(
            num_classes=1,
            weights_backbone=weights,
        )

        old_conv = self.feature_extractor.backbone["0"][0]
        new_conv = nn.Sequential(
            nn.Conv2d(
                old_conv.in_channels + self.coord_feature_ch,
                old_conv.out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(old_conv.out_channels),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(
                old_conv.out_channels,
                old_conv.out_channels,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                bias=False,
            )
        )
        self.feature_extractor.backbone["0"][0] = new_conv

        self.feature_extractor.classifier[0].project[3] = nn.Identity()

        self.pred_thr = 0.5



    def forward(self, image, points):
        image, prev_mask = self.prepare_input(image)
        coord_features = self.get_coord_features(image, prev_mask, points)
        outputs = self.backbone_forward(image, coord_features)
        return outputs

    def prepare_input(self, image):
        prev_mask = image[:, 3:, :, :]
        image = image[:, :3, :, :]
        image = self.normalization(image)
        return image, prev_mask

    def backbone_forward(self, image, coord_features):
        net_input = torch.cat((image, coord_features), dim=1)
        net_outputs = self.feature_extractor(net_input)["out"]
        return {"instances": net_outputs}

    def get_coord_features(self, image, prev_mask, points):
        coord_features = self.dist_maps(image, points)
        coord_features = torch.cat((prev_mask, coord_features), dim=1)
        return coord_features

    def restore_from_checkpoint(self, checkpoint_path, device):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.load_state_dict(checkpoint["state_dict"])
        return self



class Predictor:
    def __init__(self, model, device):
        self.original_image = None
        self.device = device
        self.prev_prediction = None
        self.net = model
        self.to_tensor = torch.nn.Sequential()

    def set_input_image(self, image):
        # image: H,W,3 np.array (RGB)
        image_nd = torch.from_numpy(image.transpose(2, 0, 1) / 255.0).float()
        self.original_image = image_nd.unsqueeze(0).to(self.device)
        self.prev_prediction = torch.zeros_like(self.original_image[:, :1, :, :])

    def get_prediction(self, clicker, prev_mask=None):
        clicks_list = deepcopy(clicker.get_clicks())

        input_image = self.original_image
        if prev_mask is None:
            prev_mask = self.prev_prediction

        input_image = torch.cat((input_image, prev_mask), dim=1)
        prev_size = input_image.shape[2:]
        new_size = (320, 320)
        input_image = F.interpolate(input_image, new_size, mode="bilinear", align_corners=True)

        for click in clicks_list:
            click.coords = (
                click.coords[0] / (prev_size[0] / new_size[0]),
                click.coords[1] / (prev_size[1] / new_size[1]),
            )

        prediction = self._get_prediction(input_image, [clicks_list])
        prediction = F.interpolate(prediction, prev_size, mode="bilinear", align_corners=True)
        prediction = torch.sigmoid(prediction)
        self.prev_prediction = prediction
        return prediction.cpu().numpy()[0, 0]

    def _get_prediction(self, image_nd, clicks_lists):
        points_nd = self.get_points_nd(clicks_lists)
        return self.net(image_nd, points_nd)["instances"]

    def get_points_nd(self, clicks_lists):
        total_clicks = []
        num_pos_clicks = [sum(click.is_positive for click in clicks_list) for clicks_list in clicks_lists]
        num_neg_clicks = [len(clicks_list) - num_pos for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)]
        num_max_points = max(num_pos_clicks + num_neg_clicks)
        num_max_points = max(1, num_max_points)

        for clicks_list in clicks_lists:
            neg_clicks, pos_clicks = [], []
            for click in clicks_list:
                if click.is_positive:
                    pos_clicks.append(click.coords_and_indx)
                else:
                    neg_clicks.append(click.coords_and_indx)
            pos_padding = num_max_points - len(pos_clicks)
            pos_clicks += pos_padding * [(-1, -1, -1)]
            neg_padding = num_max_points - len(neg_clicks)
            neg_clicks += neg_padding * [(-1, -1, -1)]
            total_clicks.append(pos_clicks + neg_clicks)

        return torch.tensor(total_clicks, device=self.device, dtype=torch.float32)


class ISTrainer:
    def __init__(self, model, cfg, instance_loss, trainset, valset,
                 image_dump_interval=10, checkpoint_interval=10,
                 max_initial_points=0, max_interactive_clicks=0):
        self.cfg = cfg
        self.max_initial_points = max_initial_points
        self.instance_loss = instance_loss
        self.max_interactive_clicks = max_interactive_clicks
        self.checkpoint_interval = checkpoint_interval
        self.image_dump_interval = image_dump_interval
        self.trainset = trainset
        self.valset = valset

        train_size = trainset.get_samples_number()
        print(f"Dataset of {train_size} samples was loaded for training.")
        val_size = valset.get_samples_number()
        print(f"Dataset of {val_size} samples was loaded for validation.")

        self.train_data = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True,
                                     drop_last=True, pin_memory=True, num_workers=4)
        self.val_data = DataLoader(valset, batch_size=cfg.val_batch_size, shuffle=False,
                                   drop_last=True, pin_memory=True, num_workers=4)

        self.device = cfg.device
        self.net = model.to(self.device)
        self.optim = torch.optim.SGD(self.net.parameters(), lr=1e-5, momentum=0.5, weight_decay=1e-7)
        self.scaler = GradScaler()
        self.scheduler = StepLR(self.optim, step_size=10, gamma=0.1)

    def run(self, num_epochs, validation=True):
        print(f"Total Epochs: {num_epochs}")
        for epoch in range(num_epochs):
            self.training(epoch)
            if validation:
                self.validation(epoch)

    def training(self, epoch):
        tbar = tqdm(self.train_data, ncols=100)
        self.net.train()
        train_loss = 0.0

        for i, batch_data in enumerate(tbar):
            global_step = epoch * len(self.train_data) + i
            with autocast():
                loss, splitted_batch_data, outputs = self.batch_forward(batch_data)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0) # !!!!!
            self.scaler.step(self.optim)
            self.scaler.update()
            train_loss += loss.item()
            if self.image_dump_interval > 0 and global_step % self.image_dump_interval == 0:
                self.save_visualization(splitted_batch_data, outputs, global_step, prefix="train")
            tbar.set_description(f"Epoch {epoch}, training loss {train_loss/(i+1):.4f}")

        save_checkpoint(self.net, self.cfg.CHECKPOINTS_PATH, epoch=None)
        if epoch % self.checkpoint_interval == 0:
            save_checkpoint(self.net, self.cfg.CHECKPOINTS_PATH, epoch=epoch)

    def validation(self, epoch):
        tbar = tqdm(self.val_data, ncols=100)
        self.net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, batch_data in enumerate(tbar):
                with autocast():
                    loss, _, _ = self.batch_forward(batch_data, validation=True)
                val_loss += loss.item()
                tbar.set_description(f"Epoch {epoch}, validation loss: {val_loss/(i+1):.4f}")

    def batch_forward(self, batch_data, validation=False):
        with torch.set_grad_enabled(not validation):
            batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
            image, gt_mask, points = batch_data["images"], batch_data["instances"], batch_data["points"]
            prev_output = torch.zeros_like(image[:, :1, :, :], dtype=torch.float32)
            num_iters = random.randint(0, self.max_interactive_clicks)
            for click_indx in range(num_iters):
                if not validation:
                    self.net.eval()
                net_input = torch.cat((image, prev_output), dim=1)
                logits = self.net(net_input, points)["instances"]
                prev_output = torch.sigmoid(logits)
                points = get_next_points(prev_output, gt_mask, points, click_indx + 1)
                if not validation:
                    self.net.train()
            net_input = torch.cat((image, prev_output), dim=1)
            output = self.net(net_input, points)
            logits = output["instances"]
            loss = self.instance_loss(logits, gt_mask)
            loss = torch.mean(loss)
        return loss, batch_data, output

    def save_visualization(self, splitted_batch_data, outputs, global_step, prefix):
        output_images_path = self.cfg.VIS_PATH / prefix
        output_images_path.mkdir(parents=True, exist_ok=True)
        image_name_prefix = f"{global_step:06d}"

        def _save_image(suffix, image):
            cv2.imwrite(str(output_images_path / f"{image_name_prefix}_{suffix}.jpg"),
                        image, [cv2.IMWRITE_JPEG_QUALITY, 85])

        images = splitted_batch_data["images"]
        points = splitted_batch_data["points"]
        instance_masks = splitted_batch_data["instances"]
        gt_instance_masks = instance_masks.cpu().numpy()
        predicted_instance_masks = torch.sigmoid(outputs["instances"]).detach().cpu().numpy()
        points = points.detach().cpu().numpy()
        image_blob, points = images[0], points[0]
        gt_mask = np.squeeze(gt_instance_masks[0], axis=0)
        predicted_mask = np.squeeze(predicted_instance_masks[0], axis=0)
        image = image_blob.cpu().numpy()
        rgb_image = (image[:3] * 255).transpose(1, 2, 0)
        image_with_points = draw_points(rgb_image, points[:self.max_initial_points], (0, 255, 0))
        image_with_points = draw_points(image_with_points, points[self.max_initial_points:], (0, 0, 255))
        gt_mask_vis = draw_probmap(gt_mask)
        predicted_mask_vis = draw_probmap(predicted_mask)
        viz_image = np.hstack((image_with_points, gt_mask_vis, predicted_mask_vis))
        viz_image = viz_image.astype(np.uint8)
        _save_image("instance_segmentation", viz_image[:, :, ::-1])



class UniformRandomResize(DualTransform):
    def __init__(self, scale_range, interpolation=cv2.INTER_LINEAR, always_apply=True, p=1):
        super().__init__(always_apply, p)
        self.scale_range = scale_range
        self.interpolation = interpolation

    def get_params_dependent_on_targets(self, params):
        scale = random.uniform(*self.scale_range)
        height = int(round(params["image"].shape[0] * scale))
        width = int(round(params["image"].shape[1] * scale))
        return {"new_height": height, "new_width": width}

    def apply(self, img, new_height=0, new_width=0, interpolation=cv2.INTER_LINEAR, **params):
        resize_op = A.augmentations.geometric.resize.Resize(height=new_height, width=new_width, interpolation=interpolation)
        return resize_op(image=img)["image"]

    def apply_to_keypoint(self, keypoint, new_height=0, new_width=0, **params):
        scale_x = new_width / params["cols"]
        scale_y = new_height / params["rows"]
        keypoint = A.augmentations.geometric.functional.keypoint_scale(keypoint, scale_x, scale_y)
        return keypoint

    def apply_to_bbox(self, *args, **kwargs):
        raise NotImplementedError()

    def get_transform_init_args_names(self):
        return ("scale_range", "interpolation")

    @property
    def targets_as_params(self):
        return ["image"]


def train_segmentation():
    input_size = (320, 320)
    model = ISModel(pretrained=True)

    cfg = SimpleNamespace()
    exp_path = Path("./experiments_fast")
    cfg.EXP_PATH = exp_path
    cfg.CHECKPOINTS_PATH = exp_path / "checkpoints"
    cfg.VIS_PATH = exp_path / "vis"
    cfg.EXP_PATH.mkdir(parents=True, exist_ok=True)
    cfg.CHECKPOINTS_PATH.mkdir(exist_ok=True)
    cfg.VIS_PATH.mkdir(exist_ok=True)

    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.max_initial_points = 24
    cfg.batch_size = 4 # 8???
    cfg.val_batch_size = cfg.batch_size

    instance_loss = nn.BCEWithLogitsLoss()

    train_epoch_len = 15000
    val_epoch_len = 1000

    train_augmentator = Compose([
        UniformRandomResize(scale_range=(0.75, 1.30)),
        PadIfNeeded(min_height=input_size[0], min_width=input_size[1], border_mode=cv2.BORDER_CONSTANT, value=0),
        RandomCrop(*input_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.ColorJitter(p=0.5),
        A.GaussianBlur(p=0.5, blur_limit=3)
    ], p=1.0)

    val_augmentator = Compose([
        PadIfNeeded(min_height=input_size[0], min_width=input_size[1], border_mode=cv2.BORDER_CONSTANT, value=0),
        RandomCrop(*input_size),
    ], p=1.0)

    points_sampler = MultiPointSampler(cfg.max_initial_points, prob_gamma=0.80, merge_objects_prob=0.15, max_num_merged_objects=2)

    trainset = CocoLvisDataset("./COCO_LVIS", split="train", augmentator=train_augmentator,
                                min_object_area=1000, points_sampler=points_sampler, epoch_len=train_epoch_len, stuff_prob=0.30)
    valset = CocoLvisDataset("./COCO_LVIS", split="val", augmentator=val_augmentator,
                              min_object_area=1000, points_sampler=points_sampler, epoch_len=val_epoch_len)

    trainer = ISTrainer(model, cfg, instance_loss, trainset, valset,
                         checkpoint_interval=1, image_dump_interval=1000,
                         max_initial_points=cfg.max_initial_points, max_interactive_clicks=3)
    trainer.run(num_epochs=100)


if __name__ == "__main__":
    train_segmentation()
