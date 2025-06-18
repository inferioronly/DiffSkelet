import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.util import img_as_bool


def get_nonorm_transform(resolution: int):
    return transforms.Compose([
        transforms.Resize((resolution, resolution),
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor()
    ])


class DiffSkeletDataset(Dataset):
    def __init__(self, args, phase: str = "train", transforms=None):
        super().__init__()
        self.root = args.data_root
        self.phase = phase.lower()
        self.transforms = transforms
        self.nonorm_transforms = get_nonorm_transform(args.resolution)

        self._collect_paths()

    def _collect_paths(self):
        self.input_images = []
        self.target_images = []

        input_dir = os.path.join(self.root, self.phase, "input")
        target_dir = os.path.join(self.root, self.phase, "target")

        if not (os.path.isdir(input_dir) and os.path.isdir(target_dir)):
            raise FileNotFoundError(
                f"Expect folders '{input_dir}' & '{target_dir}' to exist.")

        for img_name in os.listdir(input_dir):
            input_path = os.path.join(input_dir, img_name)
            target_path = os.path.join(target_dir, img_name)
            if os.path.exists(target_path):
                self.input_images.append(input_path)
                self.target_images.append(target_path)
            else:
                print(f"[Warning] Missing target for {img_name}")

    @staticmethod
    def _build_skeleton(gray_np: np.ndarray) -> np.ndarray:

        # 1) 降噪
        smoothed = cv2.GaussianBlur(gray_np, (11, 11), 0)
        kernel = np.ones((5, 5), np.uint8)
        morph = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)

        # 2) Skeletonize
        binary_bool = img_as_bool(morph)          # True = 前景
        skel = skeletonize(binary_bool)
        skel_uint8 = (skel * 255).astype(np.uint8)

        # 3) 反色，让前景为白色（与你原脚本一致）
        return cv2.bitwise_not(skel_uint8)

    def __getitem__(self, index: int):
        input_path = self.input_images[index]
        target_path = self.target_images[index]

        input_img = Image.open(input_path).convert("RGB")
        target_img = Image.open(target_path).convert("RGB")
        nonorm_target_img = self.nonorm_transforms(target_img)

        skelet_img = None
        if self.phase == "train":
            gray_np = np.array(target_img.convert("L"))
            skel_np = self._build_skeleton(gray_np)
            skelet_img = Image.fromarray(skel_np, mode="L")

        if self.transforms is not None:
            input_img = self.transforms(input_img)
            target_img = self.transforms(target_img)
            if skelet_img is not None:
                skelet_img = self.transforms(skelet_img)

        sample = {
            "input_image": input_img,
            "target_image": target_img,
            "nonorm_target_image": nonorm_target_img
        }
        if skelet_img is not None:
            sample["skelet_image"] = skelet_img

        return sample

    def __len__(self):
        return len(self.input_images)
