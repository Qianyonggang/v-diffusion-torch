import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def _angle_to_bin(angle_deg: float, angle_bin_size: float, num_bins: int) -> int:
    clipped = min(max(angle_deg, 0.0), 359.9999)
    bin_idx = int(clipped // angle_bin_size)
    return min(bin_idx, num_bins - 1)


def parse_filename(path: str) -> Tuple[str, str, str, str]:
    """Parse dataset filename into metadata fields."""

    name = os.path.splitext(os.path.basename(path))[0]
    parts = name.split("_")
    if len(parts) < 4:
        raise ValueError(f"Invalid SAR filename: {path}")
    class_parts = parts[:-3]
    if not class_parts:
        raise ValueError(f"Class name missing in filename: {path}")
    class_name = "_".join(class_parts)
    angle_str, jam_a_str, jam_p_str = parts[-3:]
    return class_name, angle_str, jam_a_str, jam_p_str


@dataclass
class SARDatasetMetadata:
    num_classes: int
    num_angles: int
    num_jam_a: int
    num_jam_p: int
    image_size: int
    class_to_id: Dict[str, int] = field(default_factory=dict)
    angle_to_id: Dict[str, int] = field(default_factory=dict)
    jam_a_to_id: Dict[str, int] = field(default_factory=dict)
    jam_p_to_id: Dict[str, int] = field(default_factory=dict)
    angle_bin_size: Optional[float] = None
    angle_id_to_label: Dict[int, str] = field(default_factory=dict)
    id_to_class: Dict[int, str] = field(default_factory=dict)
    jam_a_id_to_label: Dict[int, str] = field(default_factory=dict)
    jam_p_id_to_label: Dict[int, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "num_classes": self.num_classes,
            "num_angles": self.num_angles,
            "num_jam_a": self.num_jam_a,
            "num_jam_p": self.num_jam_p,
            "image_size": self.image_size,
            "class_to_id": self.class_to_id,
            "angle_to_id": self.angle_to_id,
            "jam_a_to_id": self.jam_a_to_id,
            "jam_p_to_id": self.jam_p_to_id,
            "angle_bin_size": self.angle_bin_size,
            "angle_id_to_label": {str(k): v for k, v in self.angle_id_to_label.items()},
            "id_to_class": {str(k): v for k, v in self.id_to_class.items()},
            "jam_a_id_to_label": {str(k): v for k, v in self.jam_a_id_to_label.items()},
            "jam_p_id_to_label": {str(k): v for k, v in self.jam_p_id_to_label.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "SARDatasetMetadata":
        meta = cls(
            num_classes=data["num_classes"],
            num_angles=data["num_angles"],
            num_jam_a=data["num_jam_a"],
            num_jam_p=data["num_jam_p"],
            image_size=data["image_size"],
            class_to_id=data["class_to_id"],
            angle_to_id=data["angle_to_id"],
            jam_a_to_id=data["jam_a_to_id"],
            jam_p_to_id=data["jam_p_to_id"],
            angle_bin_size=data.get("angle_bin_size"),
        )
        meta.angle_id_to_label = {int(k): v for k, v in data.get("angle_id_to_label", {}).items()}
        meta.id_to_class = {int(k): v for k, v in data.get("id_to_class", {}).items()}
        meta.jam_a_id_to_label = {int(k): v for k, v in data.get("jam_a_id_to_label", {}).items()}
        meta.jam_p_id_to_label = {int(k): v for k, v in data.get("jam_p_id_to_label", {}).items()}
        return meta

    def save_metadata(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_metadata(cls, path: str) -> "SARDatasetMetadata":
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def angle_str_to_id(self, angle_str: str) -> int:
        if angle_str in self.angle_to_id:
            return self.angle_to_id[angle_str]
        if self.angle_bin_size is None:
            raise KeyError(f"Angle string {angle_str} not found in metadata")
        angle_deg = float(angle_str)
        num_bins = math.ceil(360.0 / float(self.angle_bin_size))
        return _angle_to_bin(angle_deg, float(self.angle_bin_size), num_bins)

    def angle_value_to_id(self, angle_value: float) -> int:
        if self.angle_bin_size is None:
            angle_str = str(angle_value)
            return self.angle_str_to_id(angle_str)
        num_bins = math.ceil(360.0 / float(self.angle_bin_size))
        return _angle_to_bin(float(angle_value), float(self.angle_bin_size), num_bins)

    def angle_id_to_string(self, angle_id: int) -> str:
        if angle_id in self.angle_id_to_label:
            return self.angle_id_to_label[angle_id]
        return str(angle_id)

    def class_id_to_string(self, class_id: int) -> str:
        return self.id_to_class.get(class_id, str(class_id))

    def jam_active_id_to_string(self, jam_id: int) -> str:
        return self.jam_a_id_to_label.get(jam_id, str(jam_id))

    def jam_passive_id_to_string(self, jam_id: int) -> str:
        return self.jam_p_id_to_label.get(jam_id, str(jam_id))


class SARDataset(Dataset):
    """Dataset wrapper for SAR conditional generation."""

    def __init__(
            self,
            root: str,
            image_size: int = 256,
            center_crop: bool = True,
            random_flip: bool = False,
            angle_bin_size: Optional[float] = None,
            metadata: Optional[SARDatasetMetadata] = None,
    ) -> None:
        super().__init__()
        self.root = os.path.abspath(os.path.expanduser(root))
        self.image_size = image_size
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.angle_bin_size = angle_bin_size if metadata is None else metadata.angle_bin_size

        self.image_paths = sorted(str(p) for p in Path(self.root).rglob("*.png"))
        if not self.image_paths:
            raise FileNotFoundError(f"No PNG files found under {self.root}")

        self.items: List[Dict[str, object]] = []
        if metadata is None:
            metadata = self._build_metadata()
        self.metadata = metadata

        for path in self.image_paths:
            class_name, angle_str, jam_a_str, jam_p_str = parse_filename(path)
            class_id = metadata.class_to_id[class_name]
            angle_id = metadata.angle_to_id.get(angle_str)
            if angle_id is None:
                if metadata.angle_bin_size is not None:
                    angle_id = metadata.angle_value_to_id(float(angle_str))
                else:
                    raise KeyError(f"Angle {angle_str} not recognized for {path}")
            jam_a_id = metadata.jam_a_to_id[jam_a_str]
            jam_p_id = metadata.jam_p_to_id[jam_p_str]
            self.items.append({
                "path": path,
                "class_id": class_id,
                "angle_id": angle_id,
                "jam_a_id": jam_a_id,
                "jam_p_id": jam_p_id,
                "angle_str": angle_str,
                "jam_a_str": jam_a_str,
                "jam_p_str": jam_p_str,
            })

        crop_transform: List[transforms.Compose] = []
        if image_size:
            crop_transform.append(transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR))
            if center_crop:
                crop_transform.append(transforms.CenterCrop(image_size))
            else:
                crop_transform.append(transforms.RandomCrop(image_size))
        if random_flip:
            crop_transform.append(transforms.RandomHorizontalFlip())
        crop_transform.append(transforms.ToTensor())
        self.transform = transforms.Compose(crop_transform)

    def _build_metadata(self) -> SARDatasetMetadata:
        classes: List[str] = []
        angles: List[str] = []
        jam_a_vals: List[str] = []
        jam_p_vals: List[str] = []
        for path in self.image_paths:
            class_name, angle_str, jam_a_str, jam_p_str = parse_filename(path)
            classes.append(class_name)
            angles.append(angle_str)
            jam_a_vals.append(jam_a_str)
            jam_p_vals.append(jam_p_str)

        class_names = sorted(set(classes))
        class_to_id = {name: idx for idx, name in enumerate(class_names)}

        jam_a_values = sorted(set(jam_a_vals))
        jam_p_values = sorted(set(jam_p_vals))
        jam_a_to_id = {val: idx for idx, val in enumerate(jam_a_values)}
        jam_p_to_id = {val: idx for idx, val in enumerate(jam_p_values)}

        angle_to_id: Dict[str, int] = {}
        angle_id_to_label: Dict[int, str] = {}
        if self.angle_bin_size is None:
            try:
                sorted_angles = sorted(set(angles), key=lambda x: float(x))
            except ValueError:
                sorted_angles = sorted(set(angles))
            for idx, angle in enumerate(sorted_angles):
                angle_to_id[angle] = idx
                angle_id_to_label[idx] = angle
            num_angles = len(sorted_angles)
        else:
            num_bins = math.ceil(360.0 / float(self.angle_bin_size))
            for angle in set(angles):
                try:
                    angle_value = float(angle)
                except ValueError as exc:
                    raise ValueError(f"Angle {angle} is not numeric while binning is enabled") from exc
                bin_idx = _angle_to_bin(angle_value, float(self.angle_bin_size), num_bins)
                angle_to_id[angle] = bin_idx
                if bin_idx not in angle_id_to_label:
                    angle_id_to_label[bin_idx] = f"{(bin_idx + 0.5) * float(self.angle_bin_size):.4g}"
            num_angles = max(angle_to_id.values()) + 1

        id_to_class = {idx: name for name, idx in class_to_id.items()}
        jam_a_id_to_label = {idx: val for val, idx in jam_a_to_id.items()}
        jam_p_id_to_label = {idx: val for val, idx in jam_p_to_id.items()}

        return SARDatasetMetadata(
            num_classes=len(class_to_id),
            num_angles=num_angles,
            num_jam_a=len(jam_a_to_id),
            num_jam_p=len(jam_p_to_id),
            image_size=self.image_size,
            class_to_id=class_to_id,
            angle_to_id=angle_to_id,
            jam_a_to_id=jam_a_to_id,
            jam_p_to_id=jam_p_to_id,
            angle_bin_size=self.angle_bin_size,
            angle_id_to_label=angle_id_to_label,
            id_to_class=id_to_class,
            jam_a_id_to_label=jam_a_id_to_label,
            jam_p_id_to_label=jam_p_id_to_label,
        )

    @property
    def num_classes(self) -> int:
        return self.metadata.num_classes

    @property
    def num_angles(self) -> int:
        return self.metadata.num_angles

    @property
    def num_jam_a(self) -> int:
        return self.metadata.num_jam_a

    @property
    def num_jam_p(self) -> int:
        return self.metadata.num_jam_p

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self.items[index]
        with Image.open(item["path"]).convert("L") as img:
            pixel_values = self.transform(img)
        pixel_values = pixel_values * 2.0 - 1.0
        return {
            "images": pixel_values,
            "class_id": torch.tensor(item["class_id"], dtype=torch.long),
            "angle_id": torch.tensor(item["angle_id"], dtype=torch.long),
            "jam_a_id": torch.tensor(item["jam_a_id"], dtype=torch.long),
            "jam_p_id": torch.tensor(item["jam_p_id"], dtype=torch.long),
        }

    def sample_condition_batch(self, num_samples: int, seed: Optional[int] = None) -> Dict[str, torch.Tensor]:
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)
        indices = torch.randint(len(self.items), size=(num_samples,), generator=generator)
        class_ids = torch.tensor([self.items[i]["class_id"] for i in indices], dtype=torch.long)
        angle_ids = torch.tensor([self.items[i]["angle_id"] for i in indices], dtype=torch.long)
        jam_a_ids = torch.tensor([self.items[i]["jam_a_id"] for i in indices], dtype=torch.long)
        jam_p_ids = torch.tensor([self.items[i]["jam_p_id"] for i in indices], dtype=torch.long)
        return {
            "class_id": class_ids,
            "angle_id": angle_ids,
            "jam_a_id": jam_a_ids,
            "jam_p_id": jam_p_ids,
        }


__all__ = ["SARDataset", "SARDatasetMetadata", "parse_filename"]

