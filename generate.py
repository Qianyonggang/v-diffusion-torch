import json
import math
import os
import uuid
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, Optional

import torch
from PIL import Image
from tqdm import trange

from v_diffusion import *
from v_diffusion.data.sar_dataset import SARDatasetMetadata
from v_diffusion.dpm_solver_v3 import dpm_solver_v3_sample


def load_state_dict(path: str, device: torch.device, use_ema: bool) -> Dict[str, torch.Tensor]:
    state = torch.load(path, map_location=device)
    key = "ema" if use_ema else "model"
    state_dict = state[key]["shadow"] if use_ema else state[key]
    for k in list(state_dict.keys()):
        if k.startswith("module."):
            state_dict[k.split(".", maxsplit=1)[1]] = state_dict.pop(k)
    return state_dict


def build_diffusion(config: Dict, sample_steps: int, w_guide: float) -> GaussianDiffusion:
    diffusion_cfg = config["diffusion"].copy()
    logsnr_schedule = diffusion_cfg.pop("logsnr_schedule")
    logsnr_max = diffusion_cfg.pop("logsnr_max")
    logsnr_min = diffusion_cfg.pop("logsnr_min")
    logsnr_fn = get_logsnr_schedule(
        logsnr_schedule, logsnr_min, logsnr_max, rescale=diffusion_cfg.pop("allow_rescale"))
    diffusion_cfg["sample_timesteps"] = sample_steps
    diffusion_cfg.pop("train_timesteps")
    return GaussianDiffusion(logsnr_fn=logsnr_fn, w_guide=w_guide, **diffusion_cfg)


def prepare_standard_generation(args, config, device, state_dict):
    dataset = config["data"]["name"]
    data_root = args.data_root
    if "~" in data_root:
        data_root = os.path.expanduser(data_root)
    if "$" in data_root:
        data_root = os.path.expandvars(data_root)

    in_channels = DATA_INFO[dataset]["channels"]
    image_res = DATA_INFO[dataset]["resolution"][0]
    multitags = DATA_INFO[dataset].get("multitags", False)
    use_cfg = "class_embed" in {k.split(".")[0] for k in state_dict.keys()}
    if use_cfg:
        num_classes = DATA_INFO[dataset]["num_classes"]
        w_guide = 0. if args.uncond else args.w_guide
    else:
        num_classes = 0
        w_guide = 0

    diffusion = build_diffusion(config, args.sample_timesteps, w_guide)
    model_out_type = config["diffusion"].get("model_out_type", "both")
    out_channels = (2 if model_out_type == "both" else 1) * in_channels
    model = UNet(
        out_channels=out_channels,
        num_classes=num_classes,
        multitags=multitags,
        **config["model"],
    )
    model.to(device)
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        if p.requires_grad:
            p.requires_grad_(False)

    timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S%f")
    exp_name = os.path.splitext(os.path.basename(args.config_path))[0]
    save_dir = os.path.join(args.save_dir, exp_name, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    batch_size = args.batch_size
    total_size = args.total_size
    num_eval_batches = math.ceil(total_size / batch_size)
    shape = (batch_size, in_channels, image_res, image_res)

    with open(os.path.join(save_dir, "args.txt"), "w") as f:
        json.dump(vars(args), f)

    def save_image(arr):
        mode = "L" if in_channels == 1 else "RGB"
        with Image.fromarray(arr, mode=mode) as im:
            im.save(f"{save_dir}/{uuid.uuid4()}.png")

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True

    uncond = args.uncond
    if multitags:
        labels = DATA_INFO[dataset]["data"](root=args.data_root, split="all").targets

        def get_label_loader(to_device):
            while True:
                if uncond:
                    yield torch.zeros((batch_size, num_classes), dtype=torch.float32, device=to_device)
                else:
                    yield labels[torch.randint(len(labels), size=(batch_size, ))].float().to(to_device)
    else:
        def get_label_loader(to_device):
            while True:
                if uncond:
                    yield torch.zeros((batch_size, ), dtype=torch.int64, device=to_device)
                else:
                    yield torch.randint(num_classes, size=(batch_size, ), device=to_device) + 1

    label_loader = get_label_loader(to_device=device)
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        for i in trange(num_eval_batches):
            if i == num_eval_batches - 1:
                batch_size = total_size - i * batch_size
                shape = (batch_size, in_channels, image_res, image_res)

            sample = diffusion.p_sample(
                model, shape=shape, device=device,
                noise=torch.randn(shape, device=device),
                label=next(label_loader)[:batch_size],
                use_ddim=args.use_ddim
            ).cpu()
            sample = (sample * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)
            if in_channels == 1:
                sample = sample.squeeze(1).numpy()
            else:
                sample = sample.permute(0, 2, 3, 1).numpy()
            pool.map(save_image, list(sample))


def resolve_sar_condition(metadata: SARDatasetMetadata, args) -> Dict[str, torch.Tensor]:
    if args.class_name is None:
        raise ValueError("--class-name is required for SAR sampling")
    class_key = args.class_name
    if class_key not in metadata.class_to_id:
        raise KeyError(f"Unknown class name: {class_key}")
    class_id = metadata.class_to_id[class_key]

    if args.angle is None:
        raise ValueError("--angle is required for SAR sampling")
    angle_input = args.angle
    if metadata.angle_bin_size is None:
        angle_key = str(angle_input)
        if angle_key not in metadata.angle_to_id:
            raise KeyError(f"Angle {angle_key} not found in metadata")
        angle_id = metadata.angle_to_id[angle_key]
        angle_label = angle_key
    else:
        angle_value = float(angle_input)
        angle_id = metadata.angle_value_to_id(angle_value)
        angle_label = metadata.angle_id_to_string(angle_id)

    if args.jam_active is None or args.jam_passive is None:
        raise ValueError("--jam-active and --jam-passive are required for SAR sampling")
    jam_a_key = str(args.jam_active)
    jam_p_key = str(args.jam_passive)
    if jam_a_key not in metadata.jam_a_to_id or jam_p_key not in metadata.jam_p_to_id:
        raise KeyError("Invalid jamming indicator provided")
    jam_a_id = metadata.jam_a_to_id[jam_a_key]
    jam_p_id = metadata.jam_p_to_id[jam_p_key]

    batch = args.num_samples_per_cond
    cond = {
        "class_id": torch.full((batch,), class_id, dtype=torch.long),
        "angle_id": torch.full((batch,), angle_id, dtype=torch.long),
        "jam_a_id": torch.full((batch,), jam_a_id, dtype=torch.long),
        "jam_p_id": torch.full((batch,), jam_p_id, dtype=torch.long),
        "cond_mask": torch.ones(batch, dtype=torch.float32),
    }
    labels = {
        "class": class_key,
        "angle": angle_label,
        "jam_a": jam_a_key,
        "jam_p": jam_p_key,
    }
    return cond, labels


def run_sar_generation(args, config, device, state_dict):
    data_cfg = config["data"]
    metadata_path = args.metadata_path or data_cfg.get("metadata_path")
    if metadata_path is None:
        metadata_path = os.path.join(os.path.dirname(args.ckpt_path), "sar_metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
    metadata = SARDatasetMetadata.load_metadata(metadata_path)

    use_cfg = "class_embed" in {k.split(".")[0] for k in state_dict.keys()}
    num_classes = metadata.num_classes if use_cfg else 0
    w_guide = args.w_guide if use_cfg else 0.0
    diffusion = build_diffusion(config, args.sample_timesteps, w_guide)

    model_cfg = config["model"].copy()
    model_cfg["in_channels"] = 1
    model_cfg["out_channels"] = 1 if diffusion.model_out_type != "both" else 2
    model = UNet(
        num_classes=num_classes,
        multitags=False,
        num_angles=metadata.num_angles,
        num_jam_a=metadata.num_jam_a,
        num_jam_p=metadata.num_jam_p,
        **model_cfg,
    )
    model.to(device)
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        if p.requires_grad:
            p.requires_grad_(False)

    cond, labels = resolve_sar_condition(metadata, args)
    batch = args.num_samples_per_cond
    shape = (batch, 1, metadata.image_size, metadata.image_size)

    if args.sampler == "ddim":
        samples = diffusion.p_sample(
            model, shape=shape, device=device, noise=None, label=cond, use_ddim=True
        ).cpu()
    else:
        samples = dpm_solver_v3_sample(
            model, diffusion, shape=shape, device=device, steps=args.sample_timesteps,
            condition=cond
        ).cpu()

    samples = (samples * 127.5 + 127.5).clamp(0, 255).to(torch.uint8).squeeze(1)

    out_root = args.out_dir or args.save_dir
    timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S%f")
    exp_name = os.path.splitext(os.path.basename(args.config_path))[0]
    save_dir = os.path.join(out_root, exp_name, timestamp, labels["class"])
    os.makedirs(save_dir, exist_ok=True)

    angle_str = labels["angle"]
    jam_a_str = labels["jam_a"]
    jam_p_str = labels["jam_p"]
    for idx, img in enumerate(samples, start=1):
        filename = f"{labels['class']}_{angle_str}_{jam_a_str}_{jam_p_str}_{idx:04d}.png"
        with Image.fromarray(img.numpy(), mode="L") as im:
            im.save(os.path.join(save_dir, filename))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-root", type=str, default="~/datasets")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--total-size", type=int, default=50000)
    parser.add_argument("--default-config-path", default="./configs/defaults.json", type=str)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default="./images/eval")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--use-ddim", action="store_true")
    parser.add_argument("--sample-timesteps", type=int, default=1024)
    parser.add_argument("--uncond", action="store_true")
    parser.add_argument("--w-guide", type=float, default=0.1)
    parser.add_argument("--dataset", type=str, choices=list(DATA_INFO.keys()) + ["sar"], help="Override dataset name")
    parser.add_argument("--sampler", choices=["ddim", "dpm_solver_v3"], default="ddim")
    parser.add_argument("--class-name", type=str)
    parser.add_argument("--angle", type=str)
    parser.add_argument("--jam-active", type=str)
    parser.add_argument("--jam-passive", type=str)
    parser.add_argument("--num-samples-per-cond", type=int, default=1)
    parser.add_argument("--out-dir", type=str)
    parser.add_argument("--metadata-path", type=str)

    args = parser.parse_args()
    device = torch.device(args.device)

    with open(args.config_path, "r") as f:
        config = json.load(f)
    with open(args.default_config_path, "r") as f:
        defaults = json.load(f)
    fill_with_defaults(config, defaults)

    if args.dataset is not None:
        config["data"]["name"] = args.dataset
    dataset = config["data"]["name"]

    state_dict = load_state_dict(args.ckpt_path, device, args.use_ema)

    if dataset == "sar":
        run_sar_generation(args, config, device, state_dict)
    else:
        prepare_standard_generation(args, config, device, state_dict)

