"""Use the existing dual-arm Piper prompt; keep all 28 pretrained prompt slots."""
import copy

import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset


def validate_piper_prompt(cfg):
    get = lambda key, default=None: OmegaConf.select(cfg, key, default=default)
    required = {
        "joint96_posttrain.enabled": True, "joint96_posttrain.online_piper": True,
        "joint96_posttrain.cached_visuals": False, "local_robot_joint96.enabled": False,
        "model.joint96_three_view": True, "resume": None,
        "local_robot_softprompt.init_checkpoint": None,
        "joint96_posttrain.piper_softprompt.embodiment_id": 14,
        "model.action_dit_config.num_embodiments": 28,
        "model.action_dit_config.soft_prompt_tokens": 8,
        "joint96_posttrain.evaluation_membership": "train_pool",
        "joint96_posttrain.action_video_attention": "current_and_noisy_future_v1",
    }
    for key, expected in required.items():
        if get(key) != expected:
            raise ValueError(f"Piper data interface requires {key}={expected!r}")
    for split in ("train", "val"):
        if get(f"data.{split}.domain") != "piper" or get(f"data.{split}.visual_cache") is not None:
            raise ValueError("Piper requires its online reader and domain")
        if get(f"data.{split}.episode_selection") is not None:
            raise ValueError("RoboTwin periodic episode selection must not be applied to Piper")
        if float(get(f"data.{split}.val_set_proportion", 0)) != 0:
            raise ValueError("Keep the online reference's training-pool diagnostic membership")
        source = get(f"data.{split}.source_root")
        if source is None and get("joint96_posttrain.piper_softprompt.require_source_root", False):
            raise ValueError("piper_folder_posttrain requires data.source_root=datasets/<folder>")
        if source is not None and (not isinstance(source, str) or not source.strip()):
            raise ValueError("Piper source_root must be one original source folder or null")
    if get("data.train.source_root") != get("data.val.source_root"):
        raise ValueError("Piper training and diagnostics must select the same source_root")


class PiperPromptDataset(Dataset):
    def __init__(self, dataset, registry):
        if registry.content["embodiments"][14] != "openneo_aloha_piper" or 14 not in registry.assigned_ids:
            raise ValueError("Expected pretrained dual-arm Piper registry slot 14")
        self.dataset, self.embodiment_registry = dataset, registry
        if hasattr(dataset, "fixed_validation_panel"):
            self.fixed_validation_panel = copy.deepcopy(dataset.fixed_validation_panel)
            for row in self.fixed_validation_panel:
                row["embodiment_id"] = 14

    def __getattr__(self, name):
        dataset = self.__dict__.get("dataset")
        if dataset is None:
            raise AttributeError(name)
        return getattr(dataset, name)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return {**self.dataset[index], "embodiment_id": torch.tensor(14, dtype=torch.long)}


def wrap_piper_prompt(model, train, val, cfg, registry):
    if val is None or train is val:
        raise ValueError("Piper needs distinct training and diagnostic readers")
    model.soft_prompt_training_contract = {
        "domain": "piper", "embodiment_id": 14, "embodiment_name": "openneo_aloha_piper",
        "action_space": "absolute_joint14_radians_to_canonical80", "cached_visuals": False,
        "normalization": str(cfg.data.train.pretrained_norm_stats),
    }
    source = OmegaConf.select(cfg, "data.train.source_root")
    if source is not None:
        model.soft_prompt_training_contract["source_root"] = source
    return tuple(PiperPromptDataset(ds, registry) for ds in (train, val))
