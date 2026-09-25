import os
from copy import deepcopy
from typing import Dict, List, Any, Optional, Union, Tuple

import torch
import scipy
import numpy as np
import math
import random
from pathlib import Path

from torch.utils.data import Dataset, Sampler

PAD_VALUE = -100


DATASET_TO_IDX = {
    "brandman_2024_text": 0,
    "willett_2023_text": 1,
    # === Human subjects ===
    # Canonicals
    "willet_t12":           2,   # 256 ch — canonical for willet group
    "card_t15":             3,   # 512 ch — canonical for card group
    "kunz_t16":             4,   # 128 ch — separate
    "kunz_t17":             5,   # 256 ch — separate
    "willett_handwriting":  6,   # 192 ch
    "fan_handwriting":      7,   # 192 ch — canonical for T5 group
    "wairagkar":            8,   # 512 ch
    "jude_speech_anarthia": 9,   # 256 ch
    "jude_typing_t17":      10,  # 768 ch
    "jude_typing_t18":      11,  # 768 ch
    # Aliases → existing canonicals
    "kunz_t12":             2,   # 256 ch → willet_t12
    "kunz_t15":             3,   # 512 ch → card_t15
    # === Neural pile canonicals (indices 12–61) ===
    "neural_pile_M1_wojcik":                 12,
    "neural_pile_M2_wojcik":                 13,
    "neural_pile_Mahler_rajalingham":        14,
    "neural_pile_Perle_rajalingham":         15,
    "neural_pile_Router_lanzarini":          16,
    "neural_pile_T15_card":                  17,  # 256 ch — separate
    "neural_pile_Wifi_lanzarini":            18,
    "neural_pile_indy_makin":                19,
    "neural_pile_loco_makin":                20,
    "neural_pile_monkey_F_papale":           21,
    "neural_pile_monkey_N_papale":           22,
    "neural_pile_sub-An_xiao":               23,
    "neural_pile_sub-Bf_xiao":               24,
    "neural_pile_sub-Bo_xiao":               25,
    "neural_pile_sub-C_perich":              26,
    "neural_pile_sub-Fr_xiao":               27,
    "neural_pile_sub-Han_area2-bump":        28,
    "neural_pile_sub-Haydn_dmfc-rsg":        29,
    "neural_pile_sub-HumanPitt":             30,  # 176 ch — canonical for HumanPitt sessions
    "neural_pile_sub-J_perich":              31,
    "neural_pile_sub-JenkinsC_even-chen":    32,
    "neural_pile_sub-Jenkins_churchland":    33,
    "neural_pile_sub-Lalo_chen":             34,
    "neural_pile_sub-Lo_xiao":               35,
    "neural_pile_sub-MG_moore":              36,
    "neural_pile_sub-M_perich":              37,
    "neural_pile_sub-Monkey-N_temmar":       38,
    "neural_pile_sub-MonkeyL":               39,  # 64 ch — canonical for MonkeyL sessions
    "neural_pile_sub-MonkeyN":               40,  # 96 ch — canonical for MonkeyN sessions
    "neural_pile_sub-MonkeyX":               41,  # 64 ch — canonical for MonkeyX sessions
    "neural_pile_sub-Na_xiao":               42,
    "neural_pile_sub-Nitschke_churchland":   43,
    "neural_pile_sub-Oc_xiao":               44,
    "neural_pile_sub-Offenbach_chen":        45,
    "neural_pile_sub-Ot_xiao":               46,
    "neural_pile_sub-Pa_xiao":               47,
    "neural_pile_sub-Re_xiao":               48,
    "neural_pile_sub-Reggie_even-chen":      49,
    "neural_pile_sub-Sw_xiao":               50,
    "neural_pile_sub-T_perich":              51,
    "neural_pile_sub-Ve_xiao":               52,
    "neural_pile_sub-Ye_xiao":               53,
    "neural_pile_sub-Z08115_kim":            54,
    "neural_pile_sub-Z11111_kim":            55,
    "neural_pile_sub-amadeus_neupane-entorhinal": 56,
    "neural_pile_sub-amadeus_neupane-ppc":        57,
    "neural_pile_sub-mahler_neupane-entorhinal":  58,
    "neural_pile_sub-mahler_neupane-ppc":         59,
    "neural_pile_sub-monk-g_athalye":             60,
    "neural_pile_sub-monk-j_athalye":             61,
    # New canonicals for neural pile session groups
    "neural_pile_willett":                        62,  # 256 ch — canonical for 3 willett sessions
    "neural_pile_sub-T5":                         63,  # 192 ch — canonical for 3 T5 sessions
    # === Neural pile internal aliases (session → canonical, same subject) ===
    "neural_pile_diagnosticBlocks_willett":       62,  # → neural_pile_willett
    "neural_pile_sentences_willett":              62,  # → neural_pile_willett
    "neural_pile_tuningTasks_willett":            62,  # → neural_pile_willett
    "neural_pile_sub-T5-held-in-calib_h2":       63,  # → neural_pile_sub-T5
    "neural_pile_sub-T5-held-in-minival_h2":     63,  # → neural_pile_sub-T5
    "neural_pile_sub-T5-held-out-calib_h2":      63,  # → neural_pile_sub-T5
    "neural_pile_sub-HumanPitt-held-in-calib_h1":    30,  # → neural_pile_sub-HumanPitt
    "neural_pile_sub-HumanPitt-held-in-minival_h1":  30,
    "neural_pile_sub-HumanPitt-held-out-calib_h1":   30,
    "neural_pile_sub-MonkeyL-held-in-calib_m1-a":    39,  # → neural_pile_sub-MonkeyL
    "neural_pile_sub-MonkeyL-held-in-minival_m1-a":  39,
    "neural_pile_sub-MonkeyL-held-out-calib_m1-a":   39,
    "neural_pile_sub-MonkeyN-held-in-calib_m2":      40,  # → neural_pile_sub-MonkeyN
    "neural_pile_sub-MonkeyN-held-out-calib_m2":     40,
    "neural_pile_sub-MonkeyX-held-in-calib_m1-b":    41,  # → neural_pile_sub-MonkeyX
    "neural_pile_sub-MonkeyX-held-in-minival_m1-b":  41,
    "neural_pile_sub-MonkeyX-held-out-calib_m1-b":   41,
    # === New primate datasets — aliases to existing neural pile entries ===
    # (same subject → same idx → shared ReadIn/ReadOut weights)
    "temmar_Monkey-N":          38,  # → neural_pile_sub-Monkey-N_temmar   96ch
    "evenchen_sub-JenkinsC":    32,  # → neural_pile_sub-JenkinsC_even-chen 192ch
    "evenchen_sub-Reggie":      49,  # → neural_pile_sub-Reggie_even-chen  192ch
    "churchland_sub-Jenkins":   33,  # → neural_pile_sub-Jenkins_churchland 192ch
    "churchland_sub-Nitschke":  43,  # → neural_pile_sub-Nitschke_churchland 192ch
    "odoherty_indy_96":         19,  # → neural_pile_indy_makin             96ch
    "odoherty_loco_192":        20,  # → neural_pile_loco_makin            192ch
    # === New primate datasets — no neural pile equivalent (new idx 64+) ===
    "chowdhury_C_57":          64,   #  57ch
    "chowdhury_C_74":          65,   #  74ch
    "chowdhury_C_82":          66,   #  82ch
    "chowdhury_C_98":          67,   #  98ch
    "chowdhury_C_118":         68,   # 118ch
    "chowdhury_H_102":         69,   # 102ch
    "chowdhury_H_113":         70,   # 113ch
    "chowdhury_H_122":         71,   # 122ch
    "chowdhury_H_155":         72,   # 155ch
    "chowdhury_H_167":         73,   # 167ch
    "chowdhury_L_42":          74,   #  42ch
    "chowdhury_L_58":          75,   #  58ch
    "ma_Chewie_CO_2016":       76,   #  92ch (common channels across 12 sess)
    "ma_Greyson_Key_2019":     77,   #  96ch
    "ma_Jango_ISO_2015":       78,   #  85ch
    "ma_Mihili_CO_2014":       79,   #  94ch
    "ma_Mihili_RT_2013_2014":  80,   #  94ch
    "ma_Spike_ISO_2012":       81,   #  73ch
    "odoherty_indy_192":       82,   # 192ch (M1+S1, distinct from indy_96)
    # Perich per-session entries: idx 83–193 (loaded dynamically from registry)
    # NEJM cursor-control human subjects
    "cursor_karpowicz":       200,   # 192ch
    "cursor_singer_clark":    201,   # 512ch
    "cursor_wilson":          202,   # 192ch
    # Karpowicz 2024 / FALCON handwriting (DANDI:000950) — T5, 192ch
    "karpowicz_falcon":       203,   # 192ch
}

DATASET_CHANNELS = {
    "tx1": {
        "brandman_2024_text": 256,
        "willett_2023_text": 128,
    },
    "spikePow": {
        "brandman_2024_text": 256,
        "willett_2023_text": 128,
    },
    "all": {
        "brandman_2024_text": 512,
        "willett_2023_text": 256,
    },
    # ============================================================
    # Channel counts native to the local pickles in this repo.
    # Used for human-only pretraining +
    # phoneme decoding finetuning. The ReadIn / ReadOut layers in
    # ndt.py allocate one Linear per channel count, so every entry
    # below becomes a per-subject input/output projection.
    # ============================================================
    "human_all": {
        # === Human subject canonicals (aliases share same idx, not listed here) ===
        "willet_t12":           256,  # idx=2  (also: kunz_t12, willett-pile sessions)
        "card_t15":             512,  # idx=3  (also: kunz_t15)
        "kunz_t16":             128,  # idx=4
        "kunz_t17":             256,  # idx=5
        "willett_handwriting":  192,  # idx=6
        "fan_handwriting":      192,  # idx=7  (also: T5 neural-pile sessions)
        "wairagkar":            512,  # idx=8
        "jude_speech_anarthia": 256,  # idx=9
        "jude_typing_t17":      768,  # idx=10
        "jude_typing_t18":      768,  # idx=11
        # === Neural pile canonicals (aliases share same idx, not listed here) ===
        "neural_pile_M1_wojcik":                 10,   # idx=12
        "neural_pile_M2_wojcik":                 95,   # idx=13
        "neural_pile_Mahler_rajalingham":       337,   # idx=14
        "neural_pile_Perle_rajalingham":        512,   # idx=15
        "neural_pile_Router_lanzarini":         128,   # idx=16
        "neural_pile_T15_card":                 256,   # idx=17
        "neural_pile_Wifi_lanzarini":           128,   # idx=18
        "neural_pile_indy_makin":                96,   # idx=19
        "neural_pile_loco_makin":               192,   # idx=20
        "neural_pile_monkey_F_papale":          512,   # idx=21
        "neural_pile_monkey_N_papale":          512,   # idx=22
        "neural_pile_sub-An_xiao":               34,   # idx=23
        "neural_pile_sub-Bf_xiao":             137,    # idx=24
        "neural_pile_sub-Bo_xiao":              88,    # idx=25
        "neural_pile_sub-C_perich":            255,    # idx=26
        "neural_pile_sub-Fr_xiao":             167,    # idx=27
        "neural_pile_sub-Han_area2-bump":       65,    # idx=28
        "neural_pile_sub-Haydn_dmfc-rsg":       40,    # idx=29
        "neural_pile_sub-HumanPitt":           176,    # idx=30 (3 HumanPitt sessions)
        "neural_pile_sub-J_perich":             38,    # idx=31
        "neural_pile_sub-JenkinsC_even-chen":  192,    # idx=32
        "neural_pile_sub-Jenkins_churchland":  192,    # idx=33
        "neural_pile_sub-Lalo_chen":           101,    # idx=34
        "neural_pile_sub-Lo_xiao":             128,    # idx=35
        "neural_pile_sub-MG_moore":             81,    # idx=36
        "neural_pile_sub-M_perich":            114,    # idx=37
        "neural_pile_sub-Monkey-N_temmar":      96,    # idx=38
        "neural_pile_sub-MonkeyL":              64,    # idx=39 (3 MonkeyL sessions)
        "neural_pile_sub-MonkeyN":              96,    # idx=40 (2 MonkeyN sessions)
        "neural_pile_sub-MonkeyX":              64,    # idx=41 (3 MonkeyX sessions)
        "neural_pile_sub-Na_xiao":              69,    # idx=42
        "neural_pile_sub-Nitschke_churchland": 192,    # idx=43
        "neural_pile_sub-Oc_xiao":              67,    # idx=44
        "neural_pile_sub-Offenbach_chen":       77,    # idx=45
        "neural_pile_sub-Ot_xiao":              64,    # idx=46
        "neural_pile_sub-Pa_xiao":             180,    # idx=47
        "neural_pile_sub-Re_xiao":             104,    # idx=48
        "neural_pile_sub-Reggie_even-chen":    192,    # idx=49
        "neural_pile_sub-Sw_xiao":              72,    # idx=50
        "neural_pile_sub-T_perich":             69,    # idx=51
        "neural_pile_sub-Ve_xiao":             267,    # idx=52
        "neural_pile_sub-Ye_xiao":              33,    # idx=53
        "neural_pile_sub-Z08115_kim":           74,    # idx=54
        "neural_pile_sub-Z11111_kim":           22,    # idx=55
        "neural_pile_sub-amadeus_neupane-entorhinal": 64,   # idx=56
        "neural_pile_sub-amadeus_neupane-ppc":        68,   # idx=57
        "neural_pile_sub-mahler_neupane-entorhinal":  54,   # idx=58
        "neural_pile_sub-mahler_neupane-ppc":        512,   # idx=59
        "neural_pile_sub-monk-g_athalye":             42,   # idx=60
        "neural_pile_sub-monk-j_athalye":             20,   # idx=61
        "neural_pile_willett":                       256,   # idx=62 (3 willett sessions)
        "neural_pile_sub-T5":                        192,   # idx=63 (3 T5 sessions)
        # === New primate datasets — no neural pile equivalent ===
        "chowdhury_C_57":          57,   # idx=64
        "chowdhury_C_74":          74,   # idx=65
        "chowdhury_C_82":          82,   # idx=66
        "chowdhury_C_98":          98,   # idx=67
        "chowdhury_C_118":        118,   # idx=68
        "chowdhury_H_102":        102,   # idx=69
        "chowdhury_H_113":        113,   # idx=70
        "chowdhury_H_122":        122,   # idx=71
        "chowdhury_H_155":        155,   # idx=72
        "chowdhury_H_167":        167,   # idx=73
        "chowdhury_L_42":          42,   # idx=74
        "chowdhury_L_58":          58,   # idx=75
        "ma_Chewie_CO_2016":       92,   # idx=76
        "ma_Greyson_Key_2019":     96,   # idx=77
        "ma_Jango_ISO_2015":       85,   # idx=78
        "ma_Mihili_CO_2014":       94,   # idx=79
        "ma_Mihili_RT_2013_2014":  94,   # idx=80
        "ma_Spike_ISO_2012":       73,   # idx=81
        "odoherty_indy_192":      192,   # idx=82
        # Per-session Perich entries loaded dynamically below (idx 83–193)
        # NEJM cursor-control human subjects
        "cursor_karpowicz":       192,   # idx=200
        "cursor_singer_clark":    512,   # idx=201
        "cursor_wilson":          192,   # idx=202
        # Karpowicz 2024 / FALCON handwriting — T5, 192ch
        "karpowicz_falcon":       192,   # idx=203
    },
    # Phoneme finetuning subsets — only the two speech datasets with
    # phoneme labels. The encoder is loaded from the pretrained
    # ``human_all`` checkpoint; per-subject ReadIn weights still
    # match because the channel-dict ordering is preserved by
    # ReadIn (dataset_idx is read from DATASET_TO_IDX above).
    "willet_t12": {
        "willet_t12": 256,
    },
    "card_t15": {
        "card_t15": 512,
    },
    "kunz_t15": {
        "kunz_t15": 512,
    },
    "kunz_t12": {
        "kunz_t12": 256,
    },
    "kunz_t16": {
        "kunz_t16": 128,
    },
    "kunz_t17": {
        "kunz_t17": 256,
    },
    "wairagkar": {
        "wairagkar": 512,
    },
    "jude_speech_anarthia": {
        "jude_speech_anarthia": 256,
    },
    "willett_handwriting": {
        "willett_handwriting": 192,
    },
    "fan_handwriting": {
        "fan_handwriting": 192,
    },
    "jude_typing_t17": {
        "jude_typing_t17": 768,
    },
    "jude_typing_t18": {
        "jude_typing_t18": 768,
    },
    # Multi-source finetune buckets (used by finetune_joint.py).
    # "t15_card_and_kunz" keeps the per-source read-in/read-out modules from
    # the pretrained encoder so each source uses its own subject layers.
    "t15_card_and_kunz": {
        "card_t15": 512,
        "kunz_t15": 512,
    },
    # T12 twin of the above (willet_t12 + kunz_t12, both 256 ch). Note that
    # kunz_t12 aliases willet_t12 in DATASET_TO_IDX (same idx=2), mirroring how
    # t15_card_and_kunz treats card_t15/kunz_t15 (both idx=3).
    "t12_willett_and_kunz": {
        "willet_t12": 256,
        "kunz_t12": 256,
    },
    # All 6 speech subjects jointly (finetune_joint.py speech_joint sweep).
    "speech_all": {
        "card_t15":            512,
        "willet_t12":          256,
        "kunz_t15":            512,
        "kunz_t12":            256,
        "wairagkar":           512,
        "jude_speech_anarthia": 256,
    },
    # Both handwriting subjects jointly.
    "handwriting_joint": {
        "willett_handwriting": 192,
        "fan_handwriting":     192,
    },
    # Both typing subjects jointly.
    "typing_joint": {
        "jude_typing_t17": 768,
        "jude_typing_t18": 768,
    },
}

# ── Perich per-session registry (generated by prepare_perich.py) ──────────────
# Extends DATASET_TO_IDX and DATASET_CHANNELS["human_all"] at import time.
import json as _json
import os as _os

_PERICH_REGISTRY_PATH = "/path/to/perich/processed/perich_session_registry.json"
if _os.path.exists(_PERICH_REGISTRY_PATH):
    _perich_reg = _json.load(open(_PERICH_REGISTRY_PATH))
    for _key, _info in _perich_reg.items():
        DATASET_TO_IDX[_key] = _info["idx"]
        DATASET_CHANNELS["human_all"][_key] = _info["n_units"]


class SpikingDataset(Dataset):
    def __init__(
        self, 
        dataset:      Dict[str, List[Any]], 
        split:        str,
        length:       Optional[int] = None,
        spikes_name:  Optional[str] = "spikes",
        dataset_name: Optional[str] = "brandman_2024_text",
        **kwargs,
    ):  
        self.dataset = dataset[split]
        self.spikes_name = spikes_name
        self.dataset_name = DATASET_TO_IDX[dataset_name]
        self.dataset = self.dataset[:length] if length is not None else self.dataset
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        inputs = deepcopy(self.dataset[idx])

        spikes = inputs.pop(f"{self.spikes_name}")    

        inputs.update({
            "spikes": spikes,                                          
            "spikes_mask": np.ones(spikes.shape[0], dtype=np.int64),  
            "spikes_timestamp": np.arange(0, spikes.shape[0]),         
            "spikes_spacestamp": np.arange(0, spikes.shape[1]),       
            "spikes_lengths": np.asarray(spikes.shape[0]),      
            "dataset_name": np.array(self.dataset_name), 
            "day_idx": inputs["day_idx"],
            "block_idx": inputs["block_idx"],
        })
        return inputs
    

class SpikingDatasetForDecoding(SpikingDataset):
    def __init__(
        self, 
        dataset:      List[Dict[str, Union[np.ndarray, Any]]], 
        split:        str,
        length:       Optional[int] = None,
        spikes_name:  Optional[str] = "spikes",
        targets_name: Optional[str] = "targets",
        dataset_name: Optional[str] = "willett_2023_text",
        **kwargs,
    ):  
        super().__init__(dataset, split, length)
        
        self.targets_name = targets_name
        self.dataset_name = DATASET_TO_IDX[dataset_name]

    def __getitem__(self, idx):
        inputs = deepcopy(self.dataset[idx])
        
        spikes = inputs.pop(f"{self.spikes_name}")                          

        if self.targets_name == "diphones_idx":
            targets = inputs.pop("phonemes_idx") 
            targets_sub = inputs.pop("diphones_idx")
            inputs.update({
                "targets_sub":  targets_sub,
                "targets_sub_lengths": np.asarray(targets_sub.shape[0]),
            })
        else:
            targets = inputs.pop(f"{self.targets_name}") 

        inputs.update({
            "spikes": spikes,                                         
            "spikes_mask": np.ones(spikes.shape[0], dtype=np.int64),  
            "spikes_timestamp": np.arange(0,spikes.shape[0]),       
            "spikes_spacestamp": np.arange(0,spikes.shape[1]),    
            "spikes_lengths": np.asarray(spikes.shape[0]),         
            "targets":  targets,                                 
            "targets_mask": np.ones_like(targets),                
            "targets_lengths": np.asarray(targets.shape[0]),  
            "dataset_name": np.array(self.dataset_name),
            "day_idx": inputs["day_idx"],
            "block_idx": inputs["block_idx"],
        })
        return inputs


class MultiSpikingDataset(Dataset):
    """Multi-subject variant of :class:`SpikingDataset` for SSL pretraining.

    Expects each record in ``dataset[split]`` to carry its own
    ``dataset_idx`` (an int matching ``DATASET_TO_IDX[name]``). Used by the
    human-only / cross-species pretraining recipes that mix subjects into
    a single dataset object.
    """

    def __init__(
        self,
        dataset:      Dict[str, List[Any]],
        split:        str,
        length:       Optional[int] = None,
        spikes_name:  Optional[str] = "spikes",
        window_size:  Optional[int] = 300,
        **kwargs,
    ):
        self.dataset = dataset[split]
        self.dataset = self.dataset[:length] if length is not None else self.dataset
        self.spikes_name = spikes_name
        self.window_size = window_size  # None = no windowing for neural pile trials
        # Cache per-record dataset_idx for fast access by GroupBatchSampler.
        self.dataset_indices = np.asarray(
            [int(rec.get("dataset_idx", 0)) for rec in self.dataset], dtype=np.int64
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        inputs = deepcopy(self.dataset[idx])
        # Lazy loading: neural pile records store only the npz path + real channel count
        if "npz_path" in inputs and self.spikes_name not in inputs:
            npz = np.load(inputs.pop("npz_path"))
            n_ch = inputs.pop("npz_n_channels", None)
            src = inputs.pop("npz_source_dataset", None)
            feats = npz["neural_features"]  # (C_padded, T)
            if n_ch is not None:
                feats = feats[:n_ch]        # slice to real channels, discard zero-padding
            T = feats.shape[1]
            if self.window_size is not None and T > self.window_size:
                start = np.random.randint(0, T - self.window_size + 1)
                feats = feats[:, start : start + self.window_size]
            # Do NOT clip: data is already z-scored (var≈1). Heavy-tail gradient
            # signal drives fast convergence; NaN prevention via grad clipping.
            inputs["spikes"] = feats.T.astype(np.float32)  # (T, C)
        spikes = inputs.pop(f"{self.spikes_name}")
        inputs.update({
            "spikes":            spikes,
            "spikes_mask":       np.ones(spikes.shape[0], dtype=np.int64),
            "spikes_timestamp":  np.arange(0, spikes.shape[0]),
            "spikes_spacestamp": np.arange(0, spikes.shape[1]),
            "spikes_lengths":    np.asarray(spikes.shape[0]),
            "dataset_name":      np.array(int(inputs.pop("dataset_idx", 0))),
            "day_idx":           inputs.get("day_idx", np.asarray(0)),
            "block_idx":         inputs.get("block_idx", np.asarray(0)),
        })
        return inputs


class GroupBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, seed=None,
                 per_epoch_cap=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        # per_epoch_cap: keep at most this many TRAIN trials PER subject EACH
        # epoch. Unlike a fixed subset, the kept trials are re-drawn every epoch
        # (via the shuffle below), so over many epochs every trial is eventually
        # seen while per-epoch gradient steps stay balanced across subjects.
        # Only meaningful with shuffle=True (otherwise the subset never rotates).
        self.per_epoch_cap = per_epoch_cap
        self.rng = np.random.default_rng(seed)

        # Build mapping: dataset_name -> list of indices. Use a precomputed
        # ``dataset_indices`` array when the dataset exposes one (avoids
        # calling __getitem__ for each record, which deep-copies the entry).
        self.groups = {}
        if hasattr(dataset, "dataset_indices"):
            for idx, name in enumerate(dataset.dataset_indices.tolist()):
                self.groups.setdefault(int(name), []).append(idx)
        else:
            for idx in range(len(dataset)):
                name = dataset[idx]["dataset_name"]
                if hasattr(name, "item"):  # handle scalar tensors
                    name = name.item()
                self.groups.setdefault(name, []).append(idx)

    def __iter__(self):
        batches = []

        for indices in self.groups.values():
            indices = np.array(indices)
            if self.shuffle:
                self.rng.shuffle(indices)

            # After shuffling, keep only this epoch's slice of the subject. Next
            # epoch reshuffles → a different slice → all trials seen over time.
            if self.per_epoch_cap is not None and len(indices) > self.per_epoch_cap:
                indices = indices[:self.per_epoch_cap]

            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size].tolist()
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        if self.shuffle:
            self.rng.shuffle(batches)

        yield from batches

    def __len__(self):
        total_batches = 0
        for indices in self.groups.values():
            n = len(indices)
            if self.per_epoch_cap is not None:
                n = min(n, self.per_epoch_cap)
            total_batches += n // self.batch_size
            if not self.drop_last and n % self.batch_size:
                total_batches += 1
        return total_batches


def padded_array(
    arrays: List[np.ndarray],
    dim: Optional[Union[int, List[int]]] = 0,
    side: Optional[Union[str, List[str]]] = "right",
    value: Optional[Union[int, List[int]]] = 0,
    truncate: Optional[Union[int, List[int]]] = None,
    min_length: Optional[Union[int, List[int]]] = None,
    max_length: Optional[Union[int, List[int]]] = None,
) -> np.ndarray:
    """Pad multiple dimensions of numpy arrays.
    
    Args:
        arrays: List of arrays to pad
        dim: Dimension(s) to pad
        side: Side(s) to pad ('left' or 'right')
        value: Padding value(s)
        truncate: Maximum length(s) after padding
        min_length: Minimum length(s) after padding
    """
    # Convert single values to lists
    dims = [dim] if isinstance(dim, int) else dim
    sides = [side] * len(dims) if isinstance(side, str) else side
    values = [value] * len(dims) if isinstance(value, (int, float)) else value
    truncates = [truncate] * len(dims) if truncate is None or isinstance(truncate, int) else truncate
    min_lengths = [min_length] * len(dims) if min_length is None or isinstance(min_length, int) else min_length
    max_lengths = [max_length] * len(dims) if max_length is None or isinstance(max_length, int) else max_length

    # Start with the original arrays
    padded_arrays = arrays

    # Apply padding for each dimension
    for dim_idx, (d, s, v, t, m, l) in enumerate(zip(dims, sides, values, truncates, min_lengths, max_lengths)):
        # Get maximum size for this dimension
        if d == 0:
            max_size = max(arr.shape[d] for arr in padded_arrays)
        elif d == 1:
            max_size = l
        if t is None:
            t = max_size
        if m is None:
            m = 0
        assert m <= t, f"Can't truncate below minimum length for dimension {d}"
        pad_size = min(t, max(max_size, m))

        # Create padding widths
        pad_width = np.zeros((padded_arrays[0].ndim, 2), dtype=np.int64)
        if s == "left":
            pad_width[d, 0] = 1
        elif s == "right":
            pad_width[d, 1] = 1
        else:
            raise ValueError(f"'side' can only be 'right' or 'left', got {s}")

        # Create slice for truncation
        slc = [slice(None)] * padded_arrays[0].ndim
        slc[d] = slice(0, t)

        # Apply padding for this dimension
        padded_arrays = [
            np.pad(
                arr, 
                pad_width * max(0, pad_size - arr.shape[d]), 
                mode="constant", 
                constant_values=v
            )[tuple(slc)] 
            for arr in padded_arrays
        ]

    return np.stack(padded_arrays, axis=0)


def pad_collate_fn(
    batch: List[Dict[str, Union[np.ndarray, Any]]], 
    model_inputs: List[str],
    pad_dict: Dict[str, Union[Dict[str, Any], List[Dict[str, Any]]]],
) -> Tuple[Dict[str, Union[torch.Tensor, List[Union[torch.Tensor, Any]]]]]:
    """Collate function that handles multi-dimensional padding.
    
    Args:
        batch: List of examples
        model_inputs: Names of keys used by model
        pad_dict: Padding configuration for each key
    """

    def convert_dtype(tensor):
        """Convert only float64/double to float32, keep other dtypes."""
        if tensor.dtype in [torch.float64, torch.double]:
            return tensor.float()
        return tensor

    # Case when batching is done in dataset
    if isinstance(batch[0], list):
        batch = [row for sub_batch in batch for row in sub_batch]

    # Use key intersection so mixed batches (e.g. records with/without
    # phonemes_idx) don't raise KeyError when accessing row[key].
    keys = set.intersection(*[set(row.keys()) for row in batch])
    array_keys = [k for k in keys if isinstance(batch[0][k], np.ndarray)]
    string_array_keys = [k for k in keys if isinstance(batch[0][k], np.ndarray) and batch[0][k].dtype.type == np.str_]
    pad_keys = pad_dict.keys()
    
    assert set(pad_keys).issubset(array_keys), f"Can't pad non-array keys: {set(pad_keys)-set(array_keys)}"
    
    padded_batch = {}
    unused_inputs = {}
    
    for key in keys:
        if key in array_keys:
            if key in pad_keys:
                # Handle padding as before
                pad_config = pad_dict[key]
                if isinstance(pad_config, dict):
                    pad_config = [pad_config]
                value = torch.from_numpy(
                    padded_array(
                        [row[key] for row in batch],
                        dim=[p["dim"] for p in pad_config],
                        side=[p["side"] for p in pad_config],
                        value=[p["value"] for p in pad_config],
                        truncate=[p.get("truncate") for p in pad_config],
                        min_length=[p.get("min_length") for p in pad_config],
                        max_length=[p.get("max_length") for p in pad_config],
                    )
                ).clone()
                value = convert_dtype(value)
            elif key in string_array_keys:
                # Keep string arrays as numpy arrays
                value = np.stack([row[key] for row in batch], axis=0)
            elif len(set(row[key].shape for row in batch)) == 1:
                # Convert numeric arrays to tensors
                value = convert_dtype(torch.from_numpy(np.stack([row[key] for row in batch], axis=0)))
            else:
                value = [convert_dtype(torch.from_numpy(row[key])) for row in batch]
        else:
            value = [row[key] for row in batch if key in row]

        if key in model_inputs:
            padded_batch[key] = value
        else:
            unused_inputs[key] = value

    return padded_batch, unused_inputs


def save_load_arrays(
    cache_paths:  Dict[str, Path], 
    mode:         str, 
    subject:      Optional[str] = "",  
    date:         Optional[str] = "", 
    experiment:   Optional[str] = "",
    data:         Optional[np.ndarray] = None, 
):

    if mode == "save":
        for path in cache_paths.values():
            os.makedirs(path, exist_ok=True)
        
        arrays = {"spike_tx": data}
        for name, array in arrays.items():
            np.save(cache_paths[name]/f"{subject}_{date}_{experiment}.npy", array)
    else:
        # load mode
        # Sort file names to avoid introducing randomness from os.listdir
        # Use os.scandir() which is more efficient than os.listdir()
        print("Loading data...")
        with os.scandir(cache_paths["spike_tx"]) as it:
            file_names = sorted(
                f.name for f in it 
                if f.name.endswith('.npy') and f.is_file()
            )            
        return [np.load(cache_paths["spike_tx"]/f, allow_pickle=True, mmap_mode="r") for f in file_names]
    
    