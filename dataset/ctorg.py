import json
import os
from typing import Dict, List, Sequence
from torch.utils.data import Dataset, SubsetRandomSampler
from torchvision.transforms import Compose
from monai.transforms import (
    LoadImage,
    LoadImageD,
    EnsureTypeD,
    EnsureChannelFirstD,
    OrientationD,
    ResizeD,
    ScaleIntensityD,
)

class CTOrgTorchDataset(Dataset):
    def __init__(self, args, data_dir: str):
        """
        Initialize the CTOrgTorchDataset loader for the CT-ORG dataset.

        Args:
            data_dir (str): Path to the directory containing CT-ORG dataset.
            transform (callable, optional): A function/transform to apply to the samples.
        """
        self.data_dir = data_dir
        self.image_size = args.image_size
        self.transform = Compose([
            LoadImageD(keys=["volume", "label"]),
            EnsureTypeD(keys=["volume", "label"]),
            EnsureChannelFirstD(keys=["volume", "label"]),
            OrientationD(keys=["volume", "label"], axcodes="RAS"),
            ResizeD(keys=["volume", "label"], spatial_size=(self.image_size, self.image_size, -1)),
            ScaleIntensityD(keys=["volume"], minv=0.0, maxv=1.0),  # Dynamically scales per image
        ])
        self.data = self._create_dataset()
        self.slice_pairs = [
            (id, slice_idx)
            for id, item in self.data.items()
            for slice_idx in range(item["depth"])
        ]
        self.current_id = None
        self.current_data = None

    def _create_dataset(self) -> List[Dict]:
        """
        Create a dataset with volumes and labels.

        Returns:
            list: A dataset ready for loading and processing.
        """
        if os.path.exists("volume_data.json"):
            with open("volume_data.json", "r") as f:
                loaded_data = json.load(f)
                return loaded_data
        loader = LoadImage(image_only=True)
        # Create a list of tuples: (filename, depth)
        meta_volumes = sorted([
            (int(f.split('.')[0].split('-')[1]),
            os.path.join(self.data_dir, f), 
            loader(os.path.join(self.data_dir, f)).shape[-1])
            for f in os.listdir(self.data_dir)
            if f.startswith("volume") and f.endswith(".nii.gz")
        ],
        key=lambda x: x[0])

        ids = [item[0] for item in meta_volumes]
        volume_files = [item[1] for item in meta_volumes]
        volume_depths = [item[2] for item in meta_volumes]

        label_files = sorted([
            os.path.join(self.data_dir, f)
            for f in os.listdir(self.data_dir)
            if f.startswith("labels") and f.endswith(".nii.gz")
        ],
        key=lambda x: int(x.split('/')[-1].split('.')[0].split('-')[1]))

        # Check file pairing
        if len(volume_files) != len(label_files):
            raise ValueError("Mismatch between volume and label files!")

        dicts = [{"id": id, "depth": d, "volume": v, "label": l} for id, d, v, l in zip(ids, volume_depths, volume_files, label_files)]
        return dicts
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, slice_indices):
        idx, slice_idx = slice_indices
        if not (self.current_id and self.current_id==idx):
            self.current_id = idx
            data_item = self.data[idx]
            self.current_data = self.transform(data_item)
        volume = self.current_data["volume"]
        label = self.current_data["label"]
        volume_slice = volume[..., slice_idx]
        label_slice = label[..., slice_idx]

        return {
            'image_meta_dict': {'filename_or_obj': idx},
            "slice_idx": slice_idx,
            "image": volume_slice,
            "label": label_slice
        }

class SliceSampler(SubsetRandomSampler):
    def __init__(self, indices: Sequence[int], data, generator=None) -> None:
        self.indices = indices
        self.generator = generator
        self.slice_indices = [
            (idx, slice_idx)
            for idx in indices
            for slice_idx in range(data[idx]['depth'])
        ]
        
    def __iter__(self):
        return iter(self.slice_indices)

    def __len__(self):
        return len(self.slice_indices)
