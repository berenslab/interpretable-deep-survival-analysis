import os

import numpy as np
import pandas as pd
import torch
from multi_level_split.util import train_test_split
from PIL import Image
from torchvision.datasets import VisionDataset

# Needed to prevent "OSError: image file is truncated"
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

SPLIT_SEED = 12345


class AredsSurvivalDataset(VisionDataset):
    """Pytorch Dataset for Areds fundus images"""

    def __init__(
        self,
        img_dir,
        metadata_csv,
        root="",
        transform=None,
        target_transform=None,
        split="train",
        exclude_imgs_for_classification_at=None,
        use_stereo_pairs=False,
    ):
        super(AredsSurvivalDataset, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        self.split = split

        self.use_stereo_pairs = use_stereo_pairs

        self.img_dir = img_dir
        metadata_df = pd.read_csv(metadata_csv)

        # Encode values for categorical non-numerical attributes
        fields = {"F1M": 1, "F2": 2, "F3M": 3}
        field_sides = {"RS": 0, "LS": 1, np.nan: 2}
        eyes = {"right": 0, "left": 1}

        # Split dataframe into train and test
        self._split_dict = {"train": 0, "test": 1, "val": 2}
        self._split_names = {
            "train": "Train",
            "test": "Test",
            "val": "Validation",
        }

        if split not in self._split_dict:
            raise ValueError(f"split not recognised: {split}")

        dev, test = train_test_split(
            metadata_df,
            "image_id",
            split_by="patient_id",
            # stratify_by="strat_group",
            test_split=0.2,
            seed=SPLIT_SEED,
        )

        train, val = train_test_split(
            dev,
            "image_id",
            split_by="patient_id",
            # stratify_by="strat_group",
            test_split=0.25,
            seed=SPLIT_SEED,
        )

        data = {"train": train, "val": val, "test": test}

        self._metadata_df = data[split]

        # If in classification setup:
        if exclude_imgs_for_classification_at is not None:
            print(f"Excluding images with no event and duration < {exclude_imgs_for_classification_at} due to classification setup")
            self.exclude_imgs_for_classification(exclude_imgs_for_classification_at)

        if self.use_stereo_pairs:
            print("Keeping only images that have a stereo pair")
            self.filter_stereo_pairs()

        p = "stereo pairs of images" if self.use_stereo_pairs else "images"
        print(f"Len data (Number of {p} in {self.split} split): {len(self._metadata_df)}")

        # Get the clf targets
        self._amd_grades = torch.LongTensor(
            self._metadata_df["diagnosis_amd_grade_12c"].values
        )

        # Get event indicators
        self._e = torch.LongTensor(self._metadata_df["event"].values)

        # Get time to event or censoring
        self._t = torch.LongTensor(self._metadata_df["duration"].values)

        # Get the image index in the dataset
        self._idx_array = torch.LongTensor(self._metadata_df.index.values)

        self._input_array = [
            os.path.join(self.img_dir, ele)
            for ele in self._metadata_df["image_path"].values
        ]

        # Get all other attributes
        self._eye_array = torch.LongTensor(
            [eyes[ele] for ele in self._metadata_df["image_eye"].values]
        )
        self._field_array = torch.LongTensor(
            [fields[ele] for ele in self._metadata_df["image_field"].values]
        )
        self._side_array = torch.LongTensor(
            [field_sides[ele] for ele in self._metadata_df["image_side"].values]
        )

        self._metadata_array = torch.stack(
            (
                self._eye_array,
                self._field_array,
                self._side_array,
                self._idx_array,
            ),
            dim=1,
        )
        self._metadata_fields = [
            "eye",
            "field",
            "side",
            "metadata_idx",
        ]

    def __len__(self):
        return len(self.amd_grades)

    def __getitem__(self, idx):
        """Returns an image (or two images if using stereo pairs), the amd grade, the event 
        indicator, the time to event or censoring, the metadata, and the image path."""
        
        grade = self.amd_grades[idx]

        e = self._e[idx]
        t = self._t[idx]

        metadata = self.metadata_array[idx]
        path = self.get_img_path(idx)

        x = self.get_input(idx)

        if self.use_stereo_pairs:
            x2 = self.get_input(idx, other=True)
            x = torch.stack([x, x2])

        return x, grade, e.float(), t, metadata, path

    def get_e_t(self, idx: int = None):
        if idx is None:
            return self._e, self._t
        return self._e[idx], self._t[idx]

    def get_input(self, idx, other=False):
        """
        Args:
            - idx (int): Index of a data point
            - other (bool): If True, return the other image of a stereo pair
        Output:
            - x (Tensor): Input features of the idx-th data point
        """
        if other:
            input_array = self._other_input_array
        else:
            input_array = self._input_array

        img_filename = os.path.join(self.img_dir, input_array[idx])
        x = Image.open(img_filename).convert("RGB")

        if self.transform is not None:
            x = self.transform(x)

        return x
    
    def exclude_imgs_for_classification(self, at_x: int):
        """Exclude certain images for classification model during training.
            If using a classification approach like Yan 2020, i.e. when training one model for each
            inquired year at_x, we need to exclude images that have no event and a relative 
            (censoring) duration < at_x 

        Args:
            - at_x (int): time for which the classification model is trained. 
        """
        
        self._metadata_df = self._metadata_df.loc[
            (self._metadata_df["event"] == 1) | (self._metadata_df["duration"] >= at_x)
            ]

    def filter_stereo_pairs(self):
        """Keep only images that have a stereo pair. For late-fusion models á la Babenko 2019.
        Creates a second metadata df for the other image of the stereo pair."""

        df = self._metadata_df.copy()
        df["pair_id"] = df["patient_id"].str.split("_").str[0] + df["image_eye"] + df["visit_number"].astype(str)
        df = df.groupby("pair_id").filter(lambda x: len(x) == 2 and x["image_side"].nunique() == 2)
        df_rs = df.loc[df["image_side"] == "RS"].copy().reset_index(drop=True)
        df_rs["order"] = df_rs.index
        df_ls = df.loc[df["image_side"] == "LS"].copy()

        df_rs = df_rs.sort_values(by=["pair_id"]).reset_index(drop=True)
        df_ls = df_ls.sort_values(by=["pair_id"]).reset_index(drop=True)
        df_ls["order"] = df_rs["order"].values

        df_rs = df_rs.sort_values(by=["order"]).reset_index(drop=True)
        df_ls = df_ls.sort_values(by=["order"]).reset_index(drop=True)

        assert df_rs["pair_id"].equals(df_ls["pair_id"]), "Wrong sorting of stereo pairs."

        assert len(df_rs) > 0, "No matching pairs found: Please input a metadata csv that includes stereo pairs."

        self._metadata_df = df_rs
        self._other_metadata_df = df_ls

        self._other_input_array = [
            os.path.join(self.img_dir, ele)
            for ele in self._other_metadata_df["image_path"].values
        ]



    def get_img_path(self, idx):
        return self._input_array[idx]

    @property
    def amd_grades(self):
        """
        A Tensor of targets (e.g., labels for classification tasks),
        with amd_grades[i] representing the target of the i-th data point.
        amd_grades[i] can contain multiple elements.
        """
        return self._amd_grades

    @property
    def metadata_fields(self):
        """
        A list of strings naming each column of the metadata table, e.g., ['hospital', 'y'].
        Must include 'y'.
        """
        return self._metadata_fields

    @property
    def metadata_array(self):
        """
        A Tensor of metadata, with the i-th row representing the metadata associated with
        the i-th data point. The columns correspond to the metadata_fields defined above.
        """
        return self._metadata_array
    

