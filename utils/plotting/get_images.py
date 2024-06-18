
import pandas as pd
import numpy as np
import os
import sys

# Import from sibling directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + ".")

# Import from parent directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "..")

# Import from project directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "../..")

from utils.helpers import get_areds_data_dir
from data.cnn_surv_dataloader import get_dataset

class GetPlotImages():
    """Get df with image paths for plotting. Drops already converted images."""
    def __init__(self, c, len=1, converters_only=False, seed=None, high_confidence=False, confidences_df: pd.DataFrame = None, from_file:str=None):
        self.c = c
        self.len = len
        self.converters_only = converters_only
        self.seed = seed if seed is not None else c.cnn.seed
        self.from_file = from_file

        if high_confidence:
            assert confidences_df is not None
            self.confidences_df = confidences_df

        if from_file is not None:
            self.selection_df = pd.read_csv(from_file, index_col=0)

    def get_image_df(self):
        data_testset = get_dataset("test", self.c)
        metadata = pd.read_csv(self.c.metadata_csv, index_col=0)

        # Keep only test set data
        image_files = [data_testset.get_img_path(idx=i).split("/")[-1] for i in range(len(data_testset))]
        df = metadata[metadata["image_file"].isin(image_files)].copy()

        # Verify that AMD score is 0-based, i.e. 0-8 are non-advanced AMDa and 9-11 advanced AMD scores
        assert max(df["diagnosis_amd_grade_12c"].value_counts().keys()) == 11

        if self.converters_only and self.from_file is None:
            # Select coverter images before conversion
            df = df[(df["diagnosis_amd_grade_12c"] < 9) & (df["event"] == 1)].copy()
            df.reset_index(inplace=True, drop=True)

        # Choose high confidence images
        if hasattr(self, "confidences_df") and self.from_file is None:
            logit_p = np.percentile(self.confidences_df["logit"], 90)
            self.confidences_df = self.confidences_df[self.confidences_df["logit"] > logit_p].sort_values(by="logit", ascending = False).reset_index(drop=True)
            df = df[df["image_file"].isin(self.confidences_df["image_file"])].sort_values(by="logit", ascending=False).reset_index(drop=True)

        if self.from_file is not None:
            df = df[df["image_file"].isin(self.selection_df["image_file"])]#.sort_values(by="logit", ascending=False).reset_index(drop=True)
        
        # Drop already converted images
        df = df[df["diagnosis_amd_grade_12c"] < 9]

        # Randomly select len images
        if self.len is not None and self.from_file is None:
            df = self._sample(df)
        
        # Get full image paths
        df["image_path"] = df["image_path"].apply(lambda x: os.path.join(get_areds_data_dir(), "images-f2-1024px", x))

        df.reset_index(inplace=True, drop=True)

        return df

    def _sample(self, df):
        return df.sample(n=self.len, random_state=self.seed)

