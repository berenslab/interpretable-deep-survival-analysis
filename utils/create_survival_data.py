# Methods used in misc_save_survival_metadata.ipynb to create the metadata dataframe for 
# survival analysis from parsed AREDS metadata.

import os
import sys
from typing import List, Union, Dict

import numpy as np
import pandas as pd

# Import from parent directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + ".")

from .helpers import (
    get_areds_data_dir
)

def get_amd_grade_mapping(from_: int, to_: int) -> dict:
    """Returns a mapping from the 0-based AREDS AMD scale to the new binned scale and vice versa

    Args:
        from_ (int): Number of bins to map from
        to_ (int): Number of bins to map to

    Returns:
        dict: Mapping
    """

    # Mapping from 12-class to 2-class AMD scale
    mapping_12c_2c = {
        0: 0,  # No Adv. AMD
        1: 0,  # No Adv. AMD
        2: 0,  # No Adv. AMD
        3: 0,  # No Adv. AMD
        4: 0,  # No Adv. AMD
        5: 0,  # No Adv. AMD
        6: 0,  # No Adv. AMD
        7: 0,  # No Adv. AMD
        8: 0,  # No Adv. AMD
        9: 1,  # Adv. AMD
        10: 1,  # Adv. AMD
        11: 1,  # Adv. AMD
        np.nan: -1,  # Missing values
    }

    if from_ == 12 and to_ == 2:
        return mapping_12c_2c
    elif from_ == 2 and to_ == 12:
        return get_inverse_dict(mapping_12c_2c)
    elif from_ == to_:
        return {i: i for i in range(to_)}
    else:
        raise ValueError(
            f"Mapping from {from_}-class to {to_}-class AMD scale not supported yet."
        )


def get_inverse_dict(my_map: dict) -> dict:
    """Returns the inverse of a dictionary"""
    inv_map: Dict[int, list] = {}
    for k, v in my_map.items():
        inv_map[v] = inv_map.get(v, []) + [k]

    return inv_map


def bin_amd_scale(to: int, df: pd.DataFrame) -> pd.DataFrame:
    """Zero-base the AMD score and bin AMD scale to given number of bins: 2, or 12

    Args:
        to (int, optional): Number of bins to bin AMD scale to. Defaults to 2.
        df (pd.DataFrame): DataFrame to bin AMD scale in

    Returns:
        pd.DataFrame: DataFrame with binned AMD scale
    """

    if to != 12 and to != 2:
        print("Invalid number of bins. Must be 2, 6, or 12. Using 6.")
        to = 12

    df = df.copy()

    # Subtract 1 from AREDS score to make 0-based for classification
    df["diagnosis_amd_grade"] = df["diagnosis_amd_grade"] - 1
    df["diagnosis_amd_grade_other_eye"] = df["diagnosis_amd_grade_other_eye"] - 1

    if to == 12:
        df["diagnosis_amd_grade_binned"] = df["diagnosis_amd_grade"]
        df["diagnosis_amd_grade_other_eye_binned"] = df["diagnosis_amd_grade_other_eye"]

    elif to == 2:
        # Bin AMD scale to 2 bins
        def _bin_amd_scale_2(x):
            if str(x) != "nan":
                return get_amd_grade_mapping(12, 2)[x]
            else:
                return -1

        df["diagnosis_amd_grade_binned"] = df["diagnosis_amd_grade"].apply(
            _bin_amd_scale_2
        )

        df["diagnosis_amd_grade_other_eye_binned"] = df[
            "diagnosis_amd_grade_other_eye"
        ].apply(_bin_amd_scale_2)

    return df


def preprocess_metadata(
    df: pd.DataFrame,
    n_classes: int = 12,
) -> pd.DataFrame:
    """Preprocess DataFrame

        Bins AMD scale to n_classes
        and adds a column with image ids.

    Args:
        df (pd.DataFrame): DataFrame to preprocess
        impute_cols (list): List of columns to impute missing values in
        n_classes (int): Number of bins to bin AMD scale to

    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """
    metadata_df = df.copy()

    metadata_df.sort_values(
        by=["patient_id", "visit_number"], inplace=True, ignore_index=True
    )
    metadata_df["image_id"] = metadata_df.index

    # Zero-base the grades and collapse amd grade scale to n categories
    metadata_df = bin_amd_scale(to=n_classes, df=metadata_df)

    # Make sure that there's no single wrong label
    metadata_df = metadata_df[metadata_df["diagnosis_amd_grade_binned"] != -1]
    metadata_df = metadata_df[metadata_df["diagnosis_amd_grade_other_eye_binned"] != -1]

    # Use NA for missing values
    metadata_df.fillna(np.nan, inplace=True)

    # Drop columns with NA entries
    metadata_df.dropna(axis=1, how="all", inplace=True)

    return metadata_df

def print_patient_stats(df: Union[pd.DataFrame, np.ndarray]):
    """Print some statistics about the patients."""
    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df)
    # Number of rows
    n_rows = len(df)
    n_cols = len(df.columns)
    # Number of patients
    n_patients = df["patient_id"].nunique()
    # Number of patients that have multiple entries at any visit
    if "visit_number" in df.columns:
        n_patients_multiple_entries = (
            df.groupby(["patient_id", "visit_number"])
            .filter(lambda x: len(x) > 1)["patient_id"]
            .nunique()
        )
    else:
        n_patients_multiple_entries = None
    # Duplicate patient_ids
    n_duplicate_patient_ids = df["patient_id"].duplicated().sum()
    # Min and max visit number
    if "visit_number" in df.columns:
        min_visit_number = df["visit_number"].min()
        max_visit_number = df["visit_number"].max()
    else:
        min_visit_number = None
        max_visit_number = None

    print("### Patient statistics:")
    print(f"Number of rows                         : {n_rows}")
    print(f"Number of columns                      : {n_cols}")
    print(f"Number of patients                     : {n_patients}")
    print(f"N pat. w/ multiple entries at any visit: {n_patients_multiple_entries}")
    print(f"Number of duplicate patient_ids        : {n_duplicate_patient_ids}")
    print(f"Min visit number: {min_visit_number}, max visit number: {max_visit_number}")
    print("###\n")


def read_csv(
    path: str
) -> pd.DataFrame:
    """Read csv file of metadata and return pandas DataFrame with metadata and all
        AMD grades (for 2, 6, and 12 classes).

    Args:
        path (str): Path to csv file

    Returns:
        pd.DataFrame
    """
    df_parsed = pd.read_csv(path)

    # Get all the AMD grades as columns for 2, 6 and 12 class-problems
    df_2 = preprocess_metadata(df_parsed, n_classes=2)
    df_2 = df_2.rename(columns={"diagnosis_amd_grade_binned": "diagnosis_amd_grade_2c"})

    df_12 = preprocess_metadata(df_parsed, n_classes=12)

    df_12 = df_12.rename(
        columns={"diagnosis_amd_grade_binned": "diagnosis_amd_grade_12c"}
    )
    df_12.drop(
        columns=[
            "diagnosis_amd_grade_other_eye_binned",
            "diagnosis_amd_grade_other_eye",
            "diagnosis_amd_grade",
        ],
        inplace=True,
    )

    # Merge columns
    df = df_12.merge(
        df_2[["image_path", "diagnosis_amd_grade_2c"]], on="image_path", how="left"
    )

    return df


def recode_visnos(x: np.ndarray):
    """Recode the visit numbers for survival analysis.

        Recode visit_number for each patient to be 0-based. This makes visit_number relative to
        the first individual visit. If the minimal visit is odd, we have to recode it to be even,
        as odd visits are extra visits with low amounts we cannot use.

    Args:
        x (np.ndarray): Array of visit numbers for each patient.

    Returns:
        np.ndarray: Recoded visit numbers.
    """
    if x.min() % 2 == 0:
        x_new = x - x.min()
    else:
        x_new = x - x.min() - 1
    if x_new.min() > 0:
        print("Warning: x_new.min() > 0:", x_new)
    return x_new


def get_grade_col(df: pd.DataFrame, event_grade: int) -> tuple:
    """Get the column name of the diagnosis_amd_grade and the adjusted event_grade.

    Args:
        df (pd.DataFrame): DataFrame of metadata
        event_grade (int): The grade of AMD that is considered the event. It is adjusted if needed.

    Returns:
        tuple (str, int): The column name of the diagnosis_amd_grade and the adjusted event_grade.
    """

    # Add the "event", i.e. whether the patient reached advanced AMD (late) or not
    if "diagnosis_amd_grade_12c" in df.columns:
        grade_col = "diagnosis_amd_grade_12c"
    else:
        grade_col = "diagnosis_amd_grade"
        # The grades are 1-based in the original data and 0-based in the preprocessed "_12c" data
        event_grade += 1

    return grade_col, event_grade


def drop_records_after_event(df: pd.DataFrame, event_grade: int) -> pd.DataFrame:
    """Drop all records of a patient after the first event (event_grade) has occured in any eye.

    Args:
        df: DataFrame with columns "patient_id", "diagnosis_amd_grade" and "visit_number"

    Returns:
        DataFrame
    """
    grade_col, event_grade = get_grade_col(df, event_grade)

    # find the first visit number where the event occurs for each patient
    df_event_visnos = df.groupby("patient_id").apply(
        lambda x: x.loc[x[grade_col] >= event_grade, "visit_number"].min()
    )
    df_event_visnos = df_event_visnos.reset_index()
    df_event_visnos.columns = ["patient_id", "event_visit_number"]

    # Merge the event visit number with the original DataFrame
    df = df.merge(df_event_visnos, on="patient_id", how="left")

    # Drop all rows after the event visit number (if it is not NaN).
    df = df.loc[
        (df["visit_number"] <= df["event_visit_number"])
        | df["event_visit_number"].isna(),
        :,
    ]
    df.drop(columns=["event_visit_number"], inplace=True)
    return df


def add_event(df: pd.DataFrame, event_grade: int):
    """Add a column to the DataFrame that indicates if the event has occured.

    Args:
        df (pd.DataFrame): DataFrame of metadata
        event_grade (int): The grade of AMD that is considered the event.

    Returns:
        pd.DataFrame: DataFrame with new column "event"
    """
    # Add the "event", i.e. whether the patient reached advanced AMD (late) or not
    grade_col, event_grade = get_grade_col(df, event_grade)
    d = df.copy()
    # Prevent errors from some NaNs in the data
    d.astype({grade_col: "float64"})
    d[grade_col].replace(to_replace=pd.NA, value=None, inplace=True)

    # Group by patient eye and add event if any of their grades is >= event_grade
    d["event"] = d.groupby(["patient_id", "image_eye"])[grade_col].transform(
        lambda x: (x >= float(event_grade)).any()
    )
    return d


def get_survival_data(
    metadata_csv: str,
    event_grade: int = 9,
    remove_odd_visits: bool = False,
    visit_numbers: list = None,
    absolute_durations: bool = True,
    drop_after_event: bool = True,
    keep_stereo_pairs: bool = True,
    recode_visit_numbers: bool= True,
) -> pd.DataFrame:
    """Turn the data into a data frame suitable for survival analysis with time-constant models.
        E.g. pycox models and most auton models. It keeps all columns/features.

    Args:
        metadata_csv: Path to the metadata csv file
        event_grade (int): AMD grade (zero-based) to use for "the event",
            e.g. 9 for advanced AMD (late). Default is 9.
        remove_odd_visits (bool): If True, remove all visits that are uneven.
        visit_numbers (list of ints): If provided, only keep the visits in this list.
        absolute_durations (bool): If False, the duration is the visit number relative to the last visit.
        drop_after_event (bool): If True, drop all records of a patient after the first event has occured in any eye.
        keep_stereo_pairs (bool): If False, only keep one image of each stereo pair.
        recode_visit_numbers (bool): If True, recode the visit numbers to be 0-based.

    Returns:
        pd.DataFrame: Data frame with the survival data
    """

    def _add_events(df: pd.DataFrame) -> pd.DataFrame:
        """Helper function adding duration column and event column to the data frame.
        """

        data = df.copy()

        if absolute_durations:
            # Add "duration", i.e. max visit_number for each patient
            data["duration"] = data.groupby("patient_id")["visit_number"].transform("max")
        else:
            # Duration to event relative to record's visit_number
            data["duration"] = data.groupby("patient_id")["visit_number"].transform("max")
            data["duration"] = data["duration"] - data["visit_number"]
            data["duration"] = data["duration"].transform(lambda x: max(x, 0))

        data["duration"] = data["duration"].astype(int)

        # Add event
        data = add_event(data, event_grade=event_grade)

        return data

    def _rename_ids(df: pd.DataFrame) -> pd.DataFrame:
        """Turns each eye<->side combination into an own patient by appending a suffix, etc.
        df["patient_id"] is modified in-place such that
        "-<i>" is appended to the original patient_id for an artificial patient of which the
            i-th chunk ("bucket") of their data is used
        "_<eye>" is appended to the original patient_id for a patient of which the
            data of this eye is used
        "_<side>" is appended to the original patient_id for a patient of which the
            data of this side is used
        """
        data = df.copy()

        # Rename patients by their eye-side
        patient_ids = data["patient_id"]
        data["patient_id"] = (
            patient_ids + "_" + data["image_eye"] + "_" + data["image_side"]
        )

        return data

    def _recode(df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        # Re-code the visit numbers to be 0-based
        data = data.sort_values(by=["patient_id", "visit_number"])
        data["visit_number"] = data.groupby("patient_id")["visit_number"].transform(
            recode_visnos
        )
        data["visit_number"] = data["visit_number"].astype(int)

        # Drop records with visit_number -1
        data = data[data["visit_number"] != -1]
        return data

    # Read the data and preprocess AMD grades to be 0-based and impute if needed
    if not metadata_csv.startswith("/"):
        metadata_csv = os.path.join(get_areds_data_dir(), metadata_csv)
    df = read_csv(metadata_csv)

    # Drop stereo pairs, if wanted
    if not keep_stereo_pairs:
        # randomly choose RS or LS image of a stereo pair
        df = df.groupby(["patient_id", "image_eye", "visit_number"]).apply(
            lambda x: x.sample(1, random_state=123)
        )
        # Drop multiindex
        df = df.reset_index(drop=True)

    if recode_visit_numbers:
        # Re-code visit numbers
        df_new_visnos = _recode(df)
        print("Recoded visit numbers.")
    else:
        df_new_visnos = df.copy()
        df_new_visnos["visit_number"] = df_new_visnos["visit_number"].astype(int)

    print_patient_stats(df_new_visnos)

    if remove_odd_visits:
        # Drop odd visits as we cannot use them for survival analysis
        # (These are extra visits with low amounts of data)
        df_new_visnos = df_new_visnos[df_new_visnos["visit_number"] % 2 == 0]
        print("Removed odd visits.")
        print_patient_stats(df_new_visnos)

    if drop_after_event:
        df_new_visnos = drop_records_after_event(df_new_visnos, event_grade=event_grade)
    
    df_final = _add_events(df_new_visnos)

    # Rename patient_ids by their eye-side
    df_final = _rename_ids(df_final)

    if visit_numbers is not None:
        df_final = df_final[df_final["visit_number"].isin(visit_numbers)]

        # print(f"After dropping records not in visit_numbers: {len(df_final)} rows")
        # print_patient_stats(df_final)

    print("Before returning from get_survival_data:")
    print_patient_stats(df_final)

    return df_final









