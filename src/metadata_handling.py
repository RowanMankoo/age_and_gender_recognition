import pathlib
from datetime import datetime, timedelta
from logging import getLogger
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import loadmat
from tqdm import tqdm

# TODO: get this to work
logger = getLogger(__name__)


class MetaDataExtractor:
    def __init__(self, imdb_mat_path="Data/imdb_crop/imdb.mat"):
        self.meta = loadmat(imdb_mat_path)

    def generate_metadata(self):
        """
        Get the metadata for all the images from the mat file and store it as a csv file.
        """

        matlab_dob = self.meta["imdb"][0][0][0].ravel().astype(np.float64)
        date_photo_taken = self.meta["imdb"][0][0][1].ravel().astype(np.float64)
        image_paths = self.meta["imdb"][0][0][2].ravel()
        gender = self.meta["imdb"][0][0][3].ravel()
        celeb_id = self.meta["imdb"][0][0][9].ravel().astype(np.float64)
        face_locations = self.meta["imdb"][0][0][5].ravel()

        # Convert matlab datenum to python datetime
        dob = self._convert_matlab_datenum_array(matlab_dob)
        age = [(date_photo_taken[i] - dob[i]) if dob[i] is not None else None for i in range(len(dob))]
        image_paths = [Path(x.tolist()[0]) for x in image_paths]

        face_score = self.meta["imdb"][0][0][6].ravel()
        second_face_score = self.meta["imdb"][0][0][7].ravel()

        df = pd.DataFrame(
            {
                "image_path": image_paths,
                "age": age,
                "gender": gender,
                "face_score": face_score,
                "face_locations": face_locations,
                "celeb_id": celeb_id,
                "second_face_score": second_face_score,
                "dob": dob,
                "date_photo_taken": pd.to_datetime(date_photo_taken, format="%Y"),
            }
        )

        csv_path = pathlib.Path("Data/Metadata/metadata.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)

    def _extract_year_from_matlab_datenum(self, matlab_datenum: float):
        """Extract year from matlab datenum"""
        try:

            return (
                datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum % 1) - timedelta(days=366)
            ).year
        except OverflowError:
            return None

    def _convert_matlab_datenum_array(self, matlab_datenum_array: np.array):
        """Converts a MATLAB datenum array to a Python datetime array."""
        dob = [self._extract_year_from_matlab_datenum(date) for date in matlab_datenum_array]
        return dob


class MetaDataCleaner:
    def __init__(
        self,
        min_face_score: float,
        max_num_age_appearances: int,
        max_num_total_appearances: int,
    ):
        self.df = pd.read_csv("Data/Metadata/metadata.csv")
        self.num_images_in_original_dataset = len(self.df)

        self.min_face_score = min_face_score
        self.max_num_age_appearances = max_num_age_appearances
        self.max_num_total_appearances = max_num_total_appearances

    def clean_metadata(self):
        self._basic_cleaning()
        self._clean_face_score()
        self._clean_number_of_celebs()

        logger.info(f"Number of images in original dataset: {self.num_images_in_original_dataset}")
        logger.info(f"Number of images in filtered dataset: {self.df.shape[0]}")

        self.df.to_csv("Data/Metadata/metadata_cleaned.csv", index=False)

    def _is_rgb(self, image_path: str) -> bool:

        img = Image.open("Data/imdb_crop" / Path(image_path))
        return img.mode == "RGB"

    def _basic_cleaning(self):
        """
        Performs basic cleaning operations on the DataFrame.

        - drops rows with missing values in the 'age', 'gender', and 'face_score' columns.
        - removes rows containing infinity or negative infinity values.
        - removes rbg images.
        - filters out rows where the 'age' value is not within the range of 1 to 100 (inclusive).
        - filters out rows where the 'celeb_id' value is associated with multiple 'dob' values.
        """

        self.df = self.df.dropna(subset=["age", "gender", "face_score"])
        self.df = self.df[~self.df.isin([np.inf, -np.inf]).any(axis=1)]

        self.df["age"] = self.df["age"].astype(int)
        # self.df["date_photo_taken"] = pd.to_datetime(
        #     self.df["date_photo_taken"], format="%Y"
        # )
        logger.info("Filtering out non-RGB images")
        self.df["is_rgb"] = self.df["image_path"].apply(self._is_rgb)
        self.df = self.df[self.df["is_rgb"]]

        self.df = self.df[self.df["age"].isin(range(1, 101))]

        df_dob_per_celeb_id_count = self.df.groupby("celeb_id")["dob"].nunique() > 1
        celeb_ids_with_multiple_dobs = df_dob_per_celeb_id_count[df_dob_per_celeb_id_count == True].index
        self.df = self.df[~self.df["celeb_id"].isin(celeb_ids_with_multiple_dobs)]

    def _clean_face_score(self):
        self.df = self.df[self.df["face_score"] > self.min_face_score]  # noqa: E712

    def _find_max_num_celeb_id_apperence_by_age(self, df, celeb_id):
        """
        Filters down subset of df to only include a max of max_num_total_apperences images of one celeb,
        whilst taking into account the max number of times a celeb_id can appear for one age

        Args:
            df (pd.DataFrame): dataframe containing celeb_id and age columns
            id_x (int): celeb_id to find max number of times it appears in the dataset by age
        """
        max_total_appearances = self.max_num_total_appearances
        max_age_appearances = self.max_num_age_appearances
        df = df.copy()

        df_celeb = df[df["celeb_id"] == celeb_id]
        age_counts = df_celeb["age"].value_counts().reset_index()

        max_age_appearances_lower_bound = max_total_appearances // len(age_counts)

        if max_age_appearances_lower_bound >= max_age_appearances:
            max_appearances = max_age_appearances
        else:
            total_appearances = age_counts["age"].apply(lambda x: min(x, max_age_appearances_lower_bound)).sum()
            max_appearances = max_age_appearances_lower_bound

            while total_appearances < max_total_appearances and max_appearances < max_age_appearances:
                max_appearances += 1
                total_appearances = age_counts["age"].apply(lambda x: min(x, max_appearances)).sum()

            max_appearances -= 1

        df_subset = df_celeb.groupby("age").head(max_appearances)
        assert df_subset.shape[0] <= max_total_appearances

        return df_subset

    def _clean_number_of_celebs(self):
        list_of_all_celeb_ids = self.df["celeb_id"].unique()
        df_filtered = []

        print("Filtering down dataset stop max number of celeb_id apperences")
        for celeb_id in tqdm(list_of_all_celeb_ids):
            df_filtered.append(self._find_max_num_celeb_id_apperence_by_age(self.df, celeb_id))

        df_filtered = pd.concat(df_filtered)
        self.df = df_filtered

