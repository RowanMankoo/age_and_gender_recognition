import logging
from pathlib import Path

import pandas as pd

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    images_prefix = Path("Data/crop_part1")
    images = list(images_prefix.glob("*.jpg"))

    df_metadata = pd.DataFrame(columns=["filename", "age", "gender"])
    num_failed_image_names_to_parse = 0
    for i, image in enumerate(images):
        filename = image.name
        try:
            age, gender, _, _ = filename.split("_")
        except ValueError:
            num_failed_image_names_to_parse += 1
        df_metadata.loc[i] = [filename, age, gender]

    logger.warning(f"Failed to parse {num_failed_image_names_to_parse} image names")

    df_metadata["age"] = df_metadata["age"].astype(int)

    def cap_rows(df):
        return df.sample(min(len(df), 200))

    logger.info(f"Num rows before balencing: {len(df_metadata)}")
    df_metadata = df_metadata.groupby(["age", "gender"]).apply(cap_rows).reset_index(drop=True)
    logger.info(f"Num rows after balencing: {len(df_metadata)}")

    df_metadata.to_csv("Data/metadata.csv", index=False)

    logger.info("Metadata file sucessfully created and saved to Data/metadata.csv")
