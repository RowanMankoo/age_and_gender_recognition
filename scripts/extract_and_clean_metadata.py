from logging import Logger

from src.core import load_and_validate_config
from src.metadata_handling import MetaDataCleaner, MetaDataExtractor

# TODO: test if this runs, also fix logging

if __name__ == "__main__":
    logger = Logger(__name__)

    config = load_and_validate_config(
        path_to_metadata_cleaning_config="config/metadata_cleaning_config.yaml",
        path_to_model_config="config/model_config.yaml",
        path_to_model_training_config="config/model_training_config.yaml",
    )

    metadata_extractor = MetaDataExtractor()
    metadata_extractor.generate_metadata()

    # filter down metadata to remove images that are not RGB, have low face score, and so on
    metadata_cleaner = MetaDataCleaner(
        min_face_score=config.metadata_cleaning_config.min_face_score,
        max_num_age_appearances=config.metadata_cleaning_config.max_num_age_appearances,
        max_num_total_appearances=config.metadata_cleaning_config.max_num_total_appearances,
    )
    metadata_cleaner.clean_metadata()

    print("Metadata Cleansed")
