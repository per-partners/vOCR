import json
import os
import re
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import logging
import time


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("temp/dataset_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Standardizing class names
CLASSES = {
    "medical provider name": ["Medical Provider", "medical provider"],
    "national id": ["national id", "National ID"],
    "employer name": ["employer name", "Employer Name"],
    "mobile number": ["mobile number", "Mobile Number"],
    "claim ID": ["Digital", "digital"],
    "stamp": ["STAMPS", "Stamps", "stamp"],
    "name": ["name", "Name", "NAMES"],
    "drug": ["Drug", "drugs"],
    "general handwritten": ["Hand Written", "general handwritten", "WRITTEN"],
    "number": ["Number", "number"],
    "unknown": ["UNK", "unknown"],
    "medical information": ["age", "address", "Approval No"],
    "diagnosis": ["DIAGNOSIS", "diagnosis"],
    "test": ["TESTS", "Tests", "test"],
    "scan": ["scan", "Scan"],
    "signature": ["doctor signature", "Signature"],
    "instructions": ["instructions", "Instruction"],
    "drug instructions": ["Drug Instructions"],
    "price": ["Price", "price"],
    "date": ["date", "Date"],
    "dental symbol": ["dental symbol plus", "dental symbol T"],
    "medical id": ["Medical ID", "medical ID"],
}


# Preprocess classes to create a normalized lookup
logger.info("Creating normalized class lookup dictionary")
normalized_classes = {}
for canonical, variants in CLASSES.items():
    for variant in variants:
        normalized_classes[variant.lower()] = canonical
logger.info(f"Created lookup for {len(normalized_classes)} class variants")


def normalize_class_title(class_title: str) -> str:
    """Normalize the class title to its canonical form."""
    normalized = normalized_classes.get(class_title.lower(), class_title)
    if normalized != class_title:
        logger.debug(f"Normalized class '{class_title}' to '{normalized}'")
    return normalized


def check_for_valid_claim_id(claim_id: str) -> bool:
    """Check if the claim ID is valid."""
    # create regex pattern for claim ID mix between digits and letters
    is_valid = bool(re.fullmatch(r"^\d+$", claim_id)
                    or re.fullmatch(r"^NC\d+$", claim_id))
    if not is_valid:
        logger.debug(f"Invalid claim ID: {claim_id}")
    return is_valid


def duplicate_and_split_dataset(parent_dir, num_duplicates=50, train_size=0.7, seed=42):
    start_time = time.time()
    logger.info(
        f"Starting dataset processing with parameters: train_size={train_size}, seed={seed}")

    images_dir = os.path.join(parent_dir, "images")
    labels_dir = os.path.join(parent_dir, "annotation")
    dataset_root = os.path.join(parent_dir, "processed_dataset")

    logger.info(
        f"Input directories: Images: {images_dir}, Labels: {labels_dir}")
    logger.info(f"Output directory: {dataset_root}")

    # Validate input directories
    if not os.path.exists(images_dir):
        logger.error(f"Images directory does not exist: {images_dir}")
        return
    if not os.path.exists(labels_dir):
        logger.error(f"Labels directory does not exist: {labels_dir}")
        return

    splits = ["train", "valid"]  # "test"]

    # Create directories for each split
    for split in splits:
        split_dir = os.path.join(dataset_root, split)
        os.makedirs(split_dir, exist_ok=True)
        logger.info(f"Created directory: {split_dir}")

    # Get the image names list
    image_names = os.listdir(images_dir)
    logger.info(f"Found {len(image_names)} images to process")

    # Check for corresponding annotation files
    image_with_annotations = [img for img in image_names if os.path.exists(
        os.path.join(labels_dir, f"{img}.json"))]
    logger.info(
        f"Found {len(image_with_annotations)} images with matching annotation files")

    if len(image_with_annotations) < len(image_names):
        logger.warning(
            f"Missing annotations for {len(image_names) - len(image_with_annotations)} images")

    # Use only images with annotations
    image_names = image_with_annotations

    count = 1
    # Split and collect all samples
    train_images, valid_images = train_test_split(
        image_names, test_size=1-train_size, random_state=seed
    )
    logger.info(
        f"Split dataset: {len(train_images)} training images, {len(valid_images)} validation images")

    split_data = [
        (train_images, "train"),
        (valid_images, "valid")
    ]

    # Process each split
    for split, split_name in tqdm(
        split_data,
        desc="Splitting dataset",
        colour="red",
    ):
        logger.info(f"Processing {split_name} split with {len(split)} images")
        all_samples = []
        skipped_images = 0
        processed_images = 0
        total_objects = 0
        skipped_objects = 0

        for image_name in tqdm(split, desc=f"Processing {split_name} samples", colour="green"):
            label_path = os.path.join(labels_dir, f"{image_name}.json")
            image_path = os.path.join(images_dir, image_name)

            logger.debug(f"Processing image: {image_name}")

            # Open json file
            try:
                with open(label_path, "r") as f:
                    label = json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON for {image_name}")
                skipped_images += 1
                continue
            except Exception as e:
                logger.error(
                    f"Error reading annotation file {label_path}: {str(e)}")
                skipped_images += 1
                continue

            # Read image
            try:
                image = cv2.imread(image_path)
                if image is None:
                    logger.error(f"Failed to read image: {image_path}")
                    skipped_images += 1
                    continue

                logger.debug(
                    f"Successfully read image {image_name} with shape {image.shape}")
            except Exception as e:
                logger.error(
                    f"Error reading image file {image_path}: {str(e)}")
                skipped_images += 1
                continue

            # Claim level
            sample = {
                "image_name": image_name,
                "image": image,
                "prefix": "Describe the handwritten text in the image in JSON Format",
                "suffix": {},
            }

            do_we_need_to_skip_the_whole_image = False
            valid_objects = 0
            objects_in_image = len(label.get("objects", []))
            total_objects += objects_in_image

            logger.debug(f"Image {image_name} has {objects_in_image} objects")

            for obj in label.get("objects", []):
                original_class_name = obj.get("classTitle", "")
                class_name = normalize_class_title(original_class_name)
                description = obj.get("description", "")

                logger.debug(
                    f"Processing object: class={class_name}, description={description}")

                if class_name == "claim ID":
                    if not check_for_valid_claim_id(description):
                        logger.debug(
                            f"Skipping invalid claim ID: {description}")
                        skipped_objects += 1
                        continue

                if class_name == "unknown":
                    logger.debug(
                        f"Skipping whole image due to 'unknown' class")
                    do_we_need_to_skip_the_whole_image = True
                    break

                valid_objects += 1

                if class_name in sample["suffix"]:
                    sample["suffix"][class_name].append(description)
                else:
                    sample["suffix"][class_name] = [description]

            # Save the original sample with its image
            if not do_we_need_to_skip_the_whole_image:
                all_samples.append(sample)
                processed_images += 1
                logger.debug(
                    f"Added full image {image_name} with {valid_objects} valid objects")
            else:
                skipped_images += 1

            # Crop Level - Process each object separately
            crop_count = 0
            for obj in label.get("objects", []):
                class_name = normalize_class_title(obj.get("classTitle", ""))
                description = obj.get("description", "")

                if class_name == "unknown":
                    logger.debug(f"Skipping 'unknown' class crop")
                    continue

                if class_name == "claim ID" and not check_for_valid_claim_id(description):
                    logger.debug(
                        f"Skipping invalid claim ID crop: {description}")
                    continue

                geo_type = obj.get("geometryType")

                try:
                    bbox = []
                    if geo_type == "rectangle":
                        bbox = obj["points"]["exterior"]
                    elif geo_type == "polygon":
                        # Convert polygon to rectangle
                        points = obj["points"]["exterior"]
                        x = [point[0] for point in points]
                        y = [point[1] for point in points]
                        x_min, x_max = min(x), max(x)
                        y_min, y_max = min(y), max(y)
                        bbox = [x_min, y_min, x_max, y_max]
                    else:
                        logger.warning(f"Unhandled geometry type: {geo_type}")
                        continue

                    bbox = np.array(bbox).flatten().tolist()

                    # Validate bbox coordinates
                    if (bbox[0] >= bbox[2] or bbox[1] >= bbox[3] or
                        bbox[0] < 0 or bbox[1] < 0 or
                            bbox[2] > image.shape[1] or bbox[3] > image.shape[0]):
                        logger.warning(
                            f"Invalid bbox for {image_name}: {bbox}")
                        continue

                    crop = image[int(bbox[1]):int(bbox[3]),
                                 int(bbox[0]):int(bbox[2])]

                    if crop.size == 0:
                        logger.warning(
                            f"Empty crop for {image_name} with bbox {bbox}")
                        continue

                    crop_name = f"{image_name}_{class_name}_{description}.jpg"

                    sample_crop = {
                        "image_name": crop_name,
                        "image": crop,
                        "prefix": "Describe the handwritten text in the image in JSON Format.",
                        "suffix": {},
                    }
                    sample_crop["suffix"][class_name] = [description]
                    all_samples.append(sample_crop)
                    crop_count += 1
                    logger.debug(
                        f"Created crop for {class_name} with description '{description}'")

                except Exception as e:
                    logger.error(
                        f"Error creating crop from {image_name}: {str(e)}")

            logger.debug(f"Created {crop_count} crops from image {image_name}")

        logger.info(f"{split_name} split statistics:")
        logger.info(f"  - Processed images: {processed_images}")
        logger.info(f"  - Skipped images: {skipped_images}")
        logger.info(f"  - Total objects: {total_objects}")
        logger.info(f"  - Skipped objects: {skipped_objects}")
        logger.info(
            f"  - Total samples (full images + crops): {len(all_samples)}")

        # Write annotations for this split
        annotations_path = os.path.join(
            dataset_root, split_name, "annotations.jsonl")
        logger.info(f"Writing annotations to {annotations_path}")

        samples_written = 0
        with open(annotations_path, "w") as f_out:
            for sample in tqdm(
                all_samples, desc=f"Creating {split_name} dataset", colour="blue"
            ):
                try:
                    # Saves images as '000001.jpg', '000002.jpg', etc.
                    new_image_name = f"{count:06d}.jpg"
                    dst_image = os.path.join(
                        dataset_root, split_name, new_image_name)

                    # Write image file
                    result = cv2.imwrite(dst_image, sample["image"])
                    if not result:
                        logger.error(f"Failed to write image to {dst_image}")
                        continue

                    # Update sample and write to JSONL
                    sample["image_name"] = new_image_name
                    # Remove image data before JSON serialization
                    sample.pop("image")
                    f_out.write(json.dumps(sample) + "\n")
                    samples_written += 1
                    count += 1
                except Exception as e:
                    logger.error(f"Error writing sample {count}: {str(e)}")

        logger.info(
            f"Successfully wrote {samples_written} samples to {split_name} split")

    total_time = time.time() - start_time
    logger.info(f"Processing completed in {total_time:.2f} seconds")
    logger.info(f"Dataset successfully created in {dataset_root}")
    logger.info(f"Total images processed: {count-1}")


if __name__ == "__main__":
    try:
        logger.info("Starting dataset processing script")
        parent_dir = "./assets/batch_claims"
        duplicate_and_split_dataset(parent_dir)
        logger.info("Dataset processing completed successfully")
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
