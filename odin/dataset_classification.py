import base64
import json
import logging
import os
import random
import shutil

import click
import albumentations as A
from colorama import Fore
import cv2

from constants import README_CUSTOM_DATASETS
from dataset import BaseDatasetCommands


class DatasetCommandsClassification(BaseDatasetCommands):
    def __init__(self, dataset_name):
        self.type = "classification"
        self.dataset_name = dataset_name
        self.dataset_path = f"{os.path.abspath('.')}\\datasets\\{dataset_name}"

    def _create_dataset_metadata_files(self, **kwargs):
        os.makedirs(self.dataset_path)

        dataset_info = {
            "type": "classification",
            "version": "0.1.0",
            "snapshots": {},
        }

        with open(f"{self.dataset_path}\\dataset.json", "w", encoding="utf8") as wf:
            wf.write(json.dumps(dataset_info))

        with open(f"{self.dataset_path}\\snapshot.json", "w", encoding="utf8") as wf:
            wf.write(json.dumps({}))

        with open(
            f"{self.dataset_path}\\CUSTOM_DATASETS.md", "w", encoding="utf8"
        ) as wf:
            wf.write(README_CUSTOM_DATASETS)

        os.makedirs(self.dataset_path + "\\train")
        os.makedirs(self.dataset_path + "\\val")

        logging.info("Succesfully created final folders.")

        # Creating dataset staging folder

        logging.info("Creating staging folders...")

        os.makedirs(self.dataset_path + "\\staging")
        os.makedirs(self.dataset_path + "\\staging\\class_1")
        os.makedirs(self.dataset_path + "\\staging\\class_2")

        logging.info("Succesfully created staging folders.")

        # Praise the gods! Your dataset folders are created, you can now insert your images and labels on YOLO format at 'self.dataset_path\\staging' and then run 'odin dataset stage {dataset_name} --train=70 --val=30'
        logging.info(
            f"Praise the gods! Your dataset folders are created, you can now insert your {Fore.CYAN}images{Fore.RESET} at {Fore.CYAN}{self.dataset_path}\\staging{Fore.RESET} and then run {Fore.CYAN}odin dataset stage {self.dataset_name} --train=70 --val=30{Fore.RESET} (tip: you can change the {Fore.CYAN}--train{Fore.RESET} and {Fore.CYAN}--val{Fore.RESET} values to increase or decrease the split of the dataset)."
        )
        logging.info(
            f"{Fore.GREEN}NOTE{Fore.RESET}: {Fore.CYAN}Odin{Fore.RESET} created two temporary classes as placeholders to your classification project called {Fore.CYAN}class_1{Fore.RESET} and {Fore.CYAN}class_2{Fore.RESET}. You can delete them and include your own classes. If you don't know how to create a classification dataset follow the Ultralytics guide available at: https://docs.ultralytics.com/datasets/classify/#dataset-structure-for-yolo-classification-tasks"
        )

    def _add_image_to_version_snapshot(
        self, snapshot={}, dataset_split="", dataset_class="", image_name="", image_binary="", **kwargs
    ):
        try:
            snapshot[dataset_split][dataset_class].append(
                {
                    "filename": image_name,
                    "binary": base64.b64encode(image_binary).decode("utf8"),
                }
            )
        except:
            snapshot[dataset_split][dataset_class] = [
                {
                    "filename": image_name,
                    "binary": base64.b64encode(image_binary).decode("utf8"),
                }
            ]

    def _execute_data_publishment(self, snapshot={}, split="", dataset_class="", split_max_file=0, **kwargs):
        images = []

        for _, _, files in os.walk(f"{self.dataset_path}\\staging\\{dataset_class}"):
            images = files
            break

        for i in range(0, split_max_file + 1):
            try:
                image_stage_path = (
                    f"{self.dataset_path}\\staging\\{dataset_class}\\{images[i]}"
                )
                
                self._try_create_folder(f"{self.dataset_path}\\{split}\\{dataset_class}")
                
                image_publish_path = (
                    f"{self.dataset_path}\\{split}\\{dataset_class}\\{images[i]}"
                )

                shutil.move(
                    image_stage_path,
                    image_publish_path,
                )

                image_binary = open(image_publish_path, "rb").read()

                self._add_image_to_version_snapshot(
                    snapshot, split, dataset_class, images[i], image_binary
                )
            except IndexError:
                pass

    def _publish_data(self, update_type="", train=0, val=0, *args):
        classes = []

        base_version = json.loads(
            open(f"{self.dataset_path}\\dataset.json", "r", encoding="utf8").read()
        )["version"]
        temp_version = self._upgrade_version(base_version, update_type)

        # SNAPSHOT UPDATE
        snapshot = {
            "staging": {},
            "train": {},
            "val": {},
        }

        for x in os.walk(f"{self.dataset_path}\\staging"):
            if len(x[1]) > 0:
                classes = x[1]

        for dataset_class in classes:
            if (
                sum(
                    len(files)
                    for _, _, files in os.walk(
                        f"{self.dataset_path}\\staging\\{dataset_class}"
                    )
                )
                == 0
            ):
                logging.info(
                    f"The {Fore.CYAN}Staging{Fore.RESET} dataset is empty, so nothing will be published or updated."
                )
                return

            logging.info(f"Publishing {Fore.CYAN}{dataset_class}{Fore.RESET}")

            if (
                sum(
                    len(files)
                    for _, _, files in os.walk(
                        f"{self.dataset_path}\\staging\\{dataset_class}"
                    )
                )
                == 0
            ):
                logging.info(
                    f"The {Fore.CYAN}Staging{Fore.RESET} dataset for class {Fore.CYAN}{dataset_class}{Fore.RESET} is empty, so nothing will be published or updated for this class. Skipping this one."
                )
                return

            logging.info("Publishing with the following splits:")
            logging.info(f"{Fore.CYAN}train{Fore.RESET}: {train}%")
            logging.info(f"{Fore.CYAN}val{Fore.RESET}: {val}%")

            count_train = int(
                (train / 100)
                * sum(
                    len(files)
                    for _, _, files in os.walk(
                        f"{self.dataset_path}\\staging\\{dataset_class}"
                    )
                )
            )

            count_val = int(
                (val / 100)
                * sum(
                    len(files)
                    for _, _, files in os.walk(
                        f"{self.dataset_path}\\staging\\{dataset_class}"
                    )
                )
            )

            self._execute_data_publishment(
                snapshot, "train", dataset_class, count_train
            )
            logging.info(
                f"Succesfully published {Fore.GREEN}train{Fore.RESET} data for class {Fore.CYAN}{dataset_class}{Fore.RESET}"
            )
            self._execute_data_publishment(snapshot, "val", dataset_class, count_val)
            logging.info(
                f"Succesfully published {Fore.GREEN}val{Fore.RESET} data for class {Fore.CYAN}{dataset_class}{Fore.RESET}"
            )

        dataset_info = json.loads(
            open(f"{self.dataset_path}\\dataset.json", "r", encoding="utf8").read()
        )
        snapshot_info = json.loads(
            open(f"{self.dataset_path}\\snapshot.json", "r", encoding="utf8").read()
        )

        snapshot_info[temp_version] = snapshot
        dataset_info["version"] = temp_version

        with open(f"{self.dataset_path}\\dataset.json", "w", encoding="utf8") as wf:
            wf.write(json.dumps(dataset_info, indent=4))
        logging.info(
            f"Succesfully updated dataset version to {Fore.CYAN}v{temp_version}{Fore.RESET}"
        )

        with open(f"{self.dataset_path}\\snapshot.json", "w", encoding="utf8") as wf:
            wf.write(json.dumps(snapshot_info, indent=4))
        logging.info(
            f"Succesfully registered snapshots for dataset {Fore.CYAN}v{temp_version}{Fore.RESET}"
        )

    def _status_sum_staging(self, **kwargs):
        classes = []
        for x in os.walk(f"{self.dataset_path}\\staging"):
            if len(x[1]) > 0:
                classes = x[1]

        final_sum = 0
        for dataset_class in classes:
            final_sum += sum(
                len(files)
                for _, _, files in os.walk(
                    f"{self.dataset_path}\\staging\\{dataset_class}"
                )
            )

        return final_sum

    def _status_sum_train(self, **kwargs):
        classes = []
        for x in os.walk(f"{self.dataset_path}\\staging"):
            if len(x[1]) > 0:
                classes = x[1]

        final_sum = 0
        for dataset_class in classes:
            final_sum += sum(
                len(files)
                for _, _, files in os.walk(
                    f"{self.dataset_path}\\train\\{dataset_class}"
                )
            )

        return final_sum

    def _augmentate_data(self, augmentation_amount=0, **kwargs):
        try:
            classes = []
            for x in os.walk(f"{self.dataset_path}\\staging"):
                if len(x[1]) > 0:
                    classes = x[1]

            for dataset_class in classes:
                images = []
                for _, _, files in os.walk(
                    f"{self.dataset_path}\\train\\{dataset_class}"
                ):
                    images = files
                    break

                if len(images) == 0:
                    logging.info(
                        f"There are no images for {Fore.CYAN}{dataset_class}{Fore.RESET}, skipping the augmentation process for this class."
                    )
                    return

                logging.info(
                    f"Augmentating {Fore.CYAN}{dataset_class}{Fore.RESET} {Fore.CYAN}{len(images)}{Fore.RESET} images to a total of {Fore.CYAN}{len(images)+(len(images)*augmentation_amount)}{Fore.RESET} images..."
                )
                for image_file in images:
                    image = cv2.imread(
                        f"{self.dataset_path}\\train\\{dataset_class}\\{image_file}"
                    )

                    image_id = (
                        image_file.split(".png")[0].split(".jpg")[0].split(".jpeg")[0]
                    )

                    image_height, image_width, image_channels = image.shape
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    transform = A.Compose(
                        [
                            A.HorizontalFlip(p=0.5),
                            # A.ShiftScaleRotate(p=0.5),
                            A.RandomCropFromBorders(p=0.5),
                            A.RandomBrightnessContrast(p=0.5),
                            A.RGBShift(
                                r_shift_limit=(-50, 50),
                                g_shift_limit=(-50, 50),
                                b_shift_limit=(-50, 50),
                                p=0.5,
                            ),
                            A.CLAHE(p=0.5),
                            A.ISONoise(intensity=(0.3, 0.5), p=0.5),
                            A.RandomGravel(p=0.5),
                            A.HueSaturationValue(p=0.5),
                        ]
                    )

                    random.seed(7)
                    data_to_save = []

                    for i in range(0, augmentation_amount):
                        data_to_save.append(transform(image=image))

                    annotation_id = 0
                    for data in data_to_save:
                        image = cv2.cvtColor(data["image"], cv2.COLOR_BGR2RGB)
                        cv2.imwrite(
                            f"{self.dataset_path}\\train\\{dataset_class}\\{image_id}-{annotation_id}.png",
                            image,
                        )

                        annotation_id += 1

            logging.info("Succesfully augmented all images.")
        except:
            pass

    def _rollback_dataset(
        self, snapshot_info={}, rollback_version="", dataset_folder="", class_folder="", **kwargs
    ):
        shutil.rmtree(f"{self.dataset_path}\\{dataset_folder}\\{class_folder}")
        os.makedirs(f"{self.dataset_path}\\{dataset_folder}\\{class_folder}")

        for file_data in snapshot_info[rollback_version][dataset_folder][class_folder]:
            img_bin = file_data["binary"]
            img_name = file_data["filename"]

            encoded_content = img_bin.encode()
            content = base64.b64decode(encoded_content)

            with open(
                f"{self.dataset_path}\\{dataset_folder}\\{class_folder}\\{img_name}",
                "wb",
            ) as wf:
                wf.write(content)

        logging.info(
            f"Succesfully executed rollback at {Fore.CYAN}{dataset_folder}{Fore.RESET}"
        )

    def _status_show_additional_info(self, **kwargs):
        classes = []
        for x in os.walk(f"{self.dataset_path}\\staging"):
            if len(x[1]) > 0:
                classes = x[1]

        staging = self._status_sum_staging()
        train = self._status_sum_train()

        train_images_count = 0
        val_images_count = 0
        # Image Count
        for dataset_class in classes:
            train_images_count += sum(
                len(files)
                for _, _, files in os.walk(
                    f"{self.dataset_path}\\train\\{dataset_class}"
                )
            )
            val_images_count += sum(
                len(files)
                for _, _, files in os.walk(f"{self.dataset_path}\\val\\{dataset_class}")
            )

        logging.info(f"Images on {Fore.CYAN}train{Fore.RESET}: {train_images_count}")
        logging.info(f"Images on {Fore.CYAN}val{Fore.RESET}: {val_images_count}")

        # Class count
        if train == 0:
            return

        train_count_info = {}
        val_count_info = {}

        for dataset_class in classes:
            train_count_info[dataset_class] = 0
            val_count_info[dataset_class] = 0

        for dataset_class in classes:
            train_count_info[dataset_class] += sum(
                len(files)
                for _, _, files in os.walk(
                    f"{self.dataset_path}\\train\\{dataset_class}"
                )
            )
            val_count_info[dataset_class] += sum(
                len(files)
                for _, _, files in os.walk(f"{self.dataset_path}\\val\\{dataset_class}")
            )

        logging.info("")
        logging.info(f"Class count on {Fore.CYAN}train{Fore.RESET}:")
        logging.info("")
        for class_id in train_count_info:
            try:
                if train_count_info[class_id] == 0:
                    logging.info(
                        f"{Fore.CYAN}{class_id}{Fore.RESET}: {Fore.LIGHTRED_EX}{train_count_info[class_id]}{Fore.RESET}"
                    )
                else:
                    logging.info(
                        f"{Fore.CYAN}{class_id}{Fore.RESET}: {train_count_info[class_id]}"
                    )
            except:
                pass

        logging.info("")
        logging.info(f"Class count on {Fore.CYAN}val{Fore.RESET}:")
        logging.info("")
        for class_id in val_count_info:
            try:
                if val_count_info[class_id] == 0:
                    logging.info(
                        f"{Fore.CYAN}{class_id}{Fore.RESET}: {Fore.LIGHTRED_EX}{val_count_info[class_id]}{Fore.RESET}"
                    )
                else:
                    logging.info(
                        f"{Fore.CYAN}{class_id}{Fore.RESET}: {val_count_info[class_id]}"
                    )
            except:
                pass

    def yaml(self, **kwargs):
        logging.info(
            f"Your dataset type is defined as {Fore.CYAN}Classification{Fore.RESET}, this type of project doesn't require a {Fore.CYAN}data.yaml{Fore.RESET}."
        )