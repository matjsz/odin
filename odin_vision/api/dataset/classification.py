import base64
import json
import logging
import os
import shutil
from typing import Annotated, Literal

from colorama import Fore
from odin_vision.api.dataset.base import OdinDatasetBase
from odin_vision.api.internal.exceptions import InvalidSplitPercentagesError

class ClassificationSplit:
    def __init__(self, images):
        self.images = {}

class OdinDatasetClassification(OdinDatasetBase): 
    def publish(self, update_type: Literal["major", "minor", "fix"]="major", train: Annotated[int, "Value between 0 and 100"]=70, val: Annotated[int, "Value between 0 and 100"]=30):
        if train+val < 100 or train+val > 100:
            raise InvalidSplitPercentagesError(train, val)
        
        classes = []

        base_version = json.loads(
            open(f"{self.path}\\dataset.json", "r", encoding="utf8").read()
        )["version"]
        temp_version = self._upgrade_version(base_version, update_type)

        # SNAPSHOT UPDATE
        snapshot = {
            "staging": {},
            "train": {},
            "val": {},
        }

        for x in os.walk(f"{self.path}\\staging"):
            if len(x[1]) > 0:
                classes = x[1]

        for dataset_class in classes:
            if (
                sum(
                    len(files)
                    for _, _, files in os.walk(
                        f"{self.path}\\staging\\{dataset_class}"
                    )
                )
                == 0
            ):
                if self.project.verbose:
                    logging.info(
                        f"The {Fore.CYAN}Staging{Fore.RESET} dataset is empty, so nothing will be published or updated."
                    )
                return

            if self.project.verbose:
                logging.info(f"Publishing {Fore.CYAN}{dataset_class}{Fore.RESET}")

            if (
                sum(
                    len(files)
                    for _, _, files in os.walk(
                        f"{self.path}\\staging\\{dataset_class}"
                    )
                )
                == 0
            ):
                if self.project.verbose:
                    logging.info(
                        f"The {Fore.CYAN}Staging{Fore.RESET} dataset for class {Fore.CYAN}{dataset_class}{Fore.RESET} is empty, so nothing will be published or updated for this class. Skipping this one."
                    )
                return

            if self.project.verbose:
                logging.info("Publishing with the following splits:")
                logging.info(f"{Fore.CYAN}train{Fore.RESET}: {train}%")
                logging.info(f"{Fore.CYAN}val{Fore.RESET}: {val}%")

            count_train = int(
                (train / 100)
                * sum(
                    len(files)
                    for _, _, files in os.walk(
                        f"{self.path}\\staging\\{dataset_class}"
                    )
                )
            )

            count_val = int(
                (val / 100)
                * sum(
                    len(files)
                    for _, _, files in os.walk(
                        f"{self.path}\\staging\\{dataset_class}"
                    )
                )
            )

            self._execute_data_publishment(
                snapshot, "train", dataset_class, count_train
            )
            if self.project.verbose:
                logging.info(
                    f"Succesfully published {Fore.GREEN}train{Fore.RESET} data for class {Fore.CYAN}{dataset_class}{Fore.RESET}"
                )
            self._execute_data_publishment(snapshot, "val", dataset_class, count_val)
            if self.project.verbose:
                logging.info(
                    f"Succesfully published {Fore.GREEN}val{Fore.RESET} data for class {Fore.CYAN}{dataset_class}{Fore.RESET}"
                )

        dataset_info = json.loads(
            open(f"{self.path}\\dataset.json", "r", encoding="utf8").read()
        )
        snapshot_info = json.loads(
            open(f"{self.path}\\snapshot.json", "r", encoding="utf8").read()
        )

        snapshot_info[temp_version] = snapshot
        dataset_info["version"] = temp_version

        with open(f"{self.path}\\dataset.json", "w", encoding="utf8") as wf:
            wf.write(json.dumps(dataset_info, indent=4))
            
        if self.project.verbose:
            logging.info(
                f"Succesfully updated dataset version to {Fore.CYAN}v{temp_version}{Fore.RESET}"
            )

        with open(f"{self.path}\\snapshot.json", "w", encoding="utf8") as wf:
            wf.write(json.dumps(snapshot_info, indent=4))
        if self.project.verbose:
            logging.info(
                f"Succesfully registered snapshots for dataset {Fore.CYAN}v{temp_version}{Fore.RESET}"
            )
    
    def _execute_data_publishment(self, snapshot={}, split="", dataset_class="", split_max_file=0, **kwargs):
        images = []

        for _, _, files in os.walk(f"{self.path}\\staging\\{dataset_class}"):
            images = files
            break

        for i in range(0, split_max_file + 1):
            try:
                image_stage_path = (
                    f"{self.path}\\staging\\{dataset_class}\\{images[i]}"
                )
                
                self._try_create_folder(f"{self.path}\\{split}\\{dataset_class}")
                
                image_publish_path = (
                    f"{self.path}\\{split}\\{dataset_class}\\{images[i]}"
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
    
    def _status_sum_staging(self, **kwargs):
        classes = []
        for x in os.walk(f"{self.path}\\staging"):
            if len(x[1]) > 0:
                classes = x[1]

        final_sum = 0
        for dataset_class in classes:
            final_sum += sum(
                len(files)
                for _, _, files in os.walk(
                    f"{self.path}\\staging\\{dataset_class}"
                )
            )
            
        return final_sum
    
    def _status_sum_train(self, **kwargs):
        classes = []
        for x in os.walk(f"{self.path}\\staging"):
            if len(x[1]) > 0:
                classes = x[1]

        final_sum = 0
        for dataset_class in classes:
            final_sum += sum(
                len(files)
                for _, _, files in os.walk(
                    f"{self.path}\\train\\{dataset_class}"
                )
            )
        
        return final_sum
           
    def _get_split(self, split: str):
        split_images = {}
            
        dataset_classes = []
        
        for _, dataset_folders, _ in os.walk(f"{self.path}\\{split}"):
            dataset_classes = dataset_folders                
            break
        
        for dataset_class in dataset_classes:
            class_images = []
            
            for _, _, images in os.walk(f"{self.path}\\{split}\\{dataset_class}"):
                class_images = images
                break
            
            split_images[dataset_class] = class_images
            
        return ClassificationSplit(split_images)