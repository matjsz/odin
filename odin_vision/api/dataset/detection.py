import base64
import json
import logging
import os
import shutil
from typing import Annotated, Literal

from colorama import Fore
from odin_vision.api.dataset.base import OdinDatasetBase
from odin_vision.api.internal.exceptions import InvalidSplitPercentagesError

class DetectionSplit:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

class OdinDatasetDetection(OdinDatasetBase):
    def publish(self, update_type: Literal["major", "minor", "fix"]="major", train: Annotated[int, "Value between 0 and 100"]=70, val: Annotated[int, "Value between 0 and 100"]=30):
        if train+val < 100 or train+val > 100:
            raise InvalidSplitPercentagesError(train, val)
        
        if (
            sum(
                len(files)
                for _, _, files in os.walk(f"{self.path}\\staging\\images")
            )
            == 0
        ):
            logging.info(
                f"The {Fore.CYAN}Staging{Fore.RESET} dataset is empty, so nothing will be published or updated."
            )
            return
        
        base_version = json.loads(
            open(f"{self.path}\\dataset.json", "r", encoding="utf8").read()
        )["version"]
        temp_version = self._upgrade_version(base_version, update_type)
        
        snapshot = {
            "staging": {},
            "train": {},
            "val": {},
        }
            
        logging.info("Publishing with the following splits:")
        logging.info(f"{Fore.CYAN}train{Fore.RESET}: {train}%")
        logging.info(f"{Fore.CYAN}val{Fore.RESET}: {val}%")
        
        count_train = int(
            (train / 100)
            * sum(
                len(files)
                for _, _, files in os.walk(f"{self.path}\\staging\\images")
            )
        )

        count_val = int(
            (val / 100)
            * sum(
                len(files)
                for _, _, files in os.walk(f"{self.path}\\staging\\images")
            )
        )
        
        self._execute_data_publishment(snapshot, "train", count_train)
        logging.info(
            f"Succesfully published {Fore.GREEN}train{Fore.RESET} data."
        )
        self._execute_data_publishment(snapshot, "val", count_val)
        logging.info(
            f"Succesfully published {Fore.GREEN}val{Fore.RESET} data."
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
        logging.info(
            f"Succesfully updated dataset version to {Fore.CYAN}v{temp_version}{Fore.RESET}"
        )

        with open(f"{self.path}\\snapshot.json", "w", encoding="utf8") as wf:
            wf.write(json.dumps(snapshot_info, indent=4))
        logging.info(
            f"Succesfully registered snapshots for dataset {Fore.CYAN}v{temp_version}{Fore.RESET}"
        )

    def _execute_data_publishment(self, snapshot={}, split="", dataset_class="", split_max_file=0, **kwargs):
        images = []
        labels = []

        for _, _, files in os.walk(f"{self.path}\\staging\\images"):
            images = files
            break

        for _, _, files in os.walk(f"{self.path}\\staging\\labels"):
            labels = files
            break
        
        for i in range(0, split_max_file + 1):
            try:
                image_stage_path = (
                    f"{self.path}\\staging\\images\\{images[i]}"
                )
                image_publish_path = (
                    f"{self.path}\\{split}\\images\\{images[i]}"
                )

                label_stage_path = (
                    f"{self.path}\\staging\\labels\\{labels[i]}"
                )
                label_publish_path = (
                    f"{self.path}\\{split}\\labels\\{labels[i]}"
                )

                shutil.move(
                    image_stage_path,
                    image_publish_path,
                )

                shutil.move(
                    label_stage_path,
                    label_publish_path,
                )
                
                image_binary = open(image_publish_path, "rb").read()
                label_binary = open(label_publish_path, "rb").read()
                
                self._add_artifact_to_version_snapshot(snapshot, split, labels[i], label_binary, images[i], image_binary)
            except IndexError:
                pass
    
    def _status_sum_staging(self, **kwargs):
        return sum(
            len(files)
            for _, _, files in os.walk(
                f"{self.dataset_path}\\staging\\images"
            )
        )
    
    def _status_sum_train(self, **kwargs):
        return sum(
            len(files)
            for _, _, files in os.walk(
                f"{self.dataset_path}\\train\\images"
            )
        )
           
    def _get_split(self, split: str):
        split_images = []
        split_labels = []
        
        for _, _, images in os.walk(f"{self.path}\\{split}\\images"):
            split_images = images
            break
        
        for _, _, labels in os.walk(f"{self.path}\\{split}\\labels"):
            split_labels = labels
            break
            
        return DetectionSplit(split_images, split_labels)