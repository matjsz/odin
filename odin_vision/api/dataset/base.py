import base64
import json
import logging
import os
import shutil
from typing import Annotated, Literal

from colorama import Fore
from odin_vision.api.internal.exceptions import CouldNotFindDatasetError

class OdinDatasetBase:
    def __init__(self, name: str | None = None, project=None, path: str | None = None, allow_creation: bool = False, **kwargs):
        self.name = name
        self.allow_creation = allow_creation
        self.project = project
        
        self._check_dataset_path(path)
        self.path = path
        
        self.staging = self._get_split("staging")
        self.train = self._get_split("train")
        self.val = self._get_split("val")
        
        self.version = self._get_version()
        self._get_status()
    
    def publish(self, update_type: Literal["major", "minor", "fix"]="major", train: Annotated[int, "Value between 0 and 100"]=70, val: Annotated[int, "Value between 0 and 100"]=30):
        return NotImplementedError()
            
    def _upgrade_version(self, base_version, update_size):
        version = list(map(lambda x: int(x), base_version.split(".")))

        if update_size == "major":
            version[0] += 1
        elif update_size == "minor":
            version[1] += 1
        elif update_size == "fix":
            version[2] += 1

        upgraded_version = f"{version[0]}.{version[1]}.{version[2]}"

        return upgraded_version
    
    def _try_create_folder(self, folder_path):
        try:
            os.makedirs(folder_path)
        except FileExistsError:
            pass
        except Exception as e:
            if self.project.verbose:
                logging.info(f"Something went wrong while trying to create {Fore.CYAN}{folder_path}{Fore.RESET}: {e}")
        
    def _execute_data_publishment(self, snapshot={}, split="", dataset_class="", split_max_file=0, **kwargs):
        return NotImplementedError()
            
    def _add_artifact_to_version_snapshot(
        self, snapshot={}, dataset_split="", label_name="", label_binary="", image_name="", image_binary="", **kwargs
    ):
        try:
            snapshot[dataset_split]['images'].append(
                {
                    "filename": image_name,
                    "binary": base64.b64encode(image_binary).decode("utf8"),
                }
            )
        except:
            snapshot[dataset_split]['images'] = [
                {
                    "filename": image_name,
                    "binary": base64.b64encode(image_binary).decode("utf8"),
                }
            ]
            
        try:
            snapshot[dataset_split]['labels'].append(
                {
                    "filename": label_name,
                    "binary": base64.b64encode(label_binary).decode("utf8"),
                }
            )
        except:
            snapshot[dataset_split]['labels'] = [
                {
                    "filename": label_name,
                    "binary": base64.b64encode(label_binary).decode("utf8"),
                }
            ]
            
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
        
    def _status_sum_staging(self, **kwargs):
        return NotImplementedError()
    
    def _status_sum_train(self, **kwargs):
        return NotImplementedError()
        
    def _get_version(self):
        try:
            with open(f"{self.path}\\dataset.json", "r", encoding="utf8") as f:
                dataset_data = json.loads(f.read())
                self.version = dataset_data['version']
        except:
            self.version = "No version available."
            
    def _get_status(self):
        try:
            train_count = self._status_sum_train()
            staging_count = self._status_sum_staging()
            
            if staging_count > 0 and train_count > 0:
                self.status = "published_staging"
            elif staging_count > 0 and train_count == 0:
                self.status = "staging"
            elif train_count > 0 and staging_count == 0:
                self.status = "published"
            else:
                self.status = "empty"
        except:
            self.status = "No status available."
        
    def _get_split(self, split: str):
        return NotImplementedError()
        
    def _check_dataset_path(self, path):
        if not os.path.exists(path):
            if self.allow_creation:
                self.project.datasets.create(self.name)
            else:
                raise CouldNotFindDatasetError
            