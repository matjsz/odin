import base64
import datetime
import json
import logging
from math import e
import os
import shutil
from typing import Annotated, Literal
from colorama import Fore
import yaml
from odin_vision.api.internal.exceptions import CouldNotCreateDatasetInPathError, CouldNotCreateProjectInPathError, CouldNotFindDatasetByNameError, CouldNotFindDatasetError, CouldNotFindProjectError, DatasetNotInformedError, InvalidSplitPercentagesError, ProjectNameInvalidError, ProjectTypeInvalidError
from odin_vision.chronicle.utils import get_chronicle_name
from odin_vision.constants import EXAMPLE_YAML, PROJECT_TYPES, README_CHRONICLES, README_WEIGHTS

logging.basicConfig(
    level=logging.INFO, format=f"[{Fore.CYAN}ODIN{Fore.RESET}][{Fore.YELLOW}%(asctime)s{Fore.RESET}] %(message)s"
)

class ClassificationSplit:
    def __init__(self, images):
        self.images = {}

class DetectionSplit:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

class OdinDataset:
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
        self.status = self._get_status()
        
    def publish(self, update_type: Literal["major", "minor", "fix"]="major", train: Annotated[int, "Value between 0 and 100"]=70, val: Annotated[int, "Value between 0 and 100"]=30):
        if train+val < 100 or train+val > 100:
            raise InvalidSplitPercentagesError(train, val)
        
        if self.project.type == "classification":
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

                self._execute_classification_data_publishment(
                    snapshot, "train", dataset_class, count_train
                )
                if self.project.verbose:
                    logging.info(
                        f"Succesfully published {Fore.GREEN}train{Fore.RESET} data for class {Fore.CYAN}{dataset_class}{Fore.RESET}"
                    )
                self._execute_classification_data_publishment(snapshot, "val", dataset_class, count_val)
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
        elif self.project.type == "detection":
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
            
            self._execute_detection_data_publishment(snapshot, "train", count_train)
            logging.info(
                f"Succesfully published {Fore.GREEN}train{Fore.RESET} data."
            )
            self._execute_detection_data_publishment(snapshot, "val", count_val)
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
        
    def _execute_detection_data_publishment(self, snapshot={}, split="", split_max_file=0, **kwargs):
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
        
    def _execute_classification_data_publishment(self, snapshot={}, split="", dataset_class="", split_max_file=0, **kwargs):
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
        if self.project.type == "classification":
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
        elif self.project.type == "detection":
            return sum(
                len(files)
                for _, _, files in os.walk(
                    f"{self.dataset_path}\\staging\\images"
                )
            )

    def _status_sum_train(self, **kwargs):
        if self.project.type == "classification":
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
        elif self.project.type == "detection":
            return sum(
                len(files)
                for _, _, files in os.walk(
                    f"{self.dataset_path}\\train\\images"
                )
            )
        
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
        def detection():
            split_images = []
            split_labels = []
            
            for _, _, images in os.walk(f"{self.path}\\{split}\\images"):
                split_images = images
                break
            
            for _, _, labels in os.walk(f"{self.path}\\{split}\\labels"):
                split_labels = labels
                break
                
            return DetectionSplit(split_images, split_labels)
        
        def classification():
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
        
        if self.project.type == "classification":
            classification()
        elif self.project.type == "detection":
            detection()
        
    def _check_dataset_path(self, path):
        if not os.path.exists(path):
            if self.allow_creation:
                OdinDatasetsController(self.project).create(self.name)
            else:
                raise CouldNotFindDatasetError

class OdinDatasetsController:
    def __init__(self, project, datasets: list[OdinDataset] = []):
        self.project = project
        self.datasets = datasets
        
        if len(datasets) == 0:
            self._get_datasets()
        
    def _get_datasets(self):
        for _, dataset_folders, _ in os.walk(f"{self.project.path}\\datasets"):
            for dataset_folder in dataset_folders:
                dataset_path = os.path.abspath(f"{self.project.path}\\datasets\\{dataset_folder}")
                dataset_obj = OdinDataset(dataset_folder, self.project, dataset_path)
                self.datasets.append(dataset_obj)
            break
    
    def get(self, by_name: str | None = None, by_chronicle: str | None = None, allow_creation: bool = False):
        if by_name:
            for dataset in self.datasets:
                if dataset.name == by_name:
                    return dataset
            if allow_creation:                
                dataset = self.create(by_name)
                return dataset
            else:
                raise CouldNotFindDatasetByNameError(by_name)
        elif by_chronicle:
            raise NotImplementedError()
        
    def create(self, name: str | None = None, **kwargs):
        dataset_path = f"{self.project.path}\\datasets\\{name}"
        
        if os.path.exists(dataset_path):
            for dataset in self.datasets:
                if dataset.name == name:
                    return dataset
            return None
        
        try:
            dataset_info = {
                "type": self.project.type,
                "version": "0.1.0"
            }
            
            os.makedirs(f"{dataset_path}")

            with open(f"{dataset_path}\\dataset.json", "w", encoding="utf8") as wf:
                wf.write(json.dumps(dataset_info))

            with open(f"{dataset_path}\\snapshot.json", "w", encoding="utf8") as wf:
                wf.write(json.dumps({}))
            
            os.makedirs(f"{dataset_path}\\staging")
            os.makedirs(f"{dataset_path}\\train")
            os.makedirs(f"{dataset_path}\\val")
            
            if self.project.type == "detection":
                os.makedirs(f"{dataset_path}\\staging\\images")
                os.makedirs(f"{dataset_path}\\staging\\labels")
                os.makedirs(f"{dataset_path}\\train\\images")
                os.makedirs(f"{dataset_path}\\train\\labels")
                os.makedirs(f"{dataset_path}\\val\\images")
                os.makedirs(f"{dataset_path}\\val\\labels")
            elif self.project.type == "classification":
                os.makedirs(f"{dataset_path}\\staging\\class_1")
        except:
            raise CouldNotCreateDatasetInPathError(dataset_path)
        
        new_dataset = OdinDataset(name, self.project, dataset_path)
        
        self.datasets.append(new_dataset)
        
        return new_dataset

    def publish(self, dataset: str | OdinDataset | None = None):
        if dataset and isinstance(dataset, str):
            dataset_target = self.get(by_name=dataset)
            dataset_target.publish()
        elif dataset and isinstance(dataset, OdinDataset):
            dataset.publish()
        else:
            raise DatasetNotInformedError()

class OdinChronicle:
    def __init__(self, name: str | None=None, project=None, path: str | None=None):
        self.name = name
        self.project = project
        self.path = path

class OdinChroniclesController:
    def __init__(self, project, chronicles: list[OdinChronicle] = []):
        self.project = project
        self.chronicles = chronicles
        
        if len(chronicles) == 0:
            self._get_chronicles()
            
    def _get_chronicles(self):
        for _, chronicle_folders, _ in os.walk(f"{self.project.path}\\chronicles"):
            for chronicle_folder in chronicle_folders:
                chronicle_path = os.path.abspath(f"{self.project.path}\\chronicles\\{chronicle_folder}")
                
                chronicle_obj = OdinChronicle(chronicle_folder, self.project, chronicle_path)
                self.chronicles.append(chronicle_obj)
            break
        
    def get(self, by_name: str | None=None):
        if by_name:
            for chronicle in self.chronicles:
                if chronicle.name == by_name:
                    return chronicle
            else:
                raise CouldNotFindDatasetByNameError(by_name)

# OdinModel(file, chronicle_folder, self.project, staging_model_path)
class OdinModel:
    def __init__(self, name: str | None=None, chronicle: str | None=None, dataset: str | None=None, project=None, model_path: str | None=None, status: str | None=None, **kwargs):
        self.name = name
        self.chronicle = chronicle
        self.dataset = dataset
        self.project = project
        self.path = model_path
        self.status = status
        
    def _get_chronicle_data(self, chronicle_path):
        with open(chronicle_path+"\\chronicle.json", "r", encoding="utf8") as f:
            data = json.loads(f.read())
        return data
        
    def test(self):
        raise NotImplementedError()
    
    def publish(self):
        if self.status == "staging":
            chronicle_data = self._get_chronicle_data(f"{self.project.path}\\chronicles\\{self.chronicle}")
            
            logging.info(f"Publishing model {Fore.CYAN}{self.name}{Fore.RESET}...")
        
            now = datetime.datetime.now()
            new_model_name = f"{self.project.name}.{now.year}.{now.month}.{now.day}.{self.chronicle}"
            
            shutil.copy(f"{self.project.path}\\chronicles\\{self.chronicle}\\weights\\{self.name}", f"{self.project.path}\\weights\\{new_model_name}.pt")
            
            with open(f"{self.project.path}\\weights\\{new_model_name}.json", "w", encoding="utf8") as wf:
                wf.write(json.dumps({
                    "created_at": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "chronicle": self.chronicle,
                    "dataset": chronicle_data['dataset']
                }))

            logging.info(f"Succesfully published model {Fore.CYAN}{self.name}{Fore.RESET} as {Fore.CYAN}{new_model_name}{Fore.RESET}...")

class OdinModelsController:
    def __init__(self, project, staging_models: list[OdinModel]=[], published_models: list[OdinModel] = []):
        self.project = project
        self.staging_models = staging_models
        self.published_models = published_models
        
        if len(staging_models) == 0:
            self._get_all_staging_models()
        if len(published_models) == 0:
            self._get_all_published_models()
            
    def _get_all_staging_models(self, return_list=False):
        models = []
        
        for _, chronicle_folders, _ in os.walk(f"{self.project.path}\\chronicles"):
            for chronicle_folder in chronicle_folders:
                for _, _, files in os.walk(f"{self.project.path}\\chronicles\\{chronicle_folder}\\weights"):
                    for file in files:
                        if file not in ["last.pt", "best.pt", "README.md"] and not file.startswith("epoch"):
                            staging_model_path = os.path.abspath(f"{self.project.path}\\chronicles\\{chronicle_folder}\\weights\\{file}")

                            with open(f"{self.project.path}\\chronicles\\{chronicle_folder}\\chronicle.json", "r", encoding="utf8") as f:
                                model_data = json.loads(f.read())

                            staging_model_obj = OdinModel(file, chronicle_folder, model_data["dataset"], self.project, staging_model_path, "staging")
                            
                            if return_list:
                                models.append(staging_model_obj)
                            else:
                                self.staging_models.append(staging_model_obj)
                    break
            break
        
        if return_list:
            return models
        
    def _get_last_staging_model(self):
        for _, chronicle_folders, _ in os.walk(f"{self.project.path}\\chronicles"):
            for chronicle_folder in chronicle_folders:
                for _, _, files in os.walk(f"{self.project.path}\\chronicles\\{chronicle_folder}\\weights"):
                    files.sort(reverse=True)
                    for file in files:
                        if file not in ["last.pt", "best.pt", "README.md"] and not file.startswith("epoch"):
                            staging_model_path = os.path.abspath(f"{self.project.path}\\chronicles\\{chronicle_folder}\\weights\\{file}")
                            
                            with open(f"{self.project.path}\\chronicles\\{chronicle_folder}\\chronicle.json", "r", encoding="utf8") as f:
                                model_data = json.loads(f.read())

                            staging_model_obj = OdinModel(file, chronicle_folder, model_data["dataset"], self.project, staging_model_path, "staging")
                            
                            return staging_model_obj
                    break
            break
    
    def _get_all_published_models(self, return_list=False):
        models = []
        
        for _, _, files in os.walk(f"{self.project.path}\\weights"):
            for file in files:
                if file not in ["last.pt", "best.pt", "README.md"] and not file.startswith("epoch") and not file.endswith(".json"):
                    data_file = file[:-3]+".json"
                    data_file_path = os.path.abspath(f"{self.project.path}\\weights\\{data_file}")
                    with open(data_file_path, "r", encoding="utf8") as f:
                        published_model_data = json.loads(f.read())
                    chronicle_name = published_model_data['chronicle']
                    dataset_name = published_model_data['dataset']
                    
                    published_model_path = os.path.abspath(f"{self.project.path}\\weights\\{file}")
                    published_model_obj = OdinModel(file, chronicle_name, dataset_name, self.project, published_model_path, "published")

                    if return_list:
                        models.append(published_model_obj)
                    else:
                        self.published_models.append(published_model_obj)
            break
        
        if return_list:
            return models

    def _get_last_published_model(self):
        for _, _, files in os.walk(f"{self.project.path}\\weights"):
            files.sort(reverse=True)
            for file in files:
                if file not in ["last.pt", "best.pt", "README.md"] and not file.startswith("epoch"):
                    data_file = file[:-3]+".json"
                    data_file_path = os.path.abspath(f"{self.project.path}\\weights\\{data_file}")
                    with open(data_file_path, "r", encoding="utf8") as f:
                        published_model_data = json.loads(f.read())
                    chronicle_name = published_model_data['chronicle']
                    dataset_name = published_model_data['dataset']
                    
                    published_model_path = os.path.abspath(f"{self.project.path}\\weights\\{file}")
                    published_model_obj = OdinModel(file, chronicle_name, dataset_name, self.project, published_model_path, "published")

                    return published_model_obj
            break

    def _try_create_model_snapshot(self, type, chronicle_path):
        now = datetime.datetime.now()
        
        logging.info("Saving results...")
        
        if type == "naive":
            try:
                shutil.copyfile(f"{chronicle_path}\\weights\\best.pt", f"{chronicle_path}\\weights\\pre_{now.year}_{now.month}_{now.day}_{now.hour}{now.minute}{now.second}.pt")

                logging.info(f"Model saved at: {Fore.CYAN}{chronicle_path}\\weights\\pre.{now.year}.{now.month}.{now.day}.{now.hour}{now.minute}{now.second}.pt{Fore.RESET}")
            except Exception as e:
                logging.info(f"Couldn't create model snapshot, something went wrong: {e}")
        elif type == "wise":
            try:
                shutil.copyfile(f"{chronicle_path}\\weights\\best.pt", f"{chronicle_path}\\weights\\{now.year}_{now.month}_{now.day}_{now.hour}{now.minute}{now.second}.pt")

                logging.info(f"Model saved at: {Fore.CYAN}{chronicle_path}\\weights\\{now.year}_{now.month}_{now.day}_{now.hour}{now.minute}{now.second}.pt{Fore.RESET}")
            except Exception as e:
                logging.info(f"Couldn't create model snapshot, something went wrong: {e}")

    def _try_create_folder(self, folder_path):
        try:
            os.makedirs(folder_path)
        except FileExistsError:
            pass
        except Exception as e:
            logging.info(f"Something went wrong while trying to create {Fore.CYAN}{folder_path}{Fore.RESET}: {e}")

    def get(self, model_filter: str="all", model_type: str="staging", by_dataset: str | OdinDataset | None=None, by_chronicle: str | OdinChronicle | None=None, by_name: str | None=None):
        if model_filter == "all" and model_type == "staging":
            models = self._get_all_staging_models(True)
            selected_models = []
            
            for model in models:
                if by_dataset:
                    if isinstance(by_dataset, str) and model.dataset == by_dataset:
                        selected_models.append(model)
                    elif isinstance(by_dataset, OdinDataset) and model.dataset == by_dataset.name:
                        selected_models.append(model)
                elif by_chronicle:
                    if isinstance(by_chronicle, str) and model.chronicle == by_chronicle:
                        selected_models.append(model)
                    elif isinstance(by_chronicle, OdinChronicle) and model.chronicle == by_chronicle.name:
                        selected_models.append(model)
                elif by_name:
                    if model.name == by_name:
                        selected_models.append(model)
            
            return selected_models
        elif model_filter == "all" and model_type == "published":
            models = self._get_all_published_models(True)
            selected_models = []
            
            for model in models:
                if by_dataset:
                    if isinstance(by_dataset, str) and model.dataset == by_dataset:
                        selected_models.append(model)
                    elif isinstance(by_dataset, OdinDataset) and model.dataset == by_dataset.name:
                        selected_models.append(model)
                elif by_chronicle:
                    if isinstance(by_chronicle, str) and model.chronicle == by_chronicle:
                        selected_models.append(model)
                    elif isinstance(by_chronicle, OdinChronicle) and model.chronicle == by_chronicle.name:
                        selected_models.append(model)
                elif by_name:
                    if model.name == by_name:
                        selected_models.append(model)
            
            return selected_models
            
        elif model_filter == "last" and model_type == "staging":
            model = self._get_last_staging_model()
            
            if not model:
                return []
    
            if by_dataset:
                if model.dataset == by_dataset and isinstance(by_dataset, str):
                    return [model]
                elif model.dataset == by_dataset.name and isinstance(by_dataset, OdinDataset):
                    return [model]
            elif by_chronicle:
                if model.chronicle == by_chronicle and isinstance(by_chronicle, str):
                    return [model]
                elif model.chronicle == by_chronicle.name and isinstance(by_chronicle, OdinChronicle):
                    return [model]
            elif by_name:
                if model.name == by_name:
                    return [model]
            return []
        elif model_filter == "last" and model_type == "published":
            model = self._get_last_published_model()
            
            if not model:
                return []
            
            if by_dataset:
                if model.dataset == by_dataset and isinstance(by_dataset, str):
                    return [model]
                elif model.dataset == by_dataset.name and isinstance(by_dataset, OdinDataset):
                    return [model]
            elif by_chronicle:
                if model.chronicle == by_chronicle and isinstance(by_chronicle, str):
                    return [model]
                elif model.chronicle == by_chronicle.name and isinstance(by_chronicle, OdinChronicle):
                    return [model]
            elif by_name:
                if model.name == by_name:
                    return [model]
            return []

    def train(self, dataset: str | OdinDataset | None=None, epochs: int=10, device: str="cpu", base_model: str | None=None, chronicle: str | None=None, subset: int=100, **kwargs):
        if not chronicle:
            chronicle = get_chronicle_name()
        
        if subset < 100:
            model_type = "naive"
        else:
            model_type = "wise"
            
        if not base_model:
            if self.project.type == "detection":
                base_model = "yolo11n.pt"
            elif self.project.type == "classification":
                base_model = "yolo11n-cls.pt"
            
        if not dataset:
            return
        elif isinstance(dataset, OdinDataset):
            dataset = dataset.name
            
        chronicles_path = f"{self.project.path}\\chronicles" 
        chronicle_path = f"{self.project.path}\\chronicles\\{chronicle}"
        dataset_path = f"{self.project.path}\\datasets\\{dataset}"

        logging.info(f"Starting {Fore.CYAN}training{Fore.RESET}...")
        logging.info(f"Version defined as {Fore.CYAN}{model_type}{Fore.RESET}")
        
        self._try_create_folder(chronicle_path)
        self._try_create_folder(chronicle_path+"\\weights")
        
        chronicle_info = {
            "name": chronicle,
            "dataset": dataset
        }
        
        with open(chronicle_path+"\\chronicle.json", "w", encoding="utf8") as wf:
            wf.write(json.dumps(chronicle_info, indent=4, ensure_ascii=False))

        if self.project.type == "classification":
            yolo_command = [
                "yolo",
                "classify",
                "train",
                f"data={dataset_path}",
                f"epochs={epochs}",
                f"batch={'-1' if device == 'gpu' else '2'}",
                f"model={base_model}",
                "amp=false",
                "patience=10",
                "save_period=5",
                f"device={'0' if device == 'gpu' else 'cpu'}",
                f"project={chronicles_path}",
                f"name={chronicle}",
                "exist_ok=true",
                "plots=true",
            ]
        elif self.project.type == "detection":
            yolo_command = [
                "yolo",
                "detect",
                "train",
                f"data={dataset_path}\\data.yaml",
                f"epochs={epochs}",
                f"batch={'-1' if device == 'gpu' else '2'}",
                f"model={base_model}",
                "amp=false",
                "patience=10",
                "save_period=5",
                f"device={'0' if device == 'gpu' else 'cpu'}",
                f"project={chronicles_path}",
                f"name={chronicle}",
                "exist_ok=true",
                "plots=true",
            ]
            
        if model_type == "naive":
            yolo_command.append(f"fraction={subset/100}")
        
        os.system(
            " ".join(yolo_command)
        )
        
        self._try_create_model_snapshot(model_type, chronicle_path)
        
        logging.info(f"Trained to chronicle {Fore.CYAN}{chronicle}{Fore.RESET}")
        logging.info(f"You can test this version by using the command {Fore.CYAN}odin test --chronicle {chronicle}{Fore.RESET}")

    def publish(self, model: str | OdinModel | None=None):
        if model and isinstance(model, str):
            model_target = self.get(by_name=model)
            model_target.publish()
        elif model and isinstance(model, OdinModel):
            model.publish()
        else:
            raise DatasetNotInformedError()
        
    def test(self, model: str | OdinModel | None=None):
        raise NotImplementedError()

class OdinProjectBase:
    def __init__(
      self,
      name: str | None = None,
      type: str | None = None,
      allow_creation: bool = False,
      project_dir: str | None = None,
      verbose: bool = False,
      **kwargs  
    ):
        raise NotImplementedError()
        
    def _check_project_name(self, name: str | None):
        if not name:
            raise ProjectNameInvalidError(name)
        
    def _check_project_type(self, will_create: bool, type: str | None):
        if will_create:
            if type not in PROJECT_TYPES:
                raise ProjectTypeInvalidError(type)
    
    def _get_project_pretty_type(self):
        pretty_types = {
            "classification": "Image Classification",
            "detection": "Object Detection"
        }
        
        return pretty_types[self.type]
        
    def _check_project_exists(self, name: str | None):
        """Searches for the project in the folders inside the root folder."""
        
        def read_project_yaml(path):
            with open(path, "r") as f:
                data = yaml.safe_load(f)
            return data
        
        project_path = ""
        project_found = False
        for root, _, files in os.walk('.'):
            for file in files:
                if file == "project.yaml":
                    yaml_path = f"{root}\\{file}"
                    
                    project_data = read_project_yaml(yaml_path)
                    
                    if project_data['name'] == name:
                        project_found = True
                        project_path = os.path.abspath(root)
                        break
            if project_found:
                break
        
        if project_found:
            return [True, project_path]
        return [False, os.path.abspath('.')]
    
    def _retrieve_project_data(self):
        with open(f"{self.path}\\project.yaml", "r", encoding="utf8") as f:
            project_data = yaml.safe_load(f.read())
        return project_data
    
    def _create_new_project(self):
        project_root = self.path
        if self.path:
            project_root = self.path.replace('/', '\\')
        
        try:
            if not os.path.exists(project_root):
                os.makedirs(project_root)
        except:
            raise CouldNotCreateProjectInPathError(project_root)

        project_general_info = {"name": self.name, "type": self.type, "version": "0.1.0"}

        if self.verbose:
            logging.info(
                f"Creating project structure for {Fore.CYAN}{self.name}{Fore.RESET}, a {Fore.CYAN}{self._get_project_pretty_type()}{Fore.RESET} project."
            )

        with open(f"{project_root}\\project.yaml", "w", encoding="utf8") as wf:
            wf.write(yaml.dump(project_general_info))

        # Datasets
        dataset_parent = f"{project_root}\\datasets"
        if not os.path.exists(dataset_parent):
            os.makedirs(dataset_parent)

            if self.verbose:
                logging.info(f"Succesfully created {Fore.CYAN}datasets{Fore.RESET}")

        if self.verbose:
            logging.info(f"Creating {Fore.CYAN}datasets examples{Fore.RESET}...")
            
        def classification():
            dataset_example_classif = f"{self.path}\\datasets\\dataset_1_classif"
            if not os.path.exists(dataset_example_classif):
                os.makedirs(dataset_example_classif)
                
                dataset_info = {
                    "type": self.type,
                    "version": "0.1.0"
                }

                with open(f"{self.path}\\datasets\\dataset_1_classif\\dataset.json", "w", encoding="utf8") as wf:
                    wf.write(json.dumps(dataset_info))

                with open(f"{self.path}\\datasets\\dataset_1_classif\\snapshot.json", "w", encoding="utf8") as wf:
                    wf.write(json.dumps({}))

                os.makedirs(f"{self.path}\\datasets\\dataset_1_classif\\class_1")
                os.makedirs(f"{self.path}\\datasets\\dataset_1_classif\\class_2")

                logging.info(
                    f"Succesfully created dataset example at: {Fore.CYAN}datasets/dataset_1_classif{Fore.RESET}"
                )
                
        def detection():
            dataset_example_obd = f"{self.path}\\datasets\\dataset_1_obd"
            if not os.path.exists(dataset_example_obd):
                os.makedirs(dataset_example_obd)
                
                dataset_info = {
                    "type": self.type,
                    "version": "0.1.0"
                }

                with open(f"{self.path}\\datasets\\dataset_1_obd\\dataset.json", "w", encoding="utf8") as wf:
                    wf.write(json.dumps(dataset_info))

                with open(f"{self.path}\\datasets\\dataset_1_obd\\snapshot.json", "w", encoding="utf8") as wf:
                    wf.write(json.dumps({}))

                os.makedirs(f"{self.path}\\datasets\\dataset_1_obd\\train")
                os.makedirs(f"{self.path}\\datasets\\dataset_1_obd\\val")
                os.makedirs(
                    f"{self.path}\\datasets\\dataset_1_obd\\train\\images"
                )
                os.makedirs(f"{self.path}\\datasets\\dataset_1_obd\\val\\images")
                os.makedirs(
                    f"{self.path}\\datasets\\dataset_1_obd\\train\\labels"
                )
                os.makedirs(f"{self.path}\\datasets\\dataset_1_obd\\val\\labels")

                with open(
                    f"{self.path}\\datasets\\dataset_1_obd\\data.yaml", "w"
                ) as wf:
                    wf.write(EXAMPLE_YAML)

                logging.info(
                    f"Succesfully created dataset example at: {Fore.CYAN}datasets/dataset_1_obd{Fore.RESET}"
                )
                
        if self.type == "detection":
            detection()
        elif self.type == "classification":
            classification()  
        
        # Chronicles
        chronicles_parent = f"{self.path}\\chronicles"
        if not os.path.exists(chronicles_parent):
            os.makedirs(chronicles_parent)

            with open(f"{chronicles_parent}\\README.md", "w", encoding="utf8") as wf:
                wf.write(README_CHRONICLES)

            if self.verbose:
                logging.info(f"Succesfully created {Fore.CYAN}chronicles{Fore.RESET}")

        # Weights
        weights_parent = f"{self.path}\\weights"
        if not os.path.exists(weights_parent):
            os.makedirs(weights_parent)

            with open(f"{weights_parent}\\README.md", "w", encoding="utf8") as wf:
                wf.write(README_WEIGHTS)

            if self.verbose:
                logging.info(f"Succesfully created {Fore.CYAN}weights{Fore.RESET}")
    
        self.project_data = project_general_info
        
class OdinProject(OdinProjectBase):
    def __init__(
      self,
      name: str | None = None,
      type: str | None = None,
      allow_creation: bool = False,
      project_dir: str | None = None,
      verbose: bool = False,
      **kwargs  
    ):
        # Project general info loading and input checks
        self.verbose = verbose
        self.project_data = {}
        
        self._check_project_name(name)
        self._check_project_type(allow_creation, type)
        
        self.name = name
        self.type = type
        
        self.allow_creation = allow_creation
        project_exists, project_path = self._check_project_exists(name)
        
        if project_dir:
            self.path = project_dir
        else:
            self.path = project_path
        
        # Checks if project exists, otherwise create a new one if allow_creation is True
        if not project_exists:
            if allow_creation:
                self._create_new_project()
            else:
                raise CouldNotFindProjectError(project_path)
        else:
            self.project_data = self._retrieve_project_data()
            self.type = self.project_data["type"]
            
            if verbose:
                logging.info(f"Project {Fore.CYAN}{name}{Fore.RESET} succesfully loaded.")
        
        # VITAL INFO LOADING ========================================/
        
        # Loads datasets info
        self.datasets = OdinDatasetsController(self)
        
        # Loads chronicles info
        self.chronicles = OdinChroniclesController(self)
        
        # Loads weights info
        self.models = OdinModelsController(self)
    
    def see(self):
        print(f"\n{Fore.CYAN}{self.name}{Fore.RESET}\n")
        print(f"Projet Type: {Fore.CYAN}{self.type}{Fore.RESET}")
        print(f"Project Path: {Fore.CYAN}{self.path}{Fore.RESET}")
        
        if len(self.datasets.datasets) == 0:
            print(f"Datasets: {Fore.YELLOW}Empty{Fore.RESET}")
        else:
            dataset_names = list(map(lambda dataset: f"{Fore.CYAN}{dataset.name}{Fore.RESET}", self.datasets.datasets))
            print(f"Datasets: {', '.join(dataset_names)}")
        
        if len(self.chronicles.chronicles) == 0:
            print(f"Chronicles: {Fore.YELLOW}Empty{Fore.RESET}")
        else:
            pass
        
        if len(self.models.staging_models) == 0:
            print(f"Staging Models: {Fore.YELLOW}Empty{Fore.RESET}")
        else:
            pass
        
        if len(self.models.staging_models) == 0:
            print(f"Published Models: {Fore.YELLOW}Empty{Fore.RESET}")
        else:
            pass
        
        print("")
        
    