
import json
import os
from typing import Literal
from odin_vision.api.dataset.classification import OdinDatasetClassification
from odin_vision.api.dataset.detection import OdinDatasetDetection
from odin_vision.api.internal.exceptions import CouldNotCreateDatasetInPathError, CouldNotFindDatasetByNameError, DatasetNotInformedError
from odin_vision.api.typings.dataset import OdinDataset


class OdinDatasetsController:
    def __init__(self, project, datasets: list[OdinDataset] = []):
        self.project = project
        self.datasets = datasets
        
        dataset_builders = {
            "detection": OdinDatasetDetection,
            "classification": OdinDatasetClassification
        }
        
        self._dataset_builder = dataset_builders[self.project.type]
        
        if len(datasets) == 0:
            self._get_datasets()
        
    def _get_datasets(self):
        for _, dataset_folders, _ in os.walk(f"{self.project.path}\\datasets"):
            for dataset_folder in dataset_folders:
                dataset_path = os.path.abspath(f"{self.project.path}\\datasets\\{dataset_folder}")
                dataset_obj = self._dataset_builder(dataset_folder, self.project, dataset_path)
                self.datasets.append(dataset_obj)
            break
    
    def get(self, by_name: str | None = None, by_chronicle: str | None = None, allow_creation: bool = False) -> OdinDataset:
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
        
        new_dataset = self._dataset_builder(name, self.project, dataset_path)
        
        self.datasets.append(new_dataset)
        
        return new_dataset

    def publish(self, dataset: str | OdinDataset | None = None, update_type: Literal["major", "minor", "fix"] = "major"):
        if dataset and isinstance(dataset, str):
            dataset_target = self.get(by_name=dataset)
            dataset_target.publish(update_type=update_type)
        elif dataset and isinstance(dataset, OdinDataset):
            dataset.publish(update_type=update_type)
        else:
            raise DatasetNotInformedError()
