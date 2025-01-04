
import json
import logging
import os
from colorama import Fore
from odin_vision.api.internal.exceptions import ModelNotInformedError
from odin_vision.api.model.classification import OdinModelClassification
from odin_vision.api.model.detection import OdinModelDetection
from odin_vision.api.typings.chronicle import OdinChronicle
from odin_vision.api.typings.dataset import OdinDataset
from odin_vision.api.typings.model import OdinModel

class OdinModelsController:
    def __init__(self, project, staging_models: list[OdinModel]=[], published_models: list[OdinModel] = []):
        self.project = project
        self.staging_models = staging_models
        self.published_models = published_models
        
        model_builders = {
            "classification": OdinModelClassification,
            "detection": OdinModelDetection
        }
        
        self._model_builder = model_builders[project.type]
        
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

                            staging_model_obj = self._model_builder(file, chronicle_folder, model_data["dataset"], self.project, staging_model_path, "staging")
                            
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

                            staging_model_obj = self._model_builder(file, chronicle_folder, model_data["dataset"], self.project, staging_model_path, "staging")
                            
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
                    published_model_obj = self._model_builder(file, chronicle_name, dataset_name, self.project, published_model_path, "published")

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
                    published_model_obj = self._model_builder(file, chronicle_name, dataset_name, self.project, published_model_path, "published")

                    return published_model_obj
            break

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

    def train(self, dataset: str | OdinDataset | None=None, epochs: int=10, device: str="cpu", base_model: str | None=None, chronicle: str | None=None, subset: int=100, test_after_training: bool = False, publish_after_testing: bool = False, publish_after_training: bool = False, **kwargs) -> OdinModel: 
        temp_model_instance = self._model_builder(project=self.project)
        temp_model_instance.train(dataset, epochs, device, base_model, chronicle, subset)
        
        if test_after_training:
            temp_model_instance.test(publish_if_ok=publish_after_testing)
        
        if publish_after_training:
            temp_model_instance.publish()
            
        return temp_model_instance

    def publish(self, model: str | OdinModel | None=None):
        if model and isinstance(model, str):
            model_target = self.get(by_name=model)
            model_target.publish()
        elif model and isinstance(model, OdinModel):
            model.publish()
        else:
            raise ModelNotInformedError()
        
    def test(self, model: str | OdinModel | None=None,  publish_after_testing: bool = False):
        if model and isinstance(model, str):
            model_target = self.get(by_name=model)
            model_target.test(publish_if_ok=publish_after_testing)
        elif model and isinstance(model, OdinModel):
            model.test(publish_if_ok=publish_after_testing)
        else:
            raise ModelNotInformedError()