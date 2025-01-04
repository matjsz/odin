import json
import logging
import os
from colorama import Fore

from odin_vision.api.dataset.classification import OdinDatasetClassification
from odin_vision.api.dataset.detection import OdinDatasetDetection
from odin_vision.api.model.base import OdinModelBase
from odin_vision.api.typings.dataset import OdinDataset
from odin_vision.chronicle.utils import get_chronicle_name


class OdinModelClassification(OdinModelBase):
    def train(self, dataset: str | OdinDataset | None=None, epochs: int=10, device: str="cpu", base_model: str | None=None, chronicle: str | None=None, subset: int=100, **kwargs):
        if not chronicle:
            chronicle = get_chronicle_name()
        
        if subset < 100:
            model_type = "naive"
        else:
            model_type = "wise"
            
        if not base_model:
            base_model = "yolo11n-cls.pt"
            
        if not dataset:
            return
        elif isinstance(dataset, (OdinDataset, OdinDatasetClassification, OdinDatasetDetection)):
            dataset = dataset.name
            
        self.dataset = dataset
        self.chronicle = chronicle
        self.status = "preparing-traning"
            
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
        
        self.status = "training"
        
        with open(chronicle_path+"\\chronicle.json", "w", encoding="utf8") as wf:
            wf.write(json.dumps(chronicle_info, indent=4, ensure_ascii=False))
            
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
        
        if model_type == "naive":
            yolo_command.append(f"fraction={subset/100}")
        
        os.system(
            " ".join(yolo_command)
        )
        
        self._try_create_model_snapshot(model_type, chronicle_path)
        
        logging.info(f"Trained to chronicle {Fore.CYAN}{chronicle}{Fore.RESET}")
        logging.info(f"You can test this version by using the command {Fore.CYAN}odin test --chronicle {chronicle}{Fore.RESET}")