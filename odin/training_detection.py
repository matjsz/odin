
import datetime
import logging
import os
import shutil
import subprocess
import time
import uuid

import click
from colorama import Fore
from chronicle_utils import get_chronicle_name
from training import BaseTrainingCommands


class DetectionTrainingCommands(BaseTrainingCommands):
    def train(self, epochs: int, device: str, base_model: str, chronicle_name: str, subset: int, *kwargs):
        # training_images_path = f"{self.dataset_path}\\train\\images"
        # training_labels_path = f"{self.dataset_path}\\train\\labels"
        
        if subset < 100:
            model_type = "naive"
        else:
            model_type = "wise"
        
        if not chronicle_name:
            chronicle_name = click.prompt(f"What will be the name of the {Fore.CYAN}chronicle{Fore.RESET}? If left empty, will generate automatically.", confirmation_prompt=True, default=None)
        
        if not chronicle_name:
            chronicle_name = get_chronicle_name()
        
        chronicles_path = f"{os.path.abspath('.')}\\chronicles" 
        chronicle_path = f"{os.path.abspath('.')}\\chronicles\\{chronicle_name}"
        
        logging.info(f"Starting {Fore.CYAN}training{Fore.RESET}...")
        logging.info(f"Version defined as {Fore.CYAN}{model_type}{Fore.RESET}")
        
        self._try_create_folder(chronicle_path)
        self._try_create_folder(chronicle_path+"\\weights")
        
        yolo_command = [
            "yolo",
            "detect",
            "train",
            f"data={self.dataset_path}\\data.yaml",
            f"epochs={epochs}",
            f"batch={'-1' if device == 'gpu' else '2'}",
            f"model={base_model}",
            "amp=false",
            "patience=10",
            "save_period=5",
            f"device={'0' if device == 'gpu' else 'cpu'}",
            f"project={chronicles_path}",
            f"name={chronicle_name}",
            "exist_ok=true",
            "plots=true",
        ]
        
        if model_type == "naive":
            yolo_command.append(f"fraction={subset/100}")
        
        os.system(
            " ".join(yolo_command)
        )
        
        self._try_create_model_snapshot(model_type, chronicle_path)
        
        logging.info(f"Trained to chronicle {Fore.CYAN}{chronicle_name}{Fore.RESET}")
        logging.info(f"You can test this version by using the command {Fore.CYAN}odin test {chronicle_name}{Fore.RESET}")