
import datetime
import json
import logging
import os
import shutil
import time

import click
from colorama import Fore
from ultralytics import YOLO
import ultralytics.models
from chronicle_utils import get_chronicle_name
from training import BaseTrainingCommands


class DetectionTrainingCommands(BaseTrainingCommands):
    def train(self, epochs: int=0, device: str="", base_model: str="", dataset_name: str="", chronicle_name: str="", subset: int=0, **kwargs):
        if subset < 100:
            model_type = "naive"
        else:
            model_type = "wise"
            
        if not dataset_name:
            confirmed_name = False
            
            while not confirmed_name:
                dataset_name = click.prompt(
                    f"What is the dataset's name?"
                )
                
                if click.confirm(f'{Fore.CYAN}{dataset_name}{Fore.RESET}, is this name right?'):
                    confirmed_name = True
        
        if not chronicle_name:
            chronicle_name = click.prompt(f"What will be the name of the {Fore.CYAN}chronicle{Fore.RESET}? If left empty, will create at", default=get_chronicle_name())
        
        chronicles_path = f"{os.path.abspath('.')}\\chronicles" 
        chronicle_path = f"{os.path.abspath('.')}\\chronicles\\{chronicle_name}"
        
        dataset_path = f"{os.path.abspath('.')}\\datasets\\{dataset_name}"
        
        logging.info(f"Starting {Fore.CYAN}training{Fore.RESET}...")
        logging.info(f"Version defined as {Fore.CYAN}{model_type}{Fore.RESET}")
        
        
        self._try_create_folder(chronicle_path)
        self._try_create_folder(chronicle_path+"\\weights")
        
        chronicle_info = {
            "name": chronicle_name,
            "dataset": dataset_name
        }
        
        with open(chronicle_path+"\\chronicle.json", "w", encoding="utf8") as wf:
            wf.write(json.dumps(chronicle_info, indent=4, ensure_ascii=False))
        
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
        logging.info(f"You can test this version by using the command {Fore.CYAN}odin test --chronicle {chronicle_name}{Fore.RESET}")
        
    def test(self, chronicle_name: str="", **kwargs):
        if not chronicle_name:
            confirmed_name = False
            
            while not confirmed_name:
                chronicle_name = click.prompt(
                    f"What is the chronicle's name?"
                )
                
                if click.confirm(f'{Fore.CYAN}{chronicle_name}{Fore.RESET}, is this name right?'):
                    confirmed_name = True
        
        chronicle_data = self._get_chronicle_data(f"{os.path.abspath('.')}\\chronicles\\{chronicle_name}")
        dataset_name = chronicle_data["dataset"]
        
        val_images_path = f"{os.path.abspath('.')}\\datasets\\{dataset_name}\\val\\images"
        
        val_images = []
        
        chronicle_models_path = f"{os.path.abspath('.')}\\chronicles\\{chronicle_name}\\weights"
        eligible_models = []
        
        for _, _, files in os.walk(val_images_path):
            for file in files:
                image_path = f"{os.path.abspath('.')}\\datasets\\{dataset_name}\\val\\images\\{file}"
                val_images.append(image_path)
                
        for _, _, files in os.walk(chronicle_models_path):
            for file in files:
                if self._check_if_eligible_model(file):
                    eligible_models.append(file)
            
        eligible_models.sort(reverse=True)
            
        model_selection = {}
        model_id = 0
        model_id_options = []
        for model in eligible_models:
            model_selection[str(model_id)] = model
            model_id_options.append(str(model_id))
            model_id += 1
        
        print("")
        for model_selection_id in model_selection:
            print(f"{Fore.CYAN}{model_selection_id}{Fore.RESET} - {model_selection[model_selection_id]}")
        print("")
        
        model_selected = model_selection[click.prompt("Select a model to test", default="0", show_choices=True, type=click.Choice(model_id_options))]
        print("")
        
        logging.info(f"Loading model {Fore.CYAN}{model_selected}{Fore.RESET}...")
        
        model: YOLO = YOLO(f"{os.path.abspath('.')}\\chronicles\\{chronicle_name}\\weights\\{model_selected}", verbose=False)
                
        logging.info(f"Successfully loaded model {Fore.CYAN}{model_selected}{Fore.RESET}.")
                
        results = model.predict(val_images, verbose=False)
        odin_results = self._gather_yolo_results(results)
            
        for result in odin_results:
            logging.info(f"Speed: {result.inference_speed}")
            result.show()
            
    def publish(self, project_name: str="", chronicle_name: str="", **kwargs):
        if not chronicle_name:
            confirmed_name = False
            
            while not confirmed_name:
                chronicle_name = click.prompt(
                    f"What is the chronicle's name?"
                )
                
                if click.confirm(f'{Fore.CYAN}{chronicle_name}{Fore.RESET}, is this name right?'):
                    confirmed_name = True
        
        chronicle_data = self._get_chronicle_data(f"{os.path.abspath('.')}\\chronicles\\{chronicle_name}")
        
        chronicle_models_path = f"{os.path.abspath('.')}\\chronicles\\{chronicle_name}\\weights"
        eligible_models = []
        
        for _, _, files in os.walk(chronicle_models_path):
            for file in files:
                if self._check_if_eligible_model(file):
                    eligible_models.append(file)
            
        eligible_models.sort(reverse=True)
            
        model_selection = {}
        model_id = 0
        model_id_options = []
        for model in eligible_models:
            model_selection[str(model_id)] = model
            model_id_options.append(str(model_id))
            model_id += 1
        
        print("")
        for model_selection_id in model_selection:
            print(f"{Fore.CYAN}{model_selection_id}{Fore.RESET} - {model_selection[model_selection_id]}")
        print("")
        
        model_selected = model_selection[click.prompt("Select a model to test", default="0", show_choices=True, type=click.Choice(model_id_options))]
        print("")
        
        logging.info(f"Publishing model {Fore.CYAN}{model_selected}{Fore.RESET}...")
        
        now = datetime.datetime.now()
        new_model_name = f"{project_name}_{now.year}_{now.month}_{now.day}_{chronicle_name}.pt"
        
        shutil.copy(f"{os.path.abspath('.')}\\chronicles\\{chronicle_name}\\weights\\{model_selected}", f"{os.path.abspath('.')}\\weights\\{new_model_name}")
        
        logging.info(f"Succesfully published model {Fore.CYAN}{model_selected}{Fore.RESET} as {Fore.CYAN}{new_model_name}{Fore.RESET}...")