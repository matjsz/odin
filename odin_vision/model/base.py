
import datetime
import json
import logging
import os
import shutil

from colorama import Fore
from odin_vision.model.results import YOLOResult


class BaseModelCommands:
    def __init__(self, type, dataset_name):
        self.type = type
        self.dataset_name = dataset_name
        self.dataset_path = f"{os.path.abspath('.')}\\datasets\\{dataset_name}"
        
    def _get_chronicle_data(self, chronicle_path):
        with open(chronicle_path+"\\chronicle.json", "r", encoding="utf8") as f:
            data = json.loads(f.read())
        return data
        
    def _gather_yolo_results(self, results) -> list[YOLOResult]:
        odin_results = []
        
        for raw_result in results:
            result = YOLOResult(
                image_with_bboxes=raw_result.plot(),
                inference_speed=raw_result.speed
            )
            odin_results.append(result)
        
        return odin_results
        
    def _check_if_eligible_model(self, model_file_name):
        if model_file_name.startswith("epoch") or model_file_name in ["best.pt", "last.pt"]:
            return False
        return True
        
    def _try_create_folder(self, folder_path):
        try:
            os.makedirs(folder_path)
        except FileExistsError:
            pass
        except Exception as e:
            logging.info(f"Something went wrong while trying to create {Fore.CYAN}{folder_path}{Fore.RESET}: {e}")
        
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
        
    def _create_chronicle(self):
        raise NotImplementedError
        
    def train(self, epochs: int, *kwargs):
        raise NotImplementedError