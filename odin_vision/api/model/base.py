import datetime
import json
import logging
import os
import shutil

from colorama import Fore


class OdinModelBase:
    def __init__(self, name: str | None=None, chronicle: str | None=None, dataset: str | None=None, project=None, model_path: str | None=None, status: str | None=None, **kwargs):
        self.name = name
        self.type: str | None = None
        self.chronicle = chronicle
        self.dataset = dataset
        self.project = project
        self.path = model_path
        self.status = status
        
    def _get_chronicle_data(self, chronicle_path):
        with open(chronicle_path+"\\chronicle.json", "r", encoding="utf8") as f:
            data = json.loads(f.read())
        return data
    
    def _try_create_folder(self, folder_path):
        try:
            os.makedirs(folder_path)
        except FileExistsError:
            pass
        except Exception as e:
            if self.project.verbose:
                logging.info(f"Something went wrong while trying to create {Fore.CYAN}{folder_path}{Fore.RESET}: {e}")
    
    def _try_create_model_snapshot(self, type, chronicle_path):
        now = datetime.datetime.now()
        
        logging.info("Saving results...")
        
        if type == "naive":
            try:
                shutil.copyfile(f"{chronicle_path}\\weights\\best.pt", f"{chronicle_path}\\weights\\pre_{now.year}_{now.month}_{now.day}_{now.hour}{now.minute}{now.second}.pt")
                
                self.name = f"pre_{now.year}_{now.month}_{now.day}_{now.hour}{now.minute}{now.second}.pt"
                self.type = type
                self.status = "staging"

                logging.info(f"Model saved at: {Fore.CYAN}{chronicle_path}\\weights\\pre.{now.year}.{now.month}.{now.day}.{now.hour}{now.minute}{now.second}.pt{Fore.RESET}")
            except Exception as e:
                logging.info(f"Couldn't create model snapshot, something went wrong: {e}")
        elif type == "wise":
            try:
                shutil.copyfile(f"{chronicle_path}\\weights\\best.pt", f"{chronicle_path}\\weights\\{now.year}_{now.month}_{now.day}_{now.hour}{now.minute}{now.second}.pt")
                
                self.name = f"{now.year}_{now.month}_{now.day}_{now.hour}{now.minute}{now.second}.pt"
                self.type = type
                self.status = "staging"

                logging.info(f"Model saved at: {Fore.CYAN}{chronicle_path}\\weights\\{now.year}_{now.month}_{now.day}_{now.hour}{now.minute}{now.second}.pt{Fore.RESET}")
            except Exception as e:
                logging.info(f"Couldn't create model snapshot, something went wrong: {e}")
        
    def train(self):
        raise NotImplementedError()
        
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
                
            self.status = "published"

            logging.info(f"Succesfully published model {Fore.CYAN}{self.name}{Fore.RESET} as {Fore.CYAN}{new_model_name}{Fore.RESET}...")
