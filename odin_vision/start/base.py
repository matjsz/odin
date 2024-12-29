import logging
import os

from colorama import Fore
import yaml

from odin_vision.constants import EXAMPLE_YAML, README_CHRONICLES, README_WEIGHTS


class BaseStartCommand:
    def __init__(self, project_type, project_name):
        self.project_name = project_name
        self.project_type = project_type
        
    def _get_project_pretty_type(self):
        pretty_types = {
            "classification": "Image Classification",
            "detection": "Object Detection"
        }
        
        return pretty_types[self.project_type]
    
    def create_datasets_structure(self):
        project_general_info = {"name": self.project_name, "type": self.project_type, "version": "0.1.0"}

        logging.info(
            f"Creating project structure for {Fore.CYAN}{self.project_name}{Fore.RESET}, a {Fore.CYAN}{self._get_project_pretty_type()}{Fore.RESET} project."
        )

        try:
            data = open("project.yaml", "r", encoding="utf8").read()
            if len(data) > 0:
                logging.info(
                    f"There is already a project created in this location. Use {Fore.CYAN}odin wrath{Fore.RESET} to delete the project, then run the command again."
                )
                return
        except:
            pass

        with open("project.yaml", "w", encoding="utf8") as wf:
            wf.write(yaml.dump(project_general_info))

        # Datasets
        dataset_parent = f"{os.path.abspath('.')}\\datasets"
        if not os.path.exists(dataset_parent):
            os.makedirs(dataset_parent)

            logging.info(f"Succesfully created {Fore.CYAN}datasets{Fore.RESET}")

        logging.info(f"Creating {Fore.CYAN}datasets examples{Fore.RESET}...")
    
    def create_models_structure(self):
        # Chronicles
        # chronicles/chronicle-nick/run-uuid
        chronicles_parent = f"{os.path.abspath('.')}\\chronicles"
        if not os.path.exists(chronicles_parent):
            os.makedirs(chronicles_parent)

            with open(f"{chronicles_parent}\\README.md", "w", encoding="utf8") as wf:
                wf.write(README_CHRONICLES)

            logging.info(f"Succesfully created {Fore.CYAN}chronicles{Fore.RESET}")

        # Weights
        weights_parent = f"{os.path.abspath('.')}\\weights"
        if not os.path.exists(weights_parent):
            os.makedirs(weights_parent)

            with open(f"{weights_parent}\\README.md", "w", encoding="utf8") as wf:
                wf.write(README_WEIGHTS)

            logging.info(f"Succesfully created {Fore.CYAN}weights{Fore.RESET}")
    
    def classification(self):
        dataset_example_classif = f"{os.path.abspath('.')}\\datasets\\dataset_1_classif"
        if not os.path.exists(dataset_example_classif):
            os.makedirs(dataset_example_classif)

            os.makedirs(f"{os.path.abspath('.')}\\datasets\\dataset_1_classif\\class_1")
            os.makedirs(f"{os.path.abspath('.')}\\datasets\\dataset_1_classif\\class_2")

            logging.info(
                f"Succesfully created dataset example at: {Fore.CYAN}datasets/dataset_1_classif{Fore.RESET}"
            )
            
    def detection(self):
        dataset_example_obd = f"{os.path.abspath('.')}\\datasets\\dataset_1_obd"
        if not os.path.exists(dataset_example_obd):
            os.makedirs(dataset_example_obd)

            os.makedirs(f"{os.path.abspath('.')}\\datasets\\dataset_1_obd\\train")
            os.makedirs(f"{os.path.abspath('.')}\\datasets\\dataset_1_obd\\val")
            os.makedirs(
                f"{os.path.abspath('.')}\\datasets\\dataset_1_obd\\train\\images"
            )
            os.makedirs(f"{os.path.abspath('.')}\\datasets\\dataset_1_obd\\val\\images")
            os.makedirs(
                f"{os.path.abspath('.')}\\datasets\\dataset_1_obd\\train\\labels"
            )
            os.makedirs(f"{os.path.abspath('.')}\\datasets\\dataset_1_obd\\val\\labels")

            with open(
                f"{os.path.abspath('.')}\\datasets\\dataset_1_obd\\data.yaml", "w"
            ) as wf:
                wf.write(EXAMPLE_YAML)

            logging.info(
                f"Succesfully created dataset example at: {Fore.CYAN}datasets/dataset_1_obd{Fore.RESET}"
            )