
import json
import logging
import os

from colorama import Fore
import yaml
from odin_vision.api.internal.exceptions import CouldNotCreateProjectInPathError
from odin_vision.constants import README_CHRONICLES, README_WEIGHTS

def new_classification_project(project_name, project_path, verbose: bool = False, create_examples: bool = True):
    project_root = project_path
    if project_path:
        project_root = project_path.replace('/', '\\')
    
    try:
        if not os.path.exists(project_root):
            os.makedirs(project_root)
    except:
        raise CouldNotCreateProjectInPathError(project_root)

    project_general_info = {"name": project_name, "type": "classification", "version": "0.1.0"}

    if verbose:
        logging.info(
            f"Creating project structure for {Fore.CYAN}{project_name}{Fore.RESET}, a {Fore.CYAN}Image Classification{Fore.RESET} project."
        )

    with open(f"{project_root}\\project.yaml", "w", encoding="utf8") as wf:
        wf.write(yaml.dump(project_general_info))

    # Datasets
    dataset_parent = f"{project_root}\\datasets"
    if not os.path.exists(dataset_parent):
        os.makedirs(dataset_parent)

        if verbose:
            logging.info(f"Succesfully created {Fore.CYAN}datasets{Fore.RESET}")

    if create_examples:
        if verbose:
            logging.info(f"Creating {Fore.CYAN}datasets examples{Fore.RESET}...")
            
        dataset_example_classif = f"{project_path}\\datasets\\dataset_1_classif"
        if not os.path.exists(dataset_example_classif):
            os.makedirs(dataset_example_classif)
            
            dataset_info = {
                "type": "classification",
                "version": "0.1.0"
            }

            with open(f"{project_path}\\datasets\\dataset_1_classif\\dataset.json", "w", encoding="utf8") as wf:
                wf.write(json.dumps(dataset_info))

            with open(f"{project_path}\\datasets\\dataset_1_classif\\snapshot.json", "w", encoding="utf8") as wf:
                wf.write(json.dumps({}))

            os.makedirs(f"{project_path}\\datasets\\dataset_1_classif\\class_1")
            os.makedirs(f"{project_path}\\datasets\\dataset_1_classif\\class_2")

            logging.info(
                f"Succesfully created dataset example at: {Fore.CYAN}datasets/dataset_1_classif{Fore.RESET}"
            )
            
    # Chronicles
    chronicles_parent = f"{project_path}\\chronicles"
    if not os.path.exists(chronicles_parent):
        os.makedirs(chronicles_parent)

        with open(f"{chronicles_parent}\\README.md", "w", encoding="utf8") as wf:
            wf.write(README_CHRONICLES)

        if verbose:
            logging.info(f"Succesfully created {Fore.CYAN}chronicles{Fore.RESET}")

    # Weights
    weights_parent = f"{project_path}\\weights"
    if not os.path.exists(weights_parent):
        os.makedirs(weights_parent)

        with open(f"{weights_parent}\\README.md", "w", encoding="utf8") as wf:
            wf.write(README_WEIGHTS)

        if verbose:
            logging.info(f"Succesfully created {Fore.CYAN}weights{Fore.RESET}")

    return project_general_info