
import json
import logging
import os
from colorama import Fore
import yaml
from odin_vision.api.internal.exceptions import CouldNotCreateProjectInPathError, ProjectNameInvalidError, ProjectTypeInvalidError
from odin_vision.constants import EXAMPLE_YAML, PROJECT_TYPES, README_CHRONICLES, README_WEIGHTS


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
    
    # (self.name, self.path, self.verbose, create_examples)
    def _create_new_project(self, create_examples: bool = False):
        self.project_data = self._project_builder(self.name, self.path, self.verbose, create_examples)
 