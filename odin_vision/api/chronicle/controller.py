
import os
from odin_vision.api.internal.exceptions import CouldNotFindChronicleByNameError
from odin_vision.api.typings.chronicle import OdinChronicle


class OdinChroniclesController:
    def __init__(self, project, chronicles: list[OdinChronicle] = []):
        self.project = project
        self.chronicles = chronicles
        
        if len(chronicles) == 0:
            self._get_chronicles()
            
    def _get_chronicles(self):
        for _, chronicle_folders, _ in os.walk(f"{self.project.path}\\chronicles"):
            for chronicle_folder in chronicle_folders:
                chronicle_path = os.path.abspath(f"{self.project.path}\\chronicles\\{chronicle_folder}")
                
                chronicle_obj = OdinChronicle(chronicle_folder, self.project, chronicle_path)
                self.chronicles.append(chronicle_obj)
            break
        
    def get(self, by_name: str | None=None):
        if by_name:
            for chronicle in self.chronicles:
                if chronicle.name == by_name:
                    return chronicle
            else:
                raise CouldNotFindChronicleByNameError(by_name)