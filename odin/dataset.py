import json
import logging
import os
import shutil

import click
from colorama import Fore


class BaseDatasetCommands:
    def __init__(self, type, dataset_name):
        self.type = type
        self.dataset_name = dataset_name
        self.dataset_path = f"{os.path.abspath('.')}\\datasets\\{dataset_name}"

    def _try_create_folder(self, folder_path):
        try:
            os.makedirs(folder_path)
        except FileExistsError:
            pass
        except Exception as e:
            logging.info(f"Something went wrong while trying to create {Fore.CYAN}{folder_path}{Fore.RESET}: {e}")

    def _upgrade_version(self, base_version, update_size):
        version = list(map(lambda x: int(x), base_version.split(".")))

        if update_size == "major":
            version[0] += 1
        elif update_size == "minor":
            version[1] += 1
        elif update_size == "fix":
            version[2] += 1

        upgraded_version = f"{version[0]}.{version[1]}.{version[2]}"

        return upgraded_version

    def _create_dataset_metadata_files(self, **kwargs):
        raise NotImplementedError(
            "This feature is not implemented or is going through maintenance."
        )

    def _publish_data(self, update_type="", train="", val="", **kwargs):
        raise NotImplementedError(
            "This feature is not implemented or is going through maintenance."
        )

    def _status_sum_staging(self, **kwargs):
        raise NotImplementedError(
            "This feature is not implemented or is going through maintenance."
        )

    def _status_sum_train(self, **kwargs):
        raise NotImplementedError(
            "This feature is not implemented or is going through maintenance."
        )

    def _status_show_additional_info(self, **kwargs):
        raise NotImplementedError(
            "This feature is not implemented or is going through maintenance."
        )

    def _augmentate_data(self, augs=0, **kwargs):
        raise NotImplementedError(
            "This feature is not implemented or is going through maintenance."
        )

    def _rollback_dataset(
        self, snapshot_info={}, rollback_version="", dataset_folder="", class_folder="", **kwargs
    ):
        raise NotImplementedError

    def create(self, **kwargs):
        logging.info(
            f"Creating dataset '{Fore.BLUE}{self.dataset_name}{Fore.RESET}'..."
        )

        if not os.path.exists(self.dataset_path):
            logging.info("Creating final folders...")

            self._create_dataset_metadata_files()
        else:
            logging.info(
                f"This dataset already exists, if you wish to recreate it or simply delete it, use the command {Fore.CYAN}odin dataset delete {self.dataset_name}{Fore.RESET}. This will delete the entire dataset, be careful."
            )

    def publish(self, train=0, val=0, **kwargs):
        logging.info(
            f"Publishing dataset '{Fore.BLUE}{self.dataset_name}{Fore.RESET}'..."
        )

        if not os.path.exists(self.dataset_path):
            logging.info(
                f"The dataset mentioned doesn't exist, create it by using the command {Fore.CYAN}odin dataset create {self.dataset_name}{Fore.RESET}."
            )
        else:
            update_type = click.prompt(
                f"Is this update to the dataset a {Fore.CYAN}fix{Fore.RESET}, {Fore.CYAN}minor{Fore.RESET} or {Fore.CYAN}major{Fore.RESET} update?",
                show_choices=True,
                type=click.Choice(["fix", "minor", "major"]),
            )

            self._publish_data(update_type=update_type, train=train, val=val)
            logging.info(f"Succesfully published {Fore.GREEN}train{Fore.RESET} data")

    def status(self, **kwargs):
        if not os.path.exists(self.dataset_path):
            logging.info(
                f"The dataset mentioned doesn't exist, create it by using the command {Fore.CYAN}odin dataset create {self.dataset_name}{Fore.RESET}."
            )
        else:
            status = {
                "empty": f"{Fore.YELLOW}Empty{Fore.RESET}",
                "staging": f"{Fore.YELLOW}Staging{Fore.RESET}",
                "published": f"{Fore.GREEN}Published{Fore.RESET}",
                "published_staging": f"{Fore.CYAN}Published and Staging{Fore.RESET}",
            }

            staging = self._status_sum_staging()
            train = self._status_sum_train()

            if staging > 0 and train > 0:
                logging.info(
                    f"Retrieving {Fore.CYAN}{self.dataset_name}{Fore.RESET} status..."
                )
                logging.info(f"Status: {status['published_staging']}")
            elif staging > 0 and train == 0:
                logging.info(
                    f"Retrieving {Fore.CYAN}{self.dataset_name}{Fore.RESET} status..."
                )
                logging.info(f"Status: {status['staging']}")
            elif train > 0 and staging == 0:
                logging.info(
                    f"Retrieving {Fore.CYAN}{self.dataset_name}{Fore.RESET} status..."
                )
                logging.info(f"Status: {status['published']}")
            else:
                logging.info(
                    f"Retrieving {Fore.CYAN}{self.dataset_name}{Fore.RESET} status..."
                )
                logging.info(f"Status: {status['empty']}")

            self._status_show_additional_info()

    def augmentate(self, augmentation_amount=0, **kwargs):
        if not os.path.exists(self.dataset_path):
            logging.info(
                f"The dataset mentioned doesn't exist, create it by using the command {Fore.CYAN}odin dataset create {self.dataset_name}{Fore.RESET}."
            )
        else:
            self._augmentate_data(augmentation_amount=augmentation_amount)

    def rollback(self, rollver="", **kwargs):
        if not os.path.exists(self.dataset_path):
            logging.info(
                f"The dataset mentioned doesn't exist, create it by using the command {Fore.CYAN}odin dataset create {self.dataset_name}{Fore.RESET}."
            )
        else:
            dataset_info = json.loads(
                open(f"{self.dataset_path}\\dataset.json", "r", encoding="utf8").read()
            )
            snapshot_info = json.loads(
                open(f"{self.dataset_path}\\snapshot.json", "r", encoding="utf8").read()
            )
            dataset_versions = []

            for version in snapshot_info:
                if version != dataset_info["version"]:
                    dataset_versions.append(version)

            if len(dataset_versions) > 0:
                if not rollver:
                    rollback_version = click.prompt(
                        f"Which version you want to rollback to?",
                        show_choices=True,
                        type=click.Choice(dataset_versions),
                    )

                    print(rollback_version)
                else:
                    if rollver.lower().replace("v", "") in dataset_versions:
                        rollback_version = rollver
                    else:
                        logging.info(
                            f"The version {Fore.LIGHTRED_EX}{rollver}{Fore.RESET} doesn't exist."
                        )
                        return

                # Add binaries from rollback version to actual version
                rollback_format = click.prompt(
                    f"Will the rollback replace the current version with the old one or create a new major version? (replace, create)",
                    show_choices=True,
                    type=click.Choice(["replace", "create"]),
                )

                if click.confirm(
                    "Do you want to continue? This action with replace all of your current files to a older version, the files might be deleted as well, this is not a merge."
                ):
                    if rollback_format == "replace":
                        final_version = rollback_version
                    else:
                        final_version = self._upgrade_version(
                            base_version=dataset_info["version"], update_size="major"
                        )

                    for dataset_folder in snapshot_info[rollback_version]:
                        for sub_folder in snapshot_info[rollback_version][
                            dataset_folder
                        ]:
                            self._rollback_dataset(
                                snapshot_info=snapshot_info,
                                rollback_version=rollback_version,
                                dataset_folder=dataset_folder,
                                class_folder=sub_folder,
                            )

                    dataset_info["version"] = final_version

                    if rollback_format == "create":
                        snapshot_info[final_version] = snapshot_info[rollback_version]

                    with open(
                        f"{self.dataset_path}\\dataset.json", "w", encoding="utf8"
                    ) as wf:
                        wf.write(json.dumps(dataset_info))

                    with open(
                        f"{self.dataset_path}\\snapshot.json", "w", encoding="utf8"
                    ) as wf:
                        wf.write(json.dumps(snapshot_info))

                    logging.info(
                        f"Succesfuly executed rollback to version {Fore.CYAN}{rollback_version}{Fore.RESET}, now at version {Fore.CYAN}{final_version}{Fore.RESET}"
                    )
            else:
                logging.info(
                    "There are no versions to rollback to, this dataset only have one snapshoted version, which is the current one."
                )

    def delete(self, **kwargs):
        dataset_path = f"{os.path.abspath('.')}\\datasets\\{self.dataset_name}"

        if not os.path.exists(dataset_path):
            logging.info(
                f"The dataset mentioned doesn't exist, create it by using the command {Fore.CYAN}odin dataset create {self.dataset_name}{Fore.RESET}."
            )
        else:
            if click.confirm(
                f"Do you want to continue? All your data will be lost (and you will bring {Fore.RED}The Ragnarok{Fore.RESET} to thy dataset!)"
            ):
                logging.info(
                    f"Aplying {Fore.RED}Ragnarok{Fore.RESET} to {Fore.CYAN}{self.dataset_name}{Fore.RESET}!"
                )

                try:
                    shutil.rmtree(f"{dataset_path}")

                    logging.info(
                        f"Successfully deleted {Fore.CYAN}{self.dataset_name}{Fore.RESET}."
                    )
                except:
                    logging.info(
                        f"{Fore.CYAN}Odin{Fore.RESET} was unable to delete {Fore.CYAN}{self.dataset_name}{Fore.RESET}."
                    )

    def yaml(self, **kwargs):
        raise NotImplementedError