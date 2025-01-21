import os
from colorama import Fore
from matplotlib import pyplot as plt
from matcher import images_match

DEBUG = True
DEEP_DEBUG = False

MIN_MATCH_COUNT = 8


def clean_matches(dir1, dir2):
    dir1_files = []
    dir2_files = []

    for _, _, files in os.walk(dir1):
        dir1_files = files
        break

    for _, _, files in os.walk(dir2):
        dir2_files = files
        break

    for file in dir1_files:
        if DEBUG or DEEP_DEBUG:
            print(f"{Fore.CYAN}{file}{Fore.RESET} as header image.")
        for sub_file in dir2_files:
            if (
                sub_file != file
                and os.path.exists(
                    f"{dir2 if dir2.endswith('/') else f'{dir2}/'}{sub_file}"
                )
                and os.path.exists(
                    f"{dir1 if dir1.endswith('/') else f'{dir1}/'}{file}"
                )
            ):
                try:
                    if DEBUG or DEEP_DEBUG:
                        print(f"{Fore.YELLOW}{sub_file}{Fore.RESET} as compared image.")

                    img1_name = file
                    img2_name = sub_file

                    matches, matching_points, img = images_match(
                        f"{dir1 if dir1.endswith('/') else f'{dir1}/'}{img1_name}",
                        f"{dir2 if dir2.endswith('/') else f'{dir2}/'}{img2_name}",
                        min_match_count=MIN_MATCH_COUNT,
                    )

                    if DEEP_DEBUG:
                        plt.imshow(img, "gray")
                        plt.show()

                        print(matching_points)
                        print(matches)

                    if matches:
                        try:
                            if DEBUG or DEEP_DEBUG:
                                print(
                                    f"{Fore.RED}Removing {img2_name}{Fore.RESET} as compared to {img1_name}, matches SIFT and FLANN features..."
                                )
                            os.remove(
                                f"{dir2 if dir2.endswith('/') else f'{dir2}/'}{img2_name}"
                            )
                        except:
                            pass
                except Exception as e:
                    if DEBUG or DEEP_DEBUG:
                        print(
                            f"{Fore.BLUE}Passing by {sub_file}{Fore.RESET} as error has occured: {e}"
                        )


clean_matches("candidates/images", "candidates/images")
