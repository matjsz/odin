import os
from colorama import Fore
from matplotlib import pyplot as plt
from matcher import images_match

DEBUG = True
DEEP_DEBUG = False

MIN_MATCH_COUNT = 8

for _, _, files in os.walk("train/originals/images"):
    for file in files:
        if DEBUG or DEEP_DEBUG:
            print(f"{Fore.CYAN}{file}{Fore.RESET} as header image.")
        for sub_file in files:
            if (
                sub_file != file
                and os.path.exists(f"train/originals/images/{sub_file}")
                and os.path.exists(f"train/originals/images/{file}")
            ):
                try:
                    if DEBUG or DEEP_DEBUG:
                        print(f"{Fore.YELLOW}{sub_file}{Fore.RESET} as compared image.")

                    img1_name = file
                    img2_name = sub_file

                    matches, matching_points, img = images_match(
                        f"train/originals/images/{img1_name}",
                        f"train/originals/images/{img2_name}",
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
                            os.remove(f"train/originals/images/{img2_name}")
                        except:
                            pass
                except Exception as e:
                    if DEBUG or DEEP_DEBUG:
                        print(
                            f"{Fore.BLUE}Passing by {sub_file}{Fore.RESET} as error has occured: {e}"
                        )
    break
