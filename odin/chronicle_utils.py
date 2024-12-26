import random


def get_chronicle_name():
    firsts = [
        "super",
        "extra",
        "grand",
        "absolute",
        "joyful",
        "amazing",
        "relentless",
        "bold",
        "brave",
        "clever",
        "cool",
        "daring",
        "dreamy",
        "elegant",
        "epic",
        "fearless",
        "glorious",
        "heroic",
        "noble",
        "swift",
        "vibrant",
        "thunderous",
        "watching",
        "obversing",
    ]

    lasts = [
        "panda",
        "tiger",
        "hero",
        "challenger",
        "captain",
        "dragon",
        "eagle",
        "falcon",
        "fighter",
        "warrior",
        "knight",
        "leopard",
        "lion",
        "ninja",
        "runner",
        "ranger",
        "shadow",
        "sprinter",
        "observer",
        "watcher",
        "voyager",
        "starman",
        "wolf",
        "viking",
    ]

    chosen_first = random.randint(0, len(firsts) - 1)
    chosen_last = random.randint(0, len(lasts) - 1)

    return f"{firsts[chosen_first]}-{lasts[chosen_last]}"
