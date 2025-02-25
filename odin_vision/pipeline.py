import logging
import random
import plotly.graph_objects as go

from typing import Any, Callable
from colorama import Fore

logging.basicConfig(
    level=logging.INFO,
    format=f"[{Fore.CYAN}ODIN{Fore.RESET}][{Fore.YELLOW}%(asctime)s{Fore.RESET}] %(message)s",
)


class PipelineResult:
    passed: bool
    fallback_exception: None | str
    final_exception: None | str

    def __init__(
        self,
        passed: bool,
        fallback_exception: None | str = None,
    ):
        self.passed = passed
        self.fallback_exception = fallback_exception


class PipelineActor:
    def __init__(
        self,
        name: str,
        execution: Callable,
        fallback: Any | None = None,
        stop_on_failure: bool = False,
        **settings,
    ):
        self.name: str = name
        self.execution: Callable = execution
        self.fallback: PipelineActor | None = fallback
        self.stop_on_failure: bool = stop_on_failure
        self.settings: dict[str, Any] = settings

    def run(self) -> PipelineResult:
        try:
            self.execution()
            return PipelineResult(passed=True)
        except Exception as fallback_exception:
            if self.fallback:
                logging.info(
                    f"Pipeline actor {Fore.CYAN}{self.name}{Fore.RESET} executed with a {Fore.YELLOW}Fallback Exception{Fore.RESET}. Fallback to {Fore.CYAN}{self.fallback.name}{Fore.RESET}."
                )
                execution: PipelineResult = self.fallback.run()

                if execution.passed:
                    return PipelineResult(
                        passed=True, fallback_exception=repr(fallback_exception)
                    )
                else:
                    return execution
            else:
                return PipelineResult(
                    passed=False, fallback_exception=repr(fallback_exception)
                )


class Pipeline:
    def __init__(self, pipeline_actors: list[PipelineActor], **settings):
        self.pipeline_actors: list[PipelineActor] = pipeline_actors
        self.settings = settings

    def run(self):
        final_step = len(self.pipeline_actors)
        current_step = 1
        successful_steps = 0
        fatal_failure = False

        for actor in self.pipeline_actors:
            logging.info(
                f"Executing pipeline actor {Fore.CYAN}{actor.name}{Fore.RESET} ({Fore.CYAN}{current_step}{Fore.RESET}/{Fore.CYAN}{final_step}{Fore.RESET})"
            )

            actor_execution: PipelineResult = actor.run()
            current_step += 1

            if not actor_execution.passed and actor.stop_on_failure:
                logging.warning(
                    f"Couldn't execute pipeline actor {Fore.CYAN}{actor.name}{Fore.RESET} due to a {Fore.RED}Fatal Exception{Fore.RESET}:\n\n{Fore.RED}Fatal Exception{Fore.RESET}: {actor_execution.fallback_exception}\n"
                )
                fatal_failure = True
                break
            elif not actor_execution.passed and not actor.stop_on_failure:
                logging.warning(
                    f"Skipping pipeline {Fore.CYAN}{actor.name}{Fore.RESET} since no {Fore.YELLOW}fallback{Fore.RESET} reference has been passed and actor is set to {Fore.RED}not{Fore.RESET} early stop on failure."
                )
                continue

            if not actor_execution.fallback_exception:
                logging.info(
                    f"Pipeline actor {Fore.CYAN}{actor.name}{Fore.RESET} {Fore.GREEN}succesfully{Fore.RESET} executed."
                )
                successful_steps += 1

        if fatal_failure:
            logging.info(
                f"Pipeline early stop due to {Fore.RED}Fatal Exception{Fore.RESET}."
            )


def func1():
    print("did something")


def func2():
    print("did something again")


def func3():
    return int("a")


def func4():
    return int("b")


# my_actor1 = PipelineActor("actor1", func1)
# my_actor2 = PipelineActor("actor2", func2, my_actor1)
# my_actor3 = PipelineActor("actor3", func3, my_actor2)
# my_actor4 = PipelineActor("actor4", func4, my_actor3)

# my_pipeline = Pipeline(
#     [
#         my_actor1,
#         my_actor2,
#         my_actor3,
#         my_actor4,
#     ]
# )

# my_pipeline.run()

# ========================================================


class SuccessFlow:
    def __init__(self, pipeline: Pipeline):
        self._pipeline: Pipeline = pipeline

        self.colors: list[str] = []
        self.sources: list[int] = []
        self.targets: list[int] = []

        self.flow_callers: list[str] = []

        temp_color = "rgba(69, 209, 69, 0.4)"

        for i in range(0, len(pipeline.pipeline_actors)):
            if i < len(pipeline.pipeline_actors) - 1:
                self.colors.append(temp_color)
                self.sources.append(i)
                self.targets.append(i + 1)

                self.flow_callers.append(pipeline.pipeline_actors[i].name)


class FallbackIndexes:
    def __init__(self, pipeline: Pipeline):
        self._pipeline = pipeline

        self.indexes_with_fallbacks: list[int] = []
        self.fallback_callers: list[str] = []

        for i in range(0, len(pipeline.pipeline_actors)):
            if i <= len(pipeline.pipeline_actors):
                actor = pipeline.pipeline_actors[i]

                if actor.fallback:
                    self.indexes_with_fallbacks.append(i)
                    self.fallback_callers.append(pipeline.pipeline_actors[i].name)


class FallbackFlow:
    def __init__(self, pipeline: Pipeline, source_index: int):
        self._pipeline = pipeline
        self._source_index = source_index

        self.colors = []
        self.sources = []
        self.targets = []

        temp_colors = [
            "201, 167, 71",
            "201, 114, 71",
            "201, 80, 71",
            "107, 30, 25",
            "186, 43, 43",
        ]
        temp_color_index = random.randint(0, len(temp_colors) - 1)
        temp_color = f"rgba({temp_colors[temp_color_index]}, 0.4)"

        if source_index > 0 and source_index <= len(pipeline.pipeline_actors):
            self.colors.append(temp_color)
            self.sources.append(source_index)
            self.targets.append(source_index - 1)

            self.colors.append(temp_color)
            self.sources.append(source_index - 1)
            if source_index <= len(pipeline.pipeline_actors) - 1:
                self.targets.append(source_index + 1)
            else:
                self.targets.append(source_index)


def get_pipeline_diagram(pipeline: Pipeline, show_on_browser=True, save_image=True):
    successful_flow = SuccessFlow(pipeline)

    fallback_indexes = FallbackIndexes(pipeline)

    fallback_flows: list[FallbackFlow] = []
    for fallback_index in fallback_indexes.indexes_with_fallbacks:
        fallback_flows.append(FallbackFlow(pipeline, fallback_index))

    pipeline_colors: list[str] = successful_flow.colors
    pipeline_sources: list[int] = successful_flow.sources
    pipeline_targets: list[int] = successful_flow.targets
    pipeline_link_labels: list[str] = fallback_indexes.fallback_callers

    for flow in fallback_flows:
        pipeline_colors += flow.colors
        pipeline_sources += flow.sources
        pipeline_targets += flow.targets

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    # line=dict(color=["black", "blue", "blue"], width=0.5),
                    label=["actor1", "actor2", "actor3", "actor4"],
                    align="left",
                ),
                link=dict(
                    color=pipeline_colors,
                    arrowlen=15,
                    label=pipeline_link_labels,
                    source=pipeline_sources,
                    target=pipeline_targets,
                    value=[8] * len(pipeline_sources),
                ),
            )
        ]
    )

    fig.update_layout(title_text="Diagram - Pipeline", font_size=10)

    if show_on_browser:
        fig.show()

    if save_image:
        fig.write_image("pipeline.png")


# get_pipeline_diagram(my_pipeline)
