import plotly.graph_objects as go

actors = ["actor1", "actor2", "actor3"]

colors_all_rounder = [
    "rgba(69, 209, 69, 0.4)",
    "rgba(69, 209, 69, 0.4)",
    "rgba(69, 209, 69, 0.4)",
]
source_all_rounder = [0, 1, 2]
target_all_rounder = [1, 2, 3]

possib_i2_colors = ["rgba(235, 64, 52, 0.4)", "rgba(235, 64, 52, 0.4)"]
possib_i2_source = [2, 1]
possib_i2_target = [1, 3]

possib_i3_colors = ["rgba(177, 184, 53, 0.4)", "rgba(177, 184, 53, 0.4)"]
possib_i3_source = [3, 2]
possib_i3_target = [2, 3]

fig = go.Figure(
    data=[
        go.Sankey(
            arrangement="snap",
            node=dict(
                pad=15,
                thickness=20,
                # line=dict(color=["black", "blue", "blue"], width=0.5),
                label=["actor1", "actor2", "actor3", "actor4"],
                align="left",
            ),
            link=dict(
                color=colors_all_rounder + possib_i2_colors + possib_i3_colors,
                arrowlen=15,
                source=source_all_rounder + possib_i2_source + possib_i3_source,
                target=target_all_rounder + possib_i2_target + possib_i3_target,
                value=[8, 8, 8, 8, 8, 8, 8],
            ),
        )
    ]
)

fig.update_layout(title_text="Diagram - Pipeline", font_size=10)
fig.show()
