""" Dash app for simulating patterned landscapes.

A Dash app that generates synthetic patterned landscapes using a kernel based stochastic model.

"""

import dash_bootstrap_components as dbc
import dash_daq as daq
import numpy as np
import plotly.graph_objects as go
from assets import theory_text
from dash import html, dcc, callback_context, Output, Input, State, ctx
from dash_extensions.enrich import Dash, Trigger
from flask import Flask
from scipy import signal

# Static parameters
grid_size = 200
loop_iterations = 4
target_cumulative_probability = 0.99
landscape_size = (grid_size, grid_size)
default_interval_length = 500
max_intervals = 500


server = Flask(__name__)
app = Dash(
    update_title=None,
    title="Patterned Landscapes",
    external_stylesheets=[dbc.themes.MORPH],
    server=server,
)

########################################### Layout ##############################################


main_header = html.H1(
    "Patterned Landscape Synthesizer", style={"marginTop": 12}, className="text-dark"
)


navbar = dbc.Navbar(
    color="primary",
    children=[
        dbc.Row(
            [
                dbc.Col(
                    dbc.NavLink(
                        "How to use", id="usage-link", href="#", className="text-light"
                    ),
                    width="auto",
                ),
                dbc.Col(
                    dbc.NavLink(
                        "Theory", id="theory-link", href="#", className="text-light"
                    )
                ),
                dbc.Col(
                    dbc.NavItem(
                        children=[
                            dbc.NavLink(
                                "References",
                                href="#",
                                id="references-link",
                                active=False,
                                className="text-light",
                            ),
                            dbc.Popover(
                                children=[
                                    dbc.PopoverHeader("References"),
                                    dbc.PopoverBody(theory_text.references),
                                ],
                                id="references-popover",
                                target="references-link",
                                trigger="focus",
                            ),
                        ]
                    )
                ),
                dbc.Col(
                    dbc.NavLink(
                        "Code",
                        href="https://github.com/stephencasey/PatternedLandscapes",
                        className="text-light",
                    )
                ),
                dbc.Col(
                    dbc.DropdownMenu(
                        toggleClassName="text-light",
                        label="Links",
                        nav=True,
                        children=[
                            dbc.DropdownMenuItem(
                                "Portfolio", href="http://stephentcasey.com/"
                            ),
                            dbc.DropdownMenuItem(
                                "GitHub", href="https://github.com/stephencasey"
                            ),
                            dbc.DropdownMenuItem(
                                "LinkedIn",
                                href="https://www.linkedin.com/in/steve-casey/",
                            ),
                        ],
                    )
                ),
            ],
            align="center",
            style={"margin": 1},
        )
    ],
)


usage_block = dbc.Collapse(
    style={"marginTop": 24, "marginBottom": 12},
    id="usage-block",
    is_open=False,
    children=[
        html.H3("How to use"),
        html.P(
            [
                "To get started, choose a preset from the dropdown and click the "
                '"Start" button. The app will then display a video of a synthetic landscape as it '
                "self-organizes. You can customize the model by setting your own "
                "parameter values. Hovering over the parameter names will give "
                "a brief description.",
                html.Br(),
                html.Br(),
                "Check out the Theory section for an in-depth explanation of the model "
                "and some real-world examples of patterned landscapes.",
            ]
        ),
    ],
)


image_style = {"height": "250px", "width": "250px", "margin": "1px"}

theory_block = dbc.Collapse(
    style={"marginTop": 24, "marginBottom": 12},
    id="theory-block",
    is_open=False,
    children=[
        html.H3("Theory"),
        theory_text.first_paragraph,
        dbc.Row(
            children=[
                dbc.Col(
                    md=3,
                    children=[
                        html.A(
                            children=[
                                html.Img(
                                    src="assets/ColvilleRiver.png",
                                    title="View on Google Earth",
                                    style=image_style,
                                ),
                                html.Figcaption("Colville River, Alaska, USA"),
                            ],
                            href="https://earth.google.com/web/@70.37572641,-151.05022964,2.19459822a,3032.28582961d,35y,0h,0t,0r",
                        )
                    ],
                ),
                dbc.Col(
                    md=3,
                    children=[
                        html.A(
                            children=[
                                html.Img(
                                    src="assets/SomalianTigerBush.png",
                                    title="View on Google Earth",
                                    style=image_style,
                                ),
                                html.Figcaption("Tiger Bush in Somalia"),
                            ],
                            href="https://earth.google.com/web/@8.04148487,47.42627849,669.87775195a,12558.16921194d,35y,-0h,0t,0r",
                        ),
                    ],
                ),
                dbc.Col(
                    md=3,
                    children=[
                        html.A(
                            children=[
                                html.Img(
                                    src="assets/PandamatengaZimbabwe.png",
                                    title="View on Google Earth",
                                    style=image_style,
                                ),
                                html.Figcaption("Pandamatenga, Zimbabwe"),
                            ],
                            href="https://earth.google.com/web/@-18.26740141,25.65427973,1063.45767194a,1696.71541945d,35y,0h,0t,0r",
                        ),
                    ],
                ),
                dbc.Col(
                    md=3,
                    children=[
                        html.A(
                            children=[
                                html.Img(
                                    src="assets/RidgeSlough.png",
                                    title="View on Google Earth",
                                    style=image_style,
                                ),
                                html.Figcaption("Everglades, Florida, USA"),
                            ],
                            href="https://earth.google.com/web/@26.05387327,-80.72663014,5.94693549a,24833.2498212d,35y,0h,0t,0r",
                        ),
                    ],
                ),
            ]
        ),
        html.P("Source: Google Earth", className="text-sm-end"),
        theory_text.other_paragraphs,
    ],
)


main_parameter_controls = dbc.Row(
    style={"marginTop": 24, "marginBottom": 12},
    children=[
        html.H4("Parameters"),
        dbc.Col(
            md=2,
            children=[
                dbc.Label("Presets", id="presets-label", html_for="model-preset"),
                dbc.Select(
                    style={"line-height": 12, "marginBottom": 12},
                    id="model-preset",
                    value="periodic_1",
                    options=[
                        {"label": "Periodic Labyrinth", "value": "periodic_1"},
                        {"label": "Periodic Dots", "value": "periodic_2"},
                        {"label": "Periodic Anisotropic", "value": "periodic_a"},
                        {"label": "Scale-free", "value": "scale_free_i"},
                        {"label": "Scale-free Anisotropic", "value": "scale_free_a"},
                        {"label": "Random Forest", "value": "random_forest"},
                        {"label": "Custom", "value": "custom"},
                    ],
                ),
            ],
        ),
        dbc.Col(
            md=2,
            children=[
                dbc.Label(
                    "Kernel function",
                    id="kernel-function-label",
                    html_for="kernel-function",
                ),
                dbc.Select(
                    id="kernel-function",
                    options=[
                        {"label": "Sinusoid", "value": "sinusoid"},
                        {"label": "Linear", "value": "linear"},
                        {"label": "Exponential", "value": "exponential"},
                        {"label": "Power Law", "value": "power"},
                    ],
                    value="power",
                    style={"line-height": 12, "marginBottom": 12},
                ),
            ],
        ),
        dbc.Col(
            md=2,
            children=[
                dbc.Label("Density", id="density-label", html_for="scaling-parameter"),
                dcc.Slider(
                    id="target-density",
                    min=0,
                    max=1,
                    step=0.01,
                    value=0.5,
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ],
        ),
        dbc.Col(
            md=2,
            children=[
                dbc.Label(
                    "Scaling",
                    id="scaling-parameter-label",
                    html_for="scaling-parameter",
                ),
                dcc.Slider(
                    id="scaling-parameter",
                    min=0.1,
                    max=8,
                    step=0.05,
                    value=2.5,
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ],
        ),
        dbc.Col(
            md=2,
            children=[
                dbc.Label(
                    "Elongation",
                    id="elongation-parameter-label",
                    html_for="elongation-parameter",
                ),
                dcc.Slider(
                    id="elongation-parameter",
                    min=1,
                    max=5,
                    step=0.05,
                    value=1,
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ],
        ),
        dbc.Col(
            md=1,
            children=[
                dbc.Label(
                    "Expert Mode", id="expert-mode-label", html_for="expert-mode-switch"
                ),
                daq.BooleanSwitch(id="expert-mode-switch", on=False),
            ],
        ),
        dbc.Collapse(
            id="expert-mode-block",
            is_open=True,
            children=[
                dbc.Row(
                    children=[
                        dbc.Col(
                            md=2,
                            children=[
                                dbc.Label(
                                    "Invert kernel",
                                    id="invert-label",
                                    html_for="invert-switch",
                                ),
                                daq.BooleanSwitch(id="invert-switch", on=False),
                            ],
                        ),
                        dbc.Col(
                            md=2,
                            children=[
                                dbc.Collapse(
                                    id="wavelength-collapse",
                                    is_open=True,
                                    children=[
                                        dbc.Label(
                                            "Wavelength",
                                            id="wavelength-parameter-label",
                                            html_for="wavelength-parameter",
                                        ),
                                        dcc.Slider(
                                            id="wavelength-parameter",
                                            min=1,
                                            max=10,
                                            step=0.1,
                                            value=5,
                                            marks=None,
                                            tooltip={
                                                "placement": "bottom",
                                                "always_visible": True,
                                            },
                                        ),
                                    ],
                                )
                            ],
                        ),
                        dbc.Col(
                            md=2,
                            children=[
                                dbc.Label(
                                    "Density correction",
                                    id="density-correction-label",
                                    html_for="density-correction",
                                ),
                                dcc.Slider(
                                    id="density-correction",
                                    min=0,
                                    max=5,
                                    step=0.1,
                                    value=1,
                                    marks=None,
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": True,
                                    },
                                ),
                            ],
                        ),
                        dbc.Col(
                            md=2,
                            children=[
                                dbc.Label(
                                    "Interval length",
                                    id="interval-length-label",
                                    html_for="interval-length",
                                ),
                                dcc.Slider(
                                    id="interval-length",
                                    min=250,
                                    max=2000,
                                    step=50,
                                    value=default_interval_length,
                                    marks=None,
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": True,
                                    },
                                ),
                            ],
                        ),
                        dbc.Col(
                            md=2,
                            children=[
                                dbc.Label(
                                    "Δ per iteration",
                                    id="change-per-iter-label",
                                    html_for="change-per-iter",
                                ),
                                dcc.Slider(
                                    id="change-per-iteration",
                                    min=0.05,
                                    max=1,
                                    step=0.05,
                                    value=0.15,
                                    marks=None,
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": True,
                                    },
                                ),
                            ],
                        ),
                    ]
                )
            ],
        ),
    ],
)


start_stop_row = dbc.Row(
    children=[
        dbc.Col(
            md=4,
            children=[
                dbc.Button(
                    "Start",
                    id="start-stop",
                    size="lg",
                ),
                dbc.Button(
                    "Reset landscape",
                    id="reset",
                    type="reset",
                    size="lg",
                    style={
                        "line-height": 0,
                        "background-color": "#378dfc",
                        "color": "white",
                        "margin": 6,
                    },
                ),
            ],
        ),
    ]
)

top_figure_block = dbc.Row(
    style={"marginTop": 24, "marginBottom": 12},
    children=[
        html.Hr(),
        dbc.Col(
            md=6,
            children=[
                html.H4("Landscape"),
                dcc.Graph(id="landscape-plot", config={"plotGlPixelRatio": 5}),
            ],
        ),
        dbc.Col(
            md=6,
            children=[
                html.H4("Kernel"),
                dcc.Loading(dcc.Graph(id="continuous-kernel-plot")),
            ],
        ),
    ],
)


discrete_kernel_block = html.Div(
    id="discrete-kernel-div",
    hidden=True,
    style={"display": "inline-block"},
    children=[
        html.H4("Discretized Kernel"),
        dcc.Loading(dcc.Graph(id="discrete-kernel-plot")),
    ],
)


tooltips = html.Div(
    children=[
        dbc.Tooltip(
            "The shape of the kernel. The sinusoid function is used to produce periodic patterns, while the "
            "power-law, exponential, and linear functions produce scale-free patterning",
            target=f"kernel-function-label",
            placement="top",
        ),
        dbc.Tooltip(
            "Determines how many cells on the landscape are vegetated. For strictly-positive kernels, this is equal to the "
            "proportion of cells that are vegetated.",
            target=f"density-label",
            placement="top",
        ),
        dbc.Tooltip(
            "Steepness of the kernel. Higher values usually produce patches that are more densely aggregated.",
            target=f"scaling-parameter-label",
            placement="top",
        ),
        dbc.Tooltip(
            "Determines how elongated patches are. A value of 1 corresponds to an isotropic kernel",
            target=f"elongation-parameter-label",
            placement="top",
        ),
        dbc.Tooltip(
            "Wavelength of the sinusoid kernel function. For periodic landscapes, this matches the landscape pattern's "
            "wavelength.",
            target=f"wavelength-parameter-label",
            placement="top",
        ),
        dbc.Tooltip(
            "Unlocks more parameters and visualizations",
            target="expert-mode-label",
            placement="top",
        ),
        dbc.Tooltip(
            "Flips the kernel along the z-axis",
            target="invert-label",
            placement="top",
        ),
        dbc.Tooltip(
            "Weights the density algorithm to scale the density up or down. Useful for inverted and sinusoid kernels where"
            "the generated density is low.",
            target="density-correction-label",
            placement="top",
        ),
        dbc.Tooltip(
            "Determines the rate that the video gets updated from the server. Values too high cause slow model progression,"
            " values too low will cause the video to freeze.)",
            target="interval-length-label",
            placement="top",
        ),
        dbc.Tooltip(
            "Determines the proportion of cells that change per iteration.",
            target="change-per-iter-label",
            placement="top",
        ),
        dbc.Tooltip(
            "Displays the shape of the kernel. Positive values represent facilitation while negative values represent "
            "inhibition.",
            target="continuous-kernel-plot",
            placement="top",
        ),
        dbc.Tooltip(
            "The actual, implemented kernel in the model. The x,y coordinate (0,0) corresponds to the target cell. Positive"
            " neighbor cells indicate positive faciliation while negative neighbor cells indicate inhibition.",
            target="discrete-kernel-plot",
            placement="right",
        ),
    ]
)

max_iteration_alert = dbc.Alert(
    "You've reached the max number of iterations! Refresh the page to run again.",
    id="max-alert",
    dismissable=True,
    is_open=False,
)

# App layout
app.layout = dbc.Container(
    fluid=True,
    children=[
        max_iteration_alert,
        main_header,
        navbar,
        usage_block,
        theory_block,
        main_parameter_controls,
        start_stop_row,
        top_figure_block,
        discrete_kernel_block,
        tooltips,
        dcc.Interval(
            id="interval",
            interval=default_interval_length,
            max_intervals=max_intervals,
            disabled=True,
        ),
        dcc.Store(id="kernel-data"),
        dcc.Store(id="landscape-data"),
    ],
)


########################################### Model ###################################################


def make_distance_matrix(target_distance, elongation_parameter, is_continuous_kernel):
    """Produces a matrix that represents the Euclidean distance from the center cell, up to a cutoff distance.

    Args:
        target_distance (float): Distance from center cell that captures the target_cumulative_probability.
        elongation_parameter (float): Length-to-width ratio of kernel.
        is_continuous_kernel (bool):  True for continuous plotting kernel, False for discretized model kernel.

    Returns:
        ndarray: 2D array representing distance from center cell, up to target_distance.

    """

    if is_continuous_kernel:
        cell_spacing = target_distance / 100
    else:
        cell_spacing = 1
    effective_distance = (
        target_distance * max((elongation_parameter, 1 / elongation_parameter)) + 1
    )
    distance_range = list(
        np.arange(0, round(np.ceil(effective_distance) + 1), cell_spacing)
    )
    x_coord, y_coord = np.meshgrid(distance_range, distance_range)

    # Apply elongation by stretching the y-axis
    y_coord = y_coord / elongation_parameter

    # Make one quadrant of distance matrix
    distances = np.sqrt(np.square(x_coord) + np.square(y_coord))

    # Find distance that ensures AT LEAST target distance is included
    distance_list = np.unique(distances)
    cutoff_distance = min(distance_list[distance_list > target_distance])

    # Remove distances under cutoff
    distances = distances * (distances <= cutoff_distance)
    distances = distances[~(distances == 0).all(axis=1), :]
    distances = distances[:, ~(distances == 0).all(axis=0)]
    distances[distances == 0] = np.nan  # Avoid dividing by zero

    # Copy single quadrant to the other quadrants to make a complete matrix
    distance_righthalf = np.concatenate((distances[:0:-1], distances))
    distances = np.concatenate(
        (distance_righthalf.T[:0:-1].T, distance_righthalf), axis=1
    )

    # Check that kernel is small enough to be performant (truncate recursively if not)
    under_cutoff = distances <= cutoff_distance
    if not is_continuous_kernel and (np.sum(under_cutoff) > 128):
        new_target_distance = min(10, target_distance - 1)
        distances = make_distance_matrix(
            new_target_distance, elongation_parameter, is_continuous_kernel
        )

    return distances


def make_kernel(distance, model, scaling_parameter, wavelength_parameter):
    """Generates a 2-D kernel as an array"""
    if model == "power":
        kernelmodel = np.power(np.divide(1, distance), scaling_parameter)
    elif model == "exponential":
        kernelmodel = np.exp(np.multiply(-scaling_parameter, distance))
    elif model == "sinusoid":
        kernelmodel = np.exp(-scaling_parameter * distance) * np.cos(
            3 * np.pi * distance / (2 * wavelength_parameter)
        )
    else:  # linear model
        kernelmodel = 1 - np.divide(distance, scaling_parameter)

    kernelmodel[np.isnan(kernelmodel)] = 0

    return kernelmodel


@app.callback(
    Output("model-preset", "value"),
    Output("kernel-function", "value"),
    Output("invert-switch", "on"),
    Output("scaling-parameter", "value"),
    Output("scaling-parameter", "min"),
    Output("elongation-parameter", "value"),
    Output("wavelength-parameter", "value"),
    Output("continuous-kernel-plot", "figure"),
    Output("discrete-kernel-plot", "figure"),
    Output("target-density", "value"),
    Output("density-correction", "value"),
    Output("wavelength-collapse", "is_open"),
    Output("kernel-data", "data"),
    Input("model-preset", "value"),
    Input("kernel-function", "value"),
    Input("invert-switch", "on"),
    Input("scaling-parameter", "value"),
    Input("elongation-parameter", "value"),
    Input("wavelength-parameter", "value"),
    State("target-density", "value"),
    State("density-correction", "value"),
)
def build_kernels(
    model_preset,
    kernel_function,
    invert_kernel,
    scaling_parameter,
    elongation_parameter,
    wavelength_parameter,
    target_density,
    density_correction,
):

    # Preset definitions
    trigger_event = callback_context.triggered[0]["prop_id"].split(".")[0]
    if trigger_event and (trigger_event != "model-preset"):
        model_preset = "custom"
    elif model_preset == "scale_free_a":
        kernel_function = "exponential"
        invert_kernel = False
        scaling_parameter = 5
        elongation_parameter = 3
        target_density = 0.5
        density_correction = 1
    elif model_preset == "scale_free_i":
        kernel_function = "exponential"
        invert_kernel = False
        scaling_parameter = 3
        elongation_parameter = 1
        target_density = 0.5
        density_correction = 1    
    elif model_preset == "periodic_1":
        kernel_function = "sinusoid"
        invert_kernel = False
        scaling_parameter = 0.5
        elongation_parameter = 1
        target_density = 0.7
        wavelength_parameter = 7.5
        density_correction = 1.3
    elif model_preset == "periodic_2":
        kernel_function = "sinusoid"
        invert_kernel = False
        scaling_parameter = 0.5
        elongation_parameter = 1
        target_density = 0.5
        wavelength_parameter = 6
        density_correction = 1
    elif model_preset == "periodic_a":
        kernel_function = "sinusoid"
        invert_kernel = False
        scaling_parameter = 1.2
        elongation_parameter = 2.5
        wavelength_parameter = 3.5
        target_density = 0.7
        density_correction = 1.2
    elif model_preset == "random_forest":
        kernel_function = "linear"
        invert_kernel = True
        scaling_parameter = 2.2
        elongation_parameter = 1
        target_density = 0.4
        density_correction = 0.8

    # Hide/show wavelength parameter
    if kernel_function == "sinusoid":
        wavelength_open = True
    else:
        wavelength_open = False

    # Find distance that yields EXACTLY target_cumulative_p % of the entire distribution (eg. 99%)
    scaling_min = 0.1
    if kernel_function == "power":
        # Keep scaling parameter in sane range
        scaling_min = 1
        if scaling_parameter < 1:
            scaling_parameter = 1
        target_distance = (1 - target_cumulative_probability) ** (
            -1 / scaling_parameter
        )
    elif kernel_function == "exponential":
        target_distance = -np.log(1 - target_cumulative_probability) / scaling_parameter
    elif kernel_function == "sinusoid":  # Use 2nd x-intercept instead
        target_distance = wavelength_parameter
    else:  # Linear model
        target_distance = scaling_parameter  # x-intercept represents 100%

    distance = make_distance_matrix(
        target_distance, elongation_parameter, is_continuous_kernel=False
    )
    kernel = make_kernel(
        distance, kernel_function, scaling_parameter, wavelength_parameter
    )

    # Generate a continuous kernel for plotting
    # Note: c_kernel and kernel show drastically different values near the origin, due to the distance matrix for
    # c_kernel being able to appoach zero. Rescaling to (-1,1) produces an acceptable (albeit not perfect) remedy.
    c_distance = make_distance_matrix(
        target_distance, elongation_parameter, is_continuous_kernel=True
    )
    c_kernel = make_kernel(
        c_distance, kernel_function, scaling_parameter, wavelength_parameter
    )

    # Rescale kernels to (-1,1)
    kernel = kernel / np.max(np.abs(kernel))
    c_kernel = c_kernel / np.max(np.abs(c_kernel))

    # Apply inversion to both kernel models
    kernel = ((-1) ** invert_kernel) * kernel
    c_kernel = ((-1) ** invert_kernel) * c_kernel

    c_kernel[c_kernel == 0] = np.nan
    k_copy = kernel.copy()
    k_copy[k_copy == 0] = np.nan
    kernel_shape = kernel.shape
    c_kernel_shape = c_kernel.shape

    x_ticklabels = list(
        range(int(-(kernel_shape[1] - 1) / 2), int((kernel_shape[1] + 1) / 2))
    )
    y_ticklabels = list(
        range(int(-(kernel_shape[0] - 1) / 2), int((kernel_shape[0] + 1) / 2))
    )
    xticks = list(range(kernel_shape[1]))
    yticks = list(range(kernel_shape[0]))
    c_xspacing = int(c_kernel_shape[1] / len(xticks))
    c_yspacing = int((c_kernel_shape[0]) / len(yticks))
    c_xticks = list(
        range(int(c_xspacing / 2), c_kernel_shape[1] + c_xspacing, c_xspacing)
    )
    c_yticks = list(
        range(int(c_yspacing / 2), c_kernel_shape[0] + c_yspacing, c_yspacing)
    )

    # Plot continuous kernel
    c_kernel_fig = go.Figure(go.Surface(z=c_kernel, coloraxis="coloraxis"))
    c_kernel_fig.update_scenes(
        aspectmode="manual",
        aspectratio=dict(x=1, y=elongation_parameter, z=1),
    )
    c_kernel_fig.update_layout(
        {
            "plot_bgcolor": "rgba(0,0,0,0)",
            "paper_bgcolor": "rgba(0,0,0,0)",
        },
        coloraxis=dict(colorscale="Plasma"),
        scene=dict(
            xaxis=dict(
                showbackground=False,
                gridcolor="silver",
                tickvals=c_xticks,
                ticktext=x_ticklabels,
            ),
            yaxis=dict(
                showbackground=False,
                gridcolor="silver",
                tickvals=c_yticks,
                ticktext=y_ticklabels,
            ),
            zaxis=dict(
                showbackground=False,
                gridcolor="silver",
                zerolinecolor="black",
                zerolinewidth=4,
            ),
        ),
        margin=dict(l=30, r=30, b=30, t=30),
    )

    # Plot discretized kernel
    kernel_fig = go.Figure(
        go.Heatmap(z=k_copy, coloraxis="coloraxis"),
    )
    kernel_fig.update_layout(
        {
            "plot_bgcolor": "rgba(0,0,0,0)",
            "paper_bgcolor": "rgba(0,0,0,0)",
        },
        coloraxis=dict(colorscale="Plasma"),
        xaxis=dict(gridcolor="silver", tickvals=xticks, ticktext=x_ticklabels),
        yaxis=dict(
            scaleanchor="x", gridcolor="silver", tickvals=yticks, ticktext=y_ticklabels
        ),
        margin=dict(l=30, r=30, b=30, t=30),
    )

    return (
        model_preset,
        kernel_function,
        invert_kernel,
        scaling_parameter,
        scaling_min,
        elongation_parameter,
        wavelength_parameter,
        c_kernel_fig,
        kernel_fig,
        target_density,
        density_correction,
        wavelength_open,
        kernel,
    )


@app.callback(
    Output("landscape-plot", "figure"),
    Output("landscape-data", "data"),
    Trigger("interval", "n_intervals"),
    Trigger("reset", "n_clicks"),
    State("density-correction", "value"),
    State("change-per-iteration", "value"),
    State("landscape-data", "data"),
    State("kernel-data", "data"),
    State("target-density", "value"),
)
def run_model(
    density_correction,
    change_per_iteration,
    landscape,
    kernel,
    target_density,
):
    """Runs the model and produces figures using Dash interval callbacks"""

    trigger_event = callback_context.triggered[0]["prop_id"].split(".")[0]

    if trigger_event != "interval":
        landscape = np.random.randint(0, 2, landscape_size)

    # Interval acts as an outer loop
    else:
        # Determine max & min facilitation possible for each cell to normalize facilitation function in run_model
        landscape_ones = np.ones(landscape_size)
        positive_kernel = np.array(kernel.copy())
        positive_kernel[positive_kernel < 0] = 0
        max_facilitation = signal.convolve2d(
            landscape_ones, positive_kernel, boundary="fill", mode="same"
        )

        landscape = np.array(landscape)
        # Rescale target density higher for kernels with negative values
        effective_density = target_density * density_correction

        # Speed up model progression by running multiple iterations within each dcc.interval
        # Loop running time should be as close to interval time without exceeding
        for n in range(loop_iterations):
            facilitation = signal.convolve2d(
                landscape, kernel, boundary="fill", mode="same"
            )
            kernel_effect = np.divide(facilitation, max_facilitation)
            density = np.mean(landscape) / density_correction

            probability_0_to_1 = kernel_effect + (effective_density - density) / (
                1 - density
            )
            probability_1_to_0 = (1 - kernel_effect) + (
                density - effective_density
            ) / density

            # Choose which selected cells to transition
            random_numbers = np.random.random(landscape_size)
            change_to_zero = (
                landscape * probability_1_to_0 * change_per_iteration
            ) >= random_numbers
            change_to_one = (
                (1 - landscape) * probability_0_to_1 * change_per_iteration
            ) >= random_numbers
            landscape = landscape + change_to_one - change_to_zero

    landscape_trace = go.Heatmap(
        z=landscape,
        zmin=0,
        zmax=1,
        colorscale=[(0, "white"), (0.5, "white"), (0.5, "black"), (1, "black")],
        colorbar=dict(
            title="Cell Type",
            tickvals=[0.25, 0.75],
            ticktext=["Empty", "Vegetated"],
            lenmode="pixels",
            len=100,
            outlinewidth=1,
        ),
    )
    landscape_layout = go.Layout(
        {"plot_bgcolor": "rgba(0,0,0,0)", "paper_bgcolor": "rgba(0,0,0,0)"},
        margin=dict(t=30, r=30, b=30, l=30),
        yaxis=dict(scaleanchor="x", constrain="domain"),
        xaxis=dict(constrain="domain"),
    )

    landscape_fig = {"data": [landscape_trace], "layout": landscape_layout}

    return landscape_fig, landscape


####################################### Ancillary Callbacks ########################################


# Start/stop and interval length callbacks
@app.callback(
    Output("interval", "interval"),
    Output("interval", "disabled"),
    Output("start-stop", "style"),
    Output("start-stop", "children"),
    Input("interval-length", "value"),
    Input("interval", "disabled"),
    Input("start-stop", "n_clicks"),
    Input("reset", "n_clicks"),
)
def interval_options(
    length,
    interval_disabled,
    reset,
    n_clicks
):
    trigger_context = ctx.triggered_id
    if trigger_context == 'start-stop': 
        interval_disabled = not interval_disabled
    else:
        interval_disabled = True

    if interval_disabled: 
        return (
            length,
            True,
            {"background-color": "#40d933", "color": "white", "line-height": 0},
            "Start",
        )   
    else:
        return (
            length,
            False,
            {"background-color": "#FF4500", "color": "white", "line-height": 0},
            "Stop",
        )
    


# Turn on/off expert mode
@app.callback(
    Output("expert-mode-block", "is_open"),
    Output("discrete-kernel-div", "hidden"),
    Input("expert-mode-switch", "on"),
)
def expert_mode_optoins(is_displayed):
    return is_displayed, not is_displayed


# Hide/show 'How to use' and 'Theory' sections
@app.callback(
    Output("usage-block", "is_open"),
    Input("usage-link", "n_clicks"),
    State("usage-block", "is_open"),
)
def show_howtouse(n_clicks, is_displayed):
    if n_clicks:
        return not is_displayed
    return is_displayed


@app.callback(
    Output("theory-block", "is_open"),
    Input("theory-link", "n_clicks"),
    State("theory-block", "is_open"),
)
def show_theory(n_clicks, is_displayed):
    if n_clicks:
        return not is_displayed
    return is_displayed


# Display message when max intervals reached
@app.callback(Output("max-alert", "is_open"), Input("interval", "n_intervals"))
def max_reached(n_intervals):
    if n_intervals and (n_intervals >= max_intervals):
        return True
    return False


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
