from dash import html, dcc
import dash_bootstrap_components as dbc

first_paragraph = html.P(children=[
    """
    In deserts, forests, grasslands, and wetlands, the vegetation can be 
    distributed in a geometric pattern. For example, in pine forests, you may notice that large trees are usually spaced 
    out at least a few feet apart and rarely closer than that. This is because pine trees need adequate space to get 
    enough sunlight and nutrients from the soil and don't do as well when crowded. In contrast, many other types of 
    vegetation grow in dense clusters, where many individual plants are aggregated together to form a patch. These 
    patches can seem random in shape and size and location, or in some cases can form geometric patterns, such as evenly 
    spaced dots, mazes, or large stripes (see the examples below). Many of these patterns emerge from plants 
    spontaneously self-organizing due to the way they interact with each other and their environment. 
    """
])

other_paragraphs = html.P(children=[
    """
    This app explores these patterned landscapes by representing a landscape using a grid of cells, where each 
    cell is either occupied or empty. In the pine forest example, each occupied cell represents a 
    tree, and the entire grid represents a section of forest. To understand how each cell interacts with its 
    neighboring cells, we can use a kernel function. The kernel illustrates the relationship each cell has with its 
    neighboring cells. The relationship can be either positive (faciliatory) or negative (inhibitory) and is a 
    function of the distance between a cell and its neighbors. For the pine forest example, close neighbors have an 
    inhibitory effect: trees have to compete for sunlight and space and may not grow as well if they were more spaced 
    out. The kernel function for this case would be very negative for very close neighbors, somewhat negative for 
    less close neighbors, and zero at some distance further away. Choose the 'Random Forest' preset and click 'Start'
    to run this model. 
    """,
    html.Br(),
    html.Br(),
    """
    The kernel function illustrates this relationship graphically, showing a function that is very negative in the middle and 
    gradually approaches zero as the distance from the x,y origin increases. Conversely, we can have the opposite 
    effect, where having close neighbors is actually a good thing and contributes to growth. In dry landscapes where 
    water is scarce and sunlight is plentiful, larger plants may produce shading for younger seedlings. Likewise, 
    in some wetlands, vegetation can cause biomass to accumulate, promoting more growth in those areas. In many other 
    environments, plants can aggregate simply because there are more seeds dropped close to where plants already 
    exist. All of these are examples of positive facilitation, where having close neighbors is beneficial and 
    promotes growth. These relationships would be represented by a kernel that is positive in the middle and 
    approaches zero as you get further away. Patterns of this type are referred to as “scale-free" patterning because 
    they have many statistical features that can't be characterized by a characteristic scale of reference (such as 
    power-law distributions in patch sizes, similarities to fractals, and a lack of a characteristic wavelength). The 
    'Scale-Free' presets produce these patterns.
    """,
    html.Br(),
    html.Br(),
    """
    Some landscapes can have a mix of facilitation and inhibition from neighbors, depending on where those 
    neighbors are located. For instance, some boreal wetlands produce hills and hummocks, where the hills accumulate 
    nutrients (causing local facilitation on the hills), depleting nutrients at some distance away (causing 
    inhibition in the hummocks). This can be represented by a kernel that is positive at very close distances, 
    but negative at some distance further away. This can cause some very visually intriguing patterns, 
    where the landscape is characterized by patches that are spaced at regular intervals (corresponding to the 
    characteristic wavelength), forming dots, mazes, and wave-like features. This is referred to as "regular", 
    or "scale-dependent" patterning. The 'Periodic' presets show this model. Note how the wavelength of the landscape 
    (the distance between patch centers) these presets produce matche the wavelength of the kernel (the distance 
    between the minima). The pine forest example above also produces regular patterning, but lacks the clustering 
    effect because the kernel is strictly negative. 
    """,
    html.Br(),
    html.Br(),
    """
    In some landscapes, the effects of neighbors may be stronger in some directions than others. The slope of the 
    landscape in dry landscapes or the direction of water flow in wetlands can cause kernels that act differently in 
    different directions (i.e., they are anisotropic). The 'Scale-free Anisotropic' and 'Periodic Anisotropic' 
    presets illustrate this. 
    """,
    html.Br(),
    html.Br(),
    """
    The landscape always starts with a random 50/50 mix of occupied and empty cells, and then updates during each 
    iteration. During each iteration, the kernel function is applied to every cell in the 
    same way ( since each cell follows the same rules). This results in a calculation of the facilitation and 
    inhibition in each cell due to its neighbors. This is then translated into a probability of transitioning, 
    either from occupied to unoccupied or vice-versa, and then cells are randomly selected to transition based on 
    this probability. In this way, the cells collectively self-organize to produce a pattern across the entire 
    landscape using simple rules. Essentially, the landscape attempts to organize to the most stable configuration 
    possible, where the kernel functions are collectively maximized. The model also applies a global, 
    density dependent effect uniformly across the landscape, independent of the neighbor configurations. This global 
    effect represents the fact that landscapes often are limited in resources (water, nutrients, etc.) and functions 
    to keeps the landscape close to a target density. This density effect can be modified directly as a parameter in 
    the app."""
                                    ])

references = dcc.Markdown("""
[Borgogno, F., D'Odorico, P., Laio, F., & Ridolfi, L. (2009). Mathematical models of vegetation pattern formation in 
ecohydrology. _Reviews of Geophysics_, 47(1). )](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2007RG000256) 
\n\n [Casey, S. T., Cohen, M. J., Acharya, S., Kaplan, D. A., & Jawitz, J. W. (2016). Hydrologic controls on aperiodic 
spatial organization of the ridge–slough patterned landscape. _Hydrology and Earth System Sciences_, 20(11), 4457-4467.]
(https://hess.copernicus.org/articles/20/4457/2016/) 
\n\n 
[Rietkerk, M., & Van de Koppel, J. (2008). Regular pattern formation in real ecosystems. _Trends in ecology & evolution_
, 23(3), 169-175.](https://www.sciencedirect.com/science/article/pii/S0169534708000281) \n\n [Scanlon, T. M., Caylor, K.
 K., Levin, S. A., & Rodriguez-Iturbe, I. (2007). Positive feedbacks promote power-law clustering of Kalahari vegetation
. _Nature_, 449(7159), 209-212.](https://www.nature.com/articles/nature06060) """)
