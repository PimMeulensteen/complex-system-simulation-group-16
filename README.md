# Group 16 - 

## Introduction and background
Cities are anything but random; they follow patterns and structures that can and have been studied. One intriguing aspect is the Negativity Density Gradient (NGD), reflecting the uneven distribution of the population within a city [3]. Understanding how NGD relates to other urban features is crucial for unravelling the complexities of city growth.

A possible interest Diffusion-Limited Aggregation (DLA), a mathematical model usually used in physics [1]. This technique is great at mimicking complex patterns through growth processes, making it useful in understanding urban development through formation of fractals. By using DLA to model cities, we can capture patterns that closely resemble real urban complexity, offering a unique approach beyond traditional urban planning methods [2].

This research aims to further investigate the fractal dimensions, branching ratio, and Negativity Density Gradient (NGD) in cities by modelling them through DLA. We want to explore the relationships between these elements and dig into whether the Negativity Density Gradient follows a power law, giving us insights into how negativity scales within city structures. Understanding these connections can offer new perspectives on how cities form and function, contributing to our broader knowledge of complex systems and urban development. 


[1] Fotheringham, A.S., Batty, M. & Longley, P.A. Diffusion-limited aggregation and the fractal nature of urban growth. Papers of the Regional Science Association 67, 55–69 (1989). https://doi.org/10.1007/BF01934667
[2] Batty, Mike, Paul Longley, and Stewart Fotheringham. "Urban growth and form: scaling, fractal geometry, and diffusion-limited aggregation." Environment and planning A 21.11 (1989): 1447-1472. https://doi.org/10.1068/a211447
[3] Broitman, D., & Koomen, E. (2020). The attraction of urban cores: Densification in Dutch city centres. Urban Studies, 57(9), 1920-1939. https://doi.org/10.1177/0042098019864019



## Installation instructions

`pip3 install .`


## Required packages
- Numpy
- Matplotlib 
- Powerlaw 

Note that these packages are also described in `requirements.txt`, and can be installed usign `pip3 install -r requirements.txt`.

## Bonus 
For this project, some bonus points were availilbe. From that, we have implemented the following:
- 5-10% ‘assert’ statements inline
- Structured as a module (`__init__.py`; e.g. `pip install .` works)
- Documentation generated from docstrings (`pdoc css_dla`). This is also availible in the folder `documentation_from_pdoc`.