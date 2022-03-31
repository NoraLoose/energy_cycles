# Computing Lorenz-like and Bleck energy cycles via along-isopynal spatial filtering

Repository for computing the Lorenz and Bleck energy cycles in NeverWorld2

Execute notebooks in the following order:
0) make_bottom_mask.ipynb
1) filter_data.ipynb
2) compute_2d_Lorenz_cycle.ipynb and/or compute_2d_Bleck_cycle.ipynb
3) time_average_cycles.ipynb

Optionally: Compute 3d wind work diagnostics (needed for Loose et al. (2022)):
4) compute_3d_wind_work.ipynb

The directory python_scripts has python script versions of 1) and 2) that can be submitted as batch jobs (see directory submission_scripts).

[This repository](https://github.com/NoraLoose/loose_bachman_grooms_jansen_2022_jpo) has notebooks that take the data processed here and produce the plots in Loose et al. (2022), submitted to JPO.
