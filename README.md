# MolecfitWrapper
A python wrapper for running molecfit on high-resolution spectra.

# Requirements
The Python script only requires numpy and astropy (as defined in requirements.txt).
It also requires a working installation of Molecfit (installation instructions can 
be found here: https://www.eso.org/sci/software/pipelines/#pipelines_table).
Since molecfit is called via the command line you can check the availability of molecfit
with the command: ```esorex --recipes```. If the ```molecfit-model``` and ```molecfit-calctrans``` recipes
are listed it is installed.
