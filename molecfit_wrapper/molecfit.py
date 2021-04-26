import configparser
import copy
import os
import re
import subprocess
from os import getcwd
from os.path import basename, dirname, join, realpath, splitext
from tempfile import NamedTemporaryFile

import numpy as np
from astropy import constants as const
from astropy.io import fits
import matplotlib.pyplot as plt


class RecipeConfig(dict):
    """
    This class handles the configuration parameters settings file for esorex recipes.
    The files can be created using ´´´esorex --create-config=<filename> <recipe-name>```
    """
    def __init__(self, parameters=None, replacement=None):
        super().__init__()
        self.update(parameters)
        if replacement is None:
            replacement = {"FALSE": False, "TRUE": True, "NULL": None}
        #:dict: The strings replacements that are used to convert from esorex format to python
        self.replacement = replacement

    @property
    def replacement_inverse(self):
        return {v: k for k, v in self.replacement.items()}

    def parse(self, default):
        """ 
        Parse the values in the input dictonary default from strings 
        to more useful and pythonic data types.
        Replacements for specific strings are defined in self.replacement
        This method is shallow, i.e. it only works on the top level of default
        Note also that it operates in place, i.e. the input is modified

        Parameters
        ----------
        default : dict
            Dictionary with parameters to convert.

        Returns
        -------
        default : dict
            The same object as was input, but with the items converted
        """         
        for key, value in default.items():
            if value in self.replacement.keys():
                value = self.replacement[value]
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
            default[key] = value
        return default

    def unparse(self, parameters):
        """
        Similar to parse, but in reverse it replaces the values in parameters
        with strings that are understood by esorex.

        Parameters
        ----------
        parameters : dict
            Input dictionary with python datatypes

        Returns
        -------
        parameters : dict
            The same dictionary but with all strings
        """        
        default = parameters.copy()
        inverse = self.replacement_inverse
        for key, value in default.items():
            if value in inverse.keys():
                value = inverse[value]
            else:
                value = str(value)
            default[key] = value
        return default

    @classmethod
    def read(cls, filename, replacement=None):
        """
        Read a configuration file from disk

        TODO: right now we are using config parser for the file parsing
        TODO: also this removes the comments

        Parameters
        ----------
        filename : str
            Name of the configuration file to read
        replacement : dict, optional
            replacement for speciific strings, see parse, by default None

        Returns
        -------
        self : RecipeConfig
            The read and parsed configuration
        """        

        with open(filename) as f:
            content = f.read()
        # Since this is not actually an ini file (its missing sections)
        # we need to hack a section header on the top
        content = "[DEFAULT]\n" + content
        config = configparser.ConfigParser()
        config.read_string(content)
        default = dict(config["DEFAULT"])
        self = cls(replacement=replacement)
        self.update(self.parse(default))
        return self

    def write(self, filename):
        """
        Write the configuration file to disk

        Parameters
        ----------
        filename : str
            filename of the destination
        """
        # We copy self.parameters, since unparse works in place
        params = self.parameters.copy()
        params = self.unparse(params)
        content = []
        for key, value in params.items():
            content += [f"{key} = {value}\n"]

        with open(filename, "w") as f:
            f.writelines(content)

    def get_recipe_options(self):
        """
        Generate the recipe options parameters that can be passed to esorex
        instead of the recipe configuration file

        Returns
        -------
        options : list
            recipe options for exorex
        """        
        params = self.unparse(self.parameters)
        options = []
        for key, value in params.items():
            options += [f"--{key.upper()}={value}"]
        return options


class SofFile:
    """
    Handles the 'Set Of Files' data files that are used by esorex.
    Essentially a list of files and their descriptor that is then used by esorex.

    TODO: the class could be more sophisticated using e.g. a numpy array
    or even pandas, but that would be overkill for now
    """
    def __init__(self, data=None):
        if data is None:
            self.data = []
        else:
            self.data = data

    @classmethod
    def read(cls, filename):
        """
        Reads a sof file from disk

        Parameters
        ----------
        filename : str
            name of the file to read

        Returns
        -------
        self : SofFile
            The read file
        """        
        data = []
        with open(filename, "r") as f:
            for line in f.readline():
                fname, ftype = line.split(maxsplit=1)
                data += [(fname, ftype)]
        self = cls(data)
        return self

    def write(self, filename):
        """
        Writes the sof file to disk

        Parameters
        ----------
        filename : str
            The name of the file to store
        """        
        content = []
        for fname, ftype in self.data:
            content += [f"{fname} {ftype}\n"]

        with open(filename, "w") as f:
            f.writelines(content)


class Esorex:
    """
    This is a wrapper to the esorex command line utility by ESO.
    Note that this does not replace a esorex installation, 
    it simply calls esorex via the command line.

    For full documentation on esorex see: https://www.eso.org/sci/software/cpl/esorex.html
    """
    def __init__(self, esorex_exec=None, recipe_dir=None):
        if esorex_exec is None:
            exorex_exec = "esorex"
        self.esorex_exec = esorex_exec
        self.recipe_dir = recipe_dir
        self.recipes = None

    def esorex(self, recipe, esorex_options=(), recipe_options=(), sof=""):
        if isinstance(esorex_options, str):
            esorex_options = (esorex_options,)
        if isinstance(recipe_options, str):
            recipe_options = (recipe_options,)

        # Add fixed options from self
        if self.recipe_dir is not None:
            esorex_options = (f"--recipe-dir={self.recipe_dir}", *esorex_options)
        # Assemble the command line arguments
        command = ["esorex", *esorex_options, recipe, *recipe_options, sof]
        result = subprocess.run(command, capture_output=True)
        # The only output we get from esorex is the console output
        stdout = result.stdout.decode()
        return stdout

    def get_default_parameters(self, recipe):
        with NamedTemporaryFile("r", suffix=".rc") as ntf:
            esorex_options = (f"--create-config={ntf.name}",)
            stdout = self.esorex(recipe, esorex_options=esorex_options)

            config = RecipeConfig.read(ntf.name)
        return config

    def get_recipes(self):
        stdout = self.esorex("--recipes")
        lines = stdout.split("\n")
        # TODO: this is all kind of clunky but it works?
        # Keep only the lines with recipe names and info
        lines = lines[5:-3]
        lines = [l[2:] for l in lines]

        self.recipes = {}
        entry = None
        for line in lines:
            if entry is None:
                name, entry = line.split(":", 1)
                name = name.strip()
                continue
            elif line.startswith(" "):
                entry += " " + line.lstrip()
                continue
            else:
                self.recipes[name] = entry
                name, entry = line.split(":", 1)
                name = name.strip()
                continue
        self.recipes[name] = entry

        return self.recipes


class Molecfit(Esorex):
    def __init__(
        self,
        recipe_dir=None,
        output_dir=None,
        column_wave="lambda",
        column_flux="flux",
        column_err="err",
        wlg_to_micron=0.0001,
    ):
        super().__init__(recipe_dir=recipe_dir)
        #:str: Column name of the wavelength in the FITS file
        self.column_wave = column_wave
        #:str: Column name of the flux in the FITS file
        self.column_flux = column_flux
        #:str: Column name of the flux uncertainties in the FITS file
        self.column_err = column_err
        #:float: factor to convert the wavelength in the file to micron
        self.wlg_to_micron = wlg_to_micron
        if output_dir is None:
            output_dir = join(dirname(__file__), "data")
        #:str: Directory to store intermediary and output data products
        self.output_dir = realpath(output_dir)

    def prepare_sof(self, filename, data):
        sof = SofFile(data)
        sof.write(filename)
        return sof

    def prepare_rc(self, filename, data):
        config = RecipeConfig(parameters=data)
        config.write(filename)
        return config

    def prepare_fits(self, header, wave, flux, err=None):
        # Write out the s1d spectrum in a format that molecfit eats.
        # This is a fits file with an empty primary extension that contains the header of the original s1d file.
        # Plus an extension that contains a binary table with 3 columns.
        # The names of these columns need to be indicated in the molecfit parameter file,
        # as well as the name of the file itself. This is currently hardcoded.
        if err is None:
            err = np.sqrt(flux)
        col1 = fits.Column(name=self.column_wave, format="1D", array=wave)
        col2 = fits.Column(name=self.column_flux, format="1D", array=flux)
        col3 = fits.Column(name=self.column_err, format="1D", array=err)
        cols = fits.ColDefs([col1, col2, col3])
        tbhdu = fits.BinTableHDU.from_columns(cols)
        prihdr = copy.deepcopy(header)
        prihdu = fits.PrimaryHDU(header=prihdr)
        thdulist = fits.HDUList([prihdu, tbhdu])

        os.makedirs(self.output_dir, exist_ok=True)
        filename = join(self.output_dir, "input_spectrum.fits")
        thdulist.writeto(filename, overwrite=True)
        return filename

    def execute_recipe(self, recipe, rc_fname, sof_fname):
        logfile = f"{recipe}.log"
        os.makedirs(self.output_dir, exist_ok=True)

        # Run molecfit with esorex
        esorex_options = [
            f"--log-dir={self.output_dir}",
            f"--log-file={logfile}",
            f"--output-dir={self.output_dir}",
            f"--recipe-config={rc_fname}",
        ]

        stdout = self.esorex(recipe, esorex_options=esorex_options, sof=sof_fname)

        # Find the produced filenames in the output
        products = re.findall(r"Created product ([^\s]*)", stdout)
        success = len(products) != 0
        result = {"success": success, "stdout": stdout, "products": {}}
        for p in products:
            key = splitext(basename(p))[0].lower()
            result["products"][key] = p
        return result

    def molecfit_model(self, science, wave_include=None):
        if wave_include is None:
            # If no wave include value is given, use the entire wavelength range
            hdu = fits.open(science)
            # TODO: which extensions?
            for i in range(1, len(hdu)):
                wave = hdu[1].data[self.column_wave]
                wmin, wmax = np.nanmin(wave), np.nanmax(wave)
                wmin, wmax = wmin * self.wlg_to_micron, wmax * self.wlg_to_micron
                wave_include = [f"{wmin},{wmax}"]
            wave_include = ",".join(wave_include)

        with NamedTemporaryFile("w", suffix=".sof") as sof_file, NamedTemporaryFile(
            "w", suffix=".rc"
        ) as rc_file:
            sof_fname = sof_file.name
            self.prepare_sof(sof_fname, [(science, "SCIENCE"),])

            # TODO: most of these should be parameters of self
            rc_fname = rc_file.name
            self.prepare_rc(
                rc_fname,
                {
                    "WAVE_INCLUDE": wave_include,
                    "LIST_MOLEC": "H2O,O3,O2,CO2,CH4,N2O",
                    "FIT_MOLEC": "1,1,1,1,1,1",
                    "REL_COL": "0.422,2.808,1.2,1.012,1.808,0.809",
                    "COLUMN_LAMBDA": self.column_wave,
                    "COLUMN_FLUX": self.column_flux,
                    "COLUMN_DFLUX": self.column_err,
                    "WLG_TO_MICRON": self.wlg_to_micron,
                    "WAVELENGTH_FRAME": "VAC",
                    "FIT_CONTINUUM": "0",
                    "SILENT_EXTERNAL_BINS": "FALSE",
                },
            )

            result = self.execute_recipe("molecfit_model", rc_fname, sof_fname)

        return result

    def molecfit_calctrans(
        self, science, atm_parameters, model_molecules, best_fit_parameters
    ):
        with NamedTemporaryFile("w", suffix=".sof") as sof_file, NamedTemporaryFile(
            "w", suffix=".rc"
        ) as rc_file:
            sof_fname = sof_file.name
            self.prepare_sof(
                sof_fname,
                [
                    (science, "SCIENCE"),
                    (atm_parameters, "ATM_PARAMETERS"),
                    (model_molecules, "MODEL_MOLECULES"),
                    (best_fit_parameters, "BEST_FIT_PARAMETERS"),
                ],
            )
            rc_fname = rc_file.name
            self.prepare_rc(
                rc_fname,
                {
                    "CALCTRANS_MAPPING_KERNEL": "1,1",
                    "MAPPING_ATMOSPHERIC": "1,1",
                    "MAPPING_CONVOLVE": "1,1",
                    "USE_INPUT_KERNEL": "FALSE",
                },
            )

            result = self.execute_recipe("molecfit_calctrans", rc_fname, sof_fname)

        return result

mf = Molecfit(
    recipe_dir="/home/ansgar/ESO/esoreflex/lib/esopipes-plugins",
    column_wave="WAVE",
    column_flux="FLUX",
    column_err="ERR",
    wlg_to_micron=0.0001,
)
recipes = mf.get_recipes()
params = mf.get_default_parameters("molecfit_calctrans")


input_file = "/home/ansgar/Documents/Python/sme/examples/paper/data/ADP.2014-10-02T10_02_04.297.fits"
hdu = fits.open(input_file)
header = hdu[0].header
flux = hdu[1].data["FLUX"][0]
wave = hdu[1].data["WAVE"][0]

nsteps = 10000
wave = [
    wave[nsteps * i : nsteps * (i + 1)] for i in range(int(np.ceil(len(wave) / nsteps)))
]
flux = [
    flux[nsteps * i : nsteps * (i + 1)] for i in range(int(np.ceil(len(flux) / nsteps)))
]

# Take the last element since there is lots of tellurics to deal with
wave = wave[-1]
flux = flux[-1]

# quick normalization
flux /= np.nanpercentile(flux, 99)

# # Use TAPAS for comparison
ftapas = join(dirname(__file__), "tapas.ipac")
dtapas = np.genfromtxt(ftapas, comments="\\", skip_header=36)
wtapas, ftapas = dtapas[:, 0], dtapas[:, 1]
# convert to angstrom
wtapas *= 10
# Normalize
ftapas -= ftapas.min()
ftapas /= ftapas.max()

# Shift the wavelength to match the unshifted tapas wavelengths
rv = -103
c_light = const.c.to_value("km/s")
rv_factor = np.sqrt((1 - rv / c_light) / (1 + rv / c_light))
wave *= rv_factor

# plt.plot(wave, flux)
# plt.plot(wtapas, ftapas)
# plt.show()

# Step 1:
# Since we modifed the flux and wavelength we need to write the data to a new datafile
input_file = mf.prepare_fits(header, wave, flux)
wave_range = f"{wave[0] * 0.0001},{wave[-1] * 0.0001}"
output = mf.molecfit_model(input_file, wave_include=wave_range)

# Step 2:
products = output["products"]
atm_parameters = products["atm_parameters"]
model_molecules = products["model_molecules"]
best_fit_parameters = products["best_fit_parameters"]
output = mf.molecfit_calctrans(
    input_file, atm_parameters, model_molecules, best_fit_parameters
)

lblrtm_results = output["products"]["lblrtm_results"]
hdu = fits.open(lblrtm_results)
wmf = hdu[1].data["lambda"] * 10000
fmf = hdu[1].data["flux"]

plt.plot(wave, flux, label="Observation")
plt.plot(wmf, fmf, label="Molecfit")
plt.plot(wtapas, ftapas, label="Tapas")
plt.xlim(wave[0], wave[-1])
plt.legend()
plt.show()

pass
