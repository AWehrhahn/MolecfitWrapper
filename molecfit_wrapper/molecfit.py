# -*- coding: utf-8 -*-
import configparser
import copy
import os
import re
import unicodedata
import subprocess
from os import getcwd
from os.path import basename, dirname, join, realpath, splitext
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
from astropy import constants as const
from astropy.io import fits


class RecipeConfig(dict):
    """
    This class handles the configuration parameters settings file for esorex recipes.
    The files can be created using ´´´esorex --create-config=<filename> <recipe-name>```
    """

    def __init__(self, parameters=None, replacement=None):
        super().__init__()
        if parameters is not None:
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

    def unparse_value(self, value, inverse):
        if isinstance(value, (list, tuple)):
            list_inverse = {True: "1", False: "0"}
            value = [self.unparse_value(v, list_inverse) for v in value]
            value = ",".join(value)
        elif value in inverse.keys():
            value = inverse[value]
        else:
            value = str(value)
        return value

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
            value = self.unparse_value(value, inverse)
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
        params = self.copy()
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
            data = []
        #:list: content of the SOF file
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

    def __init__(self, esorex_exec=None, recipe_dir=None, output_dir=None):
        if esorex_exec is None:
            exorex_exec = "esorex"
        #:str: executable to call, by default "esorex"
        self.esorex_exec = esorex_exec
        if recipe_dir is not None:
            recipe_dir = realpath(recipe_dir)
        #:str: location of recipe files, overrides the --recipe-dir option and the environment variable when set
        self.recipe_dir = recipe_dir
        if output_dir is not None:
            output_dir = realpath(output_dir)
        #:str: location to place the output files. Note that the way esorex operates it will create the output files in the working directory and then copy them to the specified location at the end of the recipe
        self.output_dir = output_dir
        #:dict: available recipes
        self.recipes = self.get_recipes()
        #:dict: default parameters for each recipe
        self.parameters = {r: self.get_default_parameters(r) for r in self.recipes}

    def slugify(self, value, allow_unicode=False):
        """
        Taken from https://github.com/django/django/blob/master/django/utils/text.py
        Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
        dashes to single dashes. Remove characters that aren't alphanumerics,
        underscores, or hyphens. Convert to lowercase. Also strip leading and
        trailing whitespace, dashes, and underscores.
        """
        value = str(value)
        if allow_unicode:
            value = unicodedata.normalize("NFKC", value)
        else:
            value = (
                unicodedata.normalize("NFKD", value)
                .encode("ascii", "ignore")
                .decode("ascii")
            )
        value = re.sub(r"[^\w\s-]", "", value.lower())
        value = re.sub(r"[-\s]+", "-", value).strip("-_")
        return value

    def esorex(self, recipe, esorex_options=(), recipe_options=(), sof=""):
        """
        Main method to call esorex from Python
        equivalent to: 'esorex (esorex_options) recipe (recipe_options) sof'

        Parameters
        ----------
        recipe : str
            Name of the recipe to call, should be one of self.recipes
        esorex_options : tuple, optional
            options to send to esorex, not the recipe, by default ()
        recipe_options : tuple, optional
            options to send to the recipe, not esorex, by default ()
        sof : str, optional
            sof filename to send to esorex, by default ""

        Returns
        -------
        stdout : str
            the captured standard output
        """
        if isinstance(esorex_options, str):
            esorex_options = (esorex_options,)
        if isinstance(recipe_options, str):
            recipe_options = (recipe_options,)

        # Add fixed options from self
        if self.recipe_dir is not None:
            esorex_options = (f"--recipe-dir={self.recipe_dir}", *esorex_options)
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            esorex_options = (
                f"--output-dir={self.output_dir}",
                f"--log-dir={self.output_dir}",
                *esorex_options,
            )
        # Add logfile name
        logfile = self.slugify(recipe)
        esorex_options = (f"--log-file={logfile}.log", *esorex_options)
        # Assemble the command line arguments
        command = ["esorex", *esorex_options, recipe, *recipe_options, sof]
        result = subprocess.run(command, capture_output=True)
        # The only output we get from esorex is the console output
        stdout = result.stdout.decode()
        return stdout

    def get_default_parameters(self, recipe):
        """
        Retrieve the default parameters for one recipe

        Parameters
        ----------
        recipe : str
            Name of the recipe

        Returns
        -------
        config : RecipeConfig
            Default configuration for this recipe
        """
        with NamedTemporaryFile("r", suffix=".rc") as ntf:
            esorex_options = (f"--create-config={ntf.name}",)
            stdout = self.esorex(recipe, esorex_options=esorex_options)
            config = RecipeConfig.read(ntf.name)
        return config

    def get_recipes(self):
        """
        Get the names and descriptions of all available recipes

        Returns
        -------
        recipes : dict
            Recipe names as keys and descriptions as items
        """
        stdout = self.esorex("--recipes")
        lines = stdout.split("\n")
        # TODO: this is all kind of clunky but it works?
        # Keep only the lines with recipe names and info
        lines = lines[5:-3]
        lines = [l[2:] for l in lines]

        self.recipes = {}
        if len(lines) == 0:
            # There probably is some issue with the recipe dir
            return self.recipes

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
    """
    A wrapper around the esorex interface of molecfit
    
    For documentation on Molecfit see: https://www.eso.org/sci/software/pipelines/skytools/molecfit
    """

    def __init__(self, esorex_exec=None, recipe_dir=None, output_dir=None, **kwargs):
        super().__init__(
            esorex_exec=esorex_exec, recipe_dir=recipe_dir, output_dir=output_dir
        )
        #:str: The filename used by prepare_fits
        self.spectrum_filename = "input_spectrum.fits"

        # Recipe 'molecfit_model' default parameters
        #:str: column name for the flux uncertainties
        kwargs.setdefault("column_dflux", "error")
        kwargs.setdefault("silent_external_bins", False)
        #:list: list of molecules to fit
        kwargs.setdefault("list_molec", ["H2O", "O3", "O2", "CO2", "CH4", "N2O"])
        #:list: list with *initial* relative abundacnces of the gasses
        kwargs.setdefault("rel_molec", [1.0 for _ in kwargs["list_molec"]])
        #:list: list of flags whether to fit this molecule or not, same order as list_molec
        kwargs.setdefault("fit_molec", [True for _ in kwargs["list_molec"]])

        for recipe in self.recipes.keys():
            available_parameters = self.parameters[recipe].keys()
            for key, value in kwargs.items():
                if key in available_parameters:
                    self.parameters[recipe][key] = value

    def prepare_sof(self, filename, data):
        """Create a new sof with the given data

        Parameters
        ----------
        filename : str
            name of the file to create
        data : list
            sof file data, see SofFile for more details

        Returns
        -------
        sof : SofFile
            the created and written file
        """
        sof = SofFile(data)
        sof.write(filename)
        return sof

    def prepare_rc(self, filename, data):
        """ Create a recipe configuration file

        Parameters
        ----------
        filename : str
            Name of the file to create
        data : dict
            the recipe configuration, see RecipeConfig for details

        Returns
        -------
        config : RecipeConfig
            the recipe configuration
        """
        config = RecipeConfig(parameters=data)
        config.write(filename)
        return config

    def prepare_fits(self, header, wave, flux, err=None):
        """ Create a new fits file that can be read by Molecfit
        The new file is created in the self.output_dir directory

        Parameters
        ----------
        header : fits.Header
            fits header of the original file, contains all the keywords
            that are used by Molecfit
        wave : array
            wavelength array
        flux : array
            flux (spectrum) array
        err : array, optional
            flux (spectrum) uncertainties, if not set, we use the
            square root of flux, by default None

        Returns
        -------
        filename : str
            the name of the new fits file
        """
        if err is None:
            err = [np.sqrt(f) for f in flux]
        nseg = len(wave)

        prihdr = copy.deepcopy(header)
        prihdu = fits.PrimaryHDU(header=prihdr)
        thdulist = [prihdu]

        for i in range(nseg):
            col1 = fits.Column(
                name=self.parameters["molecfit_model"]["column_lambda"],
                format="1D",
                array=wave[i],
            )
            col2 = fits.Column(
                name=self.parameters["molecfit_model"]["column_flux"],
                format="1D",
                array=flux[i],
            )
            col3 = fits.Column(
                name=self.parameters["molecfit_model"]["column_dflux"],
                format="1D",
                array=err[i],
            )
            cols = fits.ColDefs([col1, col2, col3])
            tbhdu = fits.BinTableHDU.from_columns(cols)
            thdulist += [tbhdu]

        thdulist = fits.HDUList(thdulist)
        os.makedirs(self.output_dir, exist_ok=True)
        filename = join(self.output_dir, self.spectrum_filename)
        thdulist.writeto(filename, overwrite=True)
        return filename

    def execute_recipe(self, recipe, rc_fname, sof_fname):
        """
        Call a set recipe with an existing configuration and sof

        Parameters
        ----------
        recipe : str
            recipe name
        rc_fname : str
            filename of the parameter configuration file
        sof_fname : str
            filename of the sof file

        Returns
        -------
        result : dict
            dict containing the results of the recipe
        """
        logfile = f"{recipe}.log"
        os.makedirs(self.output_dir, exist_ok=True)

        # Run molecfit with esorex
        esorex_options = [
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

        with open(join(self.output_dir, logfile)) as f:
            result["log"] = f.read()
        return result

    def molecfit_model(self, science, wave_include=None):
        """
        Run the molecfit_model recipe on the science file

        Parameters
        ----------
        science : str
            Name of the input science file
        wave_include : str, optional
            the wavelength range(s) to include in the format specified by molecfit.
            I.e. 'wmin[0],wmax[0],wmin[1],wmax[1],...'. If None the full range
            specied in the science file is used. By default None

        Returns
        -------
        result : dict
            Dictionary with the recipe results
        """
        params = self.parameters["molecfit_model"]
        if wave_include is None:
            # If no wave include value is given, use the entire wavelength range
            with fits.open(science) as hdu:
                # TODO: which extensions?
                wave_include = []
                for i in range(1, len(hdu)):
                    wave = hdu[i].data[params["column_lambda"]]
                    wmin, wmax = np.nanmin(wave), np.nanmax(wave)
                    wmin, wmax = (
                        wmin * params["wlg_to_micron"],
                        wmax * params["wlg_to_micron"],
                    )
                    wave_include += [f"{wmin:.5},{wmax:.5}"]
                wave_include = ",".join(wave_include)

        with NamedTemporaryFile("w", suffix=".sof") as sof_file, NamedTemporaryFile(
            "w", suffix=".rc"
        ) as rc_file:
            sof_fname = sof_file.name
            self.prepare_sof(sof_fname, [(science, "SCIENCE"),])

            # TODO: most of these should be parameters of self
            nseg = len(wave_include.split(",")) // 2
            rc_fname = rc_file.name
            # TODO: we would like to define our own temporary directory using TemporaryDirectory
            # But this breaks the next step molecfit_calctrans, as it tries (and fails) to reuse the
            # same directory again (it has since been deleted)
            # We could fix this by keeping the directory around until both are done
            # but this is one level of abstraction higher than this...
            # This expects a seperator '/' at the end...
            # "TMP_PATH": work_dir + os.sep,
            self.parameters["molecfit_model"]["wave_include"] = wave_include
            self.parameters["molecfit_model"]["map_regions_to_chip"] = [
                i for i in range(1, nseg + 1)
            ]
            self.prepare_rc(rc_fname, self.parameters["molecfit_model"])

            result = self.execute_recipe("molecfit_model", rc_fname, sof_fname)

        return result

    def molecfit_calctrans(
        self, science, atm_parameters, model_molecules, best_fit_parameters
    ):
        """
        Execute the molecfit_calctrans recipe for the input files

        Parameters
        ----------
        science : str
            filename of the science file
        atm_parameters : str
            filename of the atm_parameters file created by molecfit_model
        model_molecules : str
            filename of the model molecules file created by molecfit_model
        best_fit_parameters : str
            filename of the best fit parameters file created by molecfit_model

        Returns
        -------
        result : dict
            Dictionary with the recipe results
        """
        # each extension is one segment, except for the primary
        with fits.open(science) as hdu:
            nseg = len(hdu)
        mapping = [i for i in range(1, nseg)]
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
            self.parameters["molecfit_calctrans"]["calctrans_mapping_kernel"] = mapping
            self.parameters["molecfit_calctrans"]["mapping_atmospheric"] = mapping
            self.parameters["molecfit_calctrans"]["mapping_convolve"] = mapping
            self.parameters["molecfit_calctrans"]["use_input_kernel"] = False
            self.prepare_rc(rc_fname, self.parameters["molecfit_calctrans"])

            result = self.execute_recipe("molecfit_calctrans", rc_fname, sof_fname)

        return result

