import subprocess
from os.path import dirname, join
import copy

import numpy as np
from astropy.io import fits
from astropy import constants as const

RECIPE_DIR = "/home/ansgar/ESO/esoreflex/lib/esopipes-plugins"
COLUMN_WAVE = "WAVE"
COLUMN_FLUX = "FLUX"
COLUMN_ERR = "ERR"


def execute_esorex(commands):
    result = subprocess.run(["esorex", *commands], capture_output=True)
    stdout = result.stdout.decode()
    return stdout


def execute_molecfit(command, parameter_file, sof_file, recipe_dir=RECIPE_DIR):
    commands = []
    if recipe_dir is not None:
        commands += [f"--recipe-dir={recipe_dir}"]
    if parameter_file is not None:
        commands += [f"--recipe-config={parameter_file}"]
    if command is not None:
        commands += [command]
    if sof_file is not None:
        commands += [sof_file]
    result = execute_esorex(commands)
    return result


def get_recipes(recipe_dir=RECIPE_DIR):
    # This is really awkward but it works, assuming nothing changes
    stdout = execute_molecfit("--recipes", None, None, recipe_dir=recipe_dir)
    lines = stdout.split("\n")
    # Keep only the lines with recipe names and info
    lines = lines[5:-3]
    lines = [l[2:] for l in lines]

    recipes = {}
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
            recipes[name] = entry
            name, entry = line.split(":", 1)
            name = name.strip()
            continue
    recipes[name] = entry

    return recipes


def create_sof(files, sof_fname=None):
    if sof_fname is None:
        sof_fname = join(dirname(__file__), "sof.sof")
    content = []
    for fname, ftype in files.items():
        content += [f"{fname} {ftype}\n"]

    with open(sof_fname, "w") as f:
        f.writelines(content)
    return sof_fname


def create_rc(params, rc_fname=None):
    if rc_fname is None:
        rc_fname = join(dirname(__file__), "params.rc")

    content = []
    for key, value in params.items():
        content += [f"{key} = {value}\n"]

    with open(rc_fname, "w") as f:
        f.writelines(content)

    return rc_fname


def create_fits(output_name, headers, wave, spectra, mode="HARPS"):
    """This is a wrapper for writing a spectrum from a list to molecfit format.
    name is the filename of the fits file that is the output.
    headers is the list of astropy header objects associated with the list of spectra
    in the spectra variable. ii is the number from that list that needs to be written.

    The wave keyword is set for when the s1d headers do not contain wavelength information like HARPS does.
    (for instance, ESPRESSO). The wave keyword needs to be set in this case, to the wavelength array as extracted from FITS files or smth.
    If you do that for HARPS and set the wave keyword, this code will still grab it from the header, and overwrite it. So dont bother.
    """
    spectrum = spectra
    npx = len(spectrum)
    speed_of_light = const.c.to_value("km/s")

    if mode == "HARPS":
        bervkeyword = "HIERARCH ESO DRS BERV"
        # Need to un-correct the s1d spectra to go back to the frame of the Earths atmosphere.
        berv = headers[bervkeyword]
        wave *= 1.0 - berv / speed_of_light
    elif mode == "HARPSN":
        bervkeyword = "HIERARCH TNG DRS BERV"
        # Need to un-correct the s1d spectra to go back to the frame of the Earths atmosphere.
        berv = headers[bervkeyword]
        wave *= 1.0 - berv / speed_of_light
    elif mode == "ESPRESSO":
        # WAVE VARIABLE NEEDS TO BE PASSED NOW.
        bervkeyword = "HIERARCH ESO QC BERV"
        # Need to un-correct the s1d spectra to go back to the frame of the Earths atmosphere.
        berv = headers[bervkeyword]
        wave *= 1.0 - berv / speed_of_light

    # at the end, when the transmission spectrum is corrected, we stay in the barycentric frame because these will be used to
    # correct the e2ds spectra which are not yet berv-corrected.
    err = np.sqrt(spectrum)

    # Write out the s1d spectrum in a format that molecfit eats.
    # This is a fits file with an empty primary extension that contains the header of the original s1d file.
    # Plus an extension that contains a binary table with 3 columns.
    # The names of these columns need to be indicated in the molecfit parameter file,
    # as well as the name of the file itself. This is currently hardcoded.
    col1 = fits.Column(name="WAVE", format="1D", array=wave)
    col2 = fits.Column(name="FLUX", format="1D", array=spectrum)
    col3 = fits.Column(name="ERR", format="1D", array=err)
    cols = fits.ColDefs([col1, col2, col3])
    tbhdu = fits.BinTableHDU.from_columns(cols)
    prihdr = copy.deepcopy(headers)
    prihdu = fits.PrimaryHDU(header=prihdr)
    thdulist = fits.HDUList([prihdu, tbhdu])
    thdulist.writeto(output_name, overwrite=True)
    return output_name


def molecfit_model(input_file, wmin, wmax):
    sof_fname = create_sof({input_file: "SCIENCE"})
    rc_fname = create_rc(
        {
            "WAVE_INCLUDE": f"{wmin},{wmax}",
            "LIST_MOLEC": "H20,CO2,O3,CO,OCS",
            "FIT_MOLEC": "1,1,1,1,0",
            "REL_COL": "0.422,1.012,2.808,0.809",
            "COLUMN_LAMBDA": COLUMN_WAVE,
            "COLUMN_FLUX": COLUMN_FLUX,
            "COLUMN_DFLUX": COLUMN_ERR,
            "WLG_TO_MICRON": "0.01",
            "WAVELENGTH_FRAME": "VAC",
            "FIT_CONTINUUM": "1",
            "CONTINUUM_N": "3",
        }
    )
    result = execute_molecfit("molecfit_model", rc_fname, sof_fname)

    # TODO: get filenames from output
    # TODO: return absolute paths
    result = {
        "stdout": result,
        "model_molecules": "MODEL_MOLECULES.fits",
        "wave_include": "WAVE_INCLUDE.fits",
        "atm_parameters": "ATM_PARAMETERS.fits",
        "best_fit_parameters": "BEST_FIT_PARAMETERS.fits",
        "best_fit_model": "BEST_FIT_MODEL.fits",
    }

    return result


print(get_recipes())

input_file = "/home/ansgar/Documents/Python/sme/examples/paper/data/ADP.2014-10-02T10_02_04.297.fits"
hdu = fits.open(input_file)
header = hdu[0].header
flux = hdu[1].data["FLUX"][0]
wave = hdu[1].data["WAVE"][0]
wmin, wmax = wave[0], wave[-1]
wmin, wmax = wmin / 1000, wmax / 1000

# This is only neces
# input_file = join(dirname(__file__), "spectra.fits")
# create_fits(input_file, header, wave, flux)

output = molecfit_model(input_file, wmin, wmax)

pass
