from os.path import dirname, join
import numpy as np
from astropy.io import fits
from astropy import constants as const
from molecfit_wrapper.molecfit import Molecfit

if __name__ ==  "__main__":
    mf = Molecfit(
        recipe_dir="/home/ansgar/ESO/esoreflex/lib/esopipes-plugins",
        column_wave="WAVE",
        column_flux="FLUX",
        column_err="ERR",
        wlg_to_micron=0.0001,
    )

    # Read the input file
    input_file = "/home/ansgar/Documents/Python/sme/examples/paper/data/ADP.2014-10-02T10_02_04.297.fits"
    hdu = fits.open(input_file)
    header = hdu[0].header
    flux = hdu[1].data["FLUX"][0]
    wave = hdu[1].data["WAVE"][0]

    # Split it into smaller blocks for testing
    stepsize = 10000
    nsteps = int(np.ceil(len(wave) / stepsize))
    wave = [
        wave[stepsize * i : stepsize * (i + 1)] for i in range(nsteps)
    ]
    flux = [
        flux[stepsize * i : stepsize * (i + 1)] for i in range(nsteps)
    ]

    # Take the last element since there is lots of tellurics to deal with
    # wave = wave[-1]
    # flux = flux[-1]

    # quick normalization
    flux = [f / np.nanpercentile(f, 99) for f in flux]

    # Shift the wavelength to match the unshifted tapas wavelengths
    # TODO: where does this value come from? radial velocity of the star is only 16 km/s
    rv = -103
    c_light = const.c.to_value("km/s")
    rv_factor = np.sqrt((1 - rv / c_light) / (1 + rv / c_light))
    wave = [w * rv_factor for w in wave]

    # Step 1:
    # Since we modifed the flux and wavelength we need to write the data to a new datafile
    input_file = mf.prepare_fits(header, wave, flux)
    output_model = mf.molecfit_model(input_file)

    # Step 2:
    products = output_model["products"]
    atm_parameters = products["atm_parameters"]
    model_molecules = products["model_molecules"]
    best_fit_parameters = products["best_fit_parameters"]
    output_calctrans = mf.molecfit_calctrans(
        input_file, atm_parameters, model_molecules, best_fit_parameters
    )

    # Plot results
    # Use TAPAS for comparison
    ftapas = join(dirname(__file__), "tapas.ipac")
    dtapas = np.genfromtxt(ftapas, comments="\\", skip_header=36)
    wtapas, ftapas = dtapas[:, 0], dtapas[:, 1]
    # convert to angstrom
    wtapas *= 10
    # Normalize
    ftapas -= ftapas.min()
    ftapas /= ftapas.max()

    lblrtm_results = output_calctrans["products"]["lblrtm_results"]
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
