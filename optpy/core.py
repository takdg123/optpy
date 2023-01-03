import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

import astropy.constants as c
import astropy.units as u

from . import const
from . import utils 

from grbpy.utils import UTC2MET

from .extinction import galactic_extinction

from astropy.io import ascii as asc

def read_data(data, full_output = False, t_shift = 0, 
		apply_extinction=True, **kwargs):

	optical_data = np.atleast_1d(data)

	load_table = Table(dtype=const.tab_dtype)
	for filename in optical_data:
	    if "lsgt" not in filename:
	        tab = asc.read(filename)
	        for row in tab:
	            load_table.add_row(row)

	if full_output:
		table = load_table
	else:
		table = Table([load_table["filter"],
		load_table["date-obs"],
		load_table["mag"], load_table["magerr"],
		load_table["ul_3sig"]])


	table.add_column(UTC2MET(table["date-obs"])+t_shift, name="time")

	idx = utils.make_tbins(table["time"], ttol=kwargs.pop("ttol", 0.05))

	table.add_column(idx, name="t_index")
	
	add_info = [list(const.filter_dict[f].values()) for f in table["filter"]]
	table.add_columns(list(zip(*add_info)), names=["lam", "bdw", "color"])
	table["bdw"] /= 2

	table["AB"] = table["mag"]

	if apply_extinction:
	    for f in set(table["filter"]):
	        extinc = galactic_extinction(f)
	        if (extinc is not None):
	            table["AB"][table["filter"]==f] -= extinc["The Galactic extinction"]
	
	table["AB_err"] = table["magerr"]
	table["AB"] *= u.AB
	table["AB_err"] *= u.AB

	conv = np.asarray([utils.AB2Flux(data["AB"], data["AB_err"], units="Jy") \
		for data in table])
	table["Jy"] = conv[:,0]
	table["Jy_err"] = conv[:,1]
	table["Jy"] *= u.Jy
	table["Jy_err"] *= u.Jy
	
	conv = np.asarray([utils.AB2Flux(data["AB"], data["AB_err"], units="cgs") \
		for data in table])
	table["e2dnde"] = conv[:,0] * utils.lam2nu(table["lam"]).value
	table["e2dnde_err"] = conv[:,1] * utils.lam2nu(table["lam"]).value
	table["e2dnde"] *= u.erg/u.cm**2/u.second
	table["e2dnde_err"] *= u.erg/u.cm**2/u.second

	table["time"] *= u.second
	table["time"].format = "10.1f"

	table["lam"].name = "lambda"
	table["bdw"].name = "lambda_err"
	table["lambda"] *= u.Angstrom
	table["lambda_err"] *= u.Angstrom

	table["Hz_err"] = abs(utils.lam2nu(table["lambda"]+table["lambda_err"]) -  utils.lam2nu(table["lambda"]))
	table["Hz"] = utils.lam2nu(table["lambda"])
	
	table["eV_err"] = abs(utils.lam2eV(table["lambda"]+table["lambda_err"]) -  utils.lam2eV(table["lambda"]))
	table["eV"] = utils.lam2eV(table["lambda"])

	return table
