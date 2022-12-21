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
		time_units = "day",
		energy_units = "A", 
		apply_extinction=True,
		flux_units="AB", **kwargs):

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


	table.add_column(UTC2MET(table["date-obs"])-t_shift, name="time")

	idx = utils.make_tbins(table["time"], ttol=kwargs.pop("ttol", 0.05))

	table.add_column(idx, name="t_index")
	
	add_info = [list(const.filter_dict[f].values()) for f in table["filter"]]
	table.add_columns(list(zip(*add_info)), names=["lam", "bdw", "color"])
	table["bdw"] /= 2
    
	if apply_extinction:
	    for f in set(table["filter"]):
	        extinc = galactic_extinction(f)
	        if (extinc is not None):
	            table["mag"][table["filter"]==f] -= extinc["The Galactic extinction"]

	if flux_units == "flux":
	    conv = np.asarray([utils.AB2Flux(data["mag"], data["magerr"], units="Jy") \
	                       for data in table])
	    table["mag"] = conv[:,0]
	    table["magerr"] = conv[:,1]

	    table["mag"].name = "flux"
	    table["magerr"].name = "flux_err"

	    table["flux"] *= u.Jy
	    table["flux_err"] *= u.Jy

	elif flux_units == "e2dnde":
		conv = np.asarray([utils.AB2Flux(data["mag"], data["magerr"], units="cgs") \
		                   for data in table])
		table["mag"] = conv[:,0]
		table["magerr"] = conv[:,1]
		table["mag"] *= utils.lam2nu(table["lam"]).value
		table["magerr"] *= utils.lam2nu(table["lam"]).value
		table["mag"].name = "e2dnde"
		table["magerr"].name = "e2dnde_err"
		table["e2dnde"] *= u.erg/u.cm**2/u.second
		table["e2dnde_err"] *= u.erg/u.cm**2/u.second
	else:
		table["mag"].name = "AB"
		table["magerr"].name = "AB_err"
		table["AB"] *= u.AB
		table["AB_err"] *= u.AB

	if time_units == "day":
		table["time"] = table["time"] / const.sec2day *u.day
		table["time"].format = "10.3f"
	else:	
		table["time"] *= u.second
		table["time"].format = "10.1f"

	if energy_units == "A":
		table["lam"].name = "wavelength"
		table["bdw"].name = "wavelength_err"
		table["wavelength"] *= u.Angstrom
		table["wavelength_err"] *= u.Angstrom

	elif energy_units == "Hz":
		table["bdw"] = abs(utils.lam2nu(table["lam"]+table["bdw"]) -  utils.lam2nu(table["lam"])).to(u.THz)
		table["lam"] = utils.lam2nu(table["lam"]).to(u.THz)
		
		table["lam"].name = "frequency"
		table["bdw"].name = "frequency_err"

	elif energy_units == "eV":
		table["bdw"] = abs(utils.lam2eV(table["lam"]+table["bdw"]) -  utils.lam2eV(table["lam"]))
		table["lam"] = utils.lam2eV(table["lam"])
		
		table["lam"].name = "energy"
		table["bdw"].name = "energy_err"




	return table
