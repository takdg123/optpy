import numpy as np
import matplotlib.pyplot as plt
from . import const
from . import utils
import astropy.units as u
from astropy.table import Table

from . import fit

def set_flux_axis(flux_unit, ax):
    if flux_unit == "AB":
        ax.invert_yaxis()
        ax.set_ylabel(r"AB", fontsize=15)
        
    elif flux_unit == "Jy":
        ax.set_yscale("log")
        ax.set_ylabel(r"Flux [Jy]", fontsize=15)
        
    elif flux_unit == "e2dnde":
        ax.set_yscale("log")
        ax.set_ylabel(r"Flux [erg/cm$^2$/s]", fontsize=15)
        
    return ax

def set_energy_axis(energy_unit, ax):
    if energy_unit == "lambda":
        ax.set_xlabel(r"Wavelength [A]", fontsize=15)

    elif energy_unit == "Hz":
        ax.set_xlabel(r"Frequency [Hz]", fontsize=15)

    elif energy_unit == "eV":
        ax.set_xlabel(r"Energy [eV]", fontsize=15)
    
    ax.set_xscale("log")
    return ax

def plot_lightcurve(table, show_plot = True, flux_unit="Jy", time_unit="sec", ax = None, t_shift=0, do_fit=False, **kwargs):
    
    t_max = kwargs.pop("t_max", None)
    num_pts = kwargs.pop("num_pts", None)

    if ax is None:
        ax = plt.gca()
    
    ax = set_flux_axis(flux_unit, ax)
    
    flags = (~np.isnan(table[flux_unit]))*(table[flux_unit]>0)
    table = table[flags]

    ls = kwargs.pop("ls", ":")

    filt = np.atleast_1d(kwargs.pop("filter", np.unique(table["filter"])))

    props = []

    for f in filt:
        lc = table[table["filter"]==f]
        lc["time"] += t_shift

        if time_unit == "day":
            t_conv = const.sec2day
        else:
            t_conv = 1
        
        if len(lc)>=4 and (flux_unit == "e2dnde" or flux_unit == "Jy") and do_fit:
            
            if t_max is not None:
                mask = lc["time"]<t_max
            elif num_pts is not None:
                mask = [False] * len(lc)
                mask[:num_pts] = [True] * num_pts
            else:
                mask = [True] * len(lc)

            fit_result, l = fit.temporal_fit(lc, t_conv=t_conv, return_fit=True, flux_unit=flux_unit, mask=mask, num_pts=num_pts)
        else:
            l = None
        
        if l is not None:
            prop = ax.errorbar(lc["time"]*t_conv, lc[flux_unit], 
             yerr=lc[f"{flux_unit}_err"], ls=ls, marker="+", color=kwargs.pop("color", lc["color"][0]),
             label=r"{} ($\alpha$ = {:.2f}$\pm${:.2f})".format(f, fit_result["alpha"][0], fit_result["alpha_err"][0]), **kwargs)
            ax.plot(lc["time"][mask]*t_conv, l.model(lc["time"][mask]*t_conv), color=prop[0].get_color(), ls="-", lw=0.5)
        
        else:
            prop = ax.errorbar(lc["time"]*t_conv, lc[flux_unit], color=kwargs.pop("color", lc["color"][0]), 
                           yerr=lc[f"{flux_unit}_err"], ls=ls, marker="+", label="{}".format(kwargs.pop("label", f)), **kwargs)
        props.append(prop)
    
    ax.set_xlabel(f"Time since trigger [{time_unit}]", fontsize=15)
    ax.set_xscale("log")

    ax.legend(ncols=2, fontsize=8)
    ax.grid(which="major")
    ax.grid(which="minor", ls=":", alpha=0.5)

    return props

def plot_spectrum(table, model="MW", show_plot=True, flux_unit = "Jy", energy_unit = "Hz", ax=None, color=None, z=None, **kwargs):
    
    if ax is None:
        ax = plt.gca()

    if color is None:
        clist = utils.make_clist(max(table["t_index"])+1, palette='viridis')
    else:
        clist = color

    ax = set_flux_axis(flux_unit, ax)
    ax = set_energy_axis(energy_unit, ax)

    flags = (~np.isnan(table[flux_unit]))*(table[flux_unit]>0)
    table = table[flags]

    for i in np.unique(table["t_index"]):
        spec = table[table["t_index"] == i]
        spec.sort(energy_unit)

        if spec["time"].unit == u.second:
            t_ave = np.average(spec["time"])*const.sec2day
        else:
            t_ave = np.average(spec["time"])
        
        if len(spec)>=4 and flux_unit=="Jy" and energy_unit=="Hz":

            fit_result, l = fit.spectral_fit(spec, z=z, model=model, Av = kwargs.pop("Av", 0.15), return_fit=True)
            
            prop = ax.errorbar(spec[energy_unit], spec[flux_unit], 
                               xerr=spec[f"{energy_unit}_err"],
                               yerr=spec[f"{flux_unit}_err"], ls=":", color=clist[i],
                               marker="+", label=r"{:.1f} days, $\beta$ ={:.2f}$\pm${:.2f}".format(t_ave,fit_result["beta"][0], fit_result["beta_err"][0]))
            ax.plot(spec[energy_unit], l.model(spec[energy_unit], *l.p), color=prop[0].get_color(), ls="-", lw=0.5)
        elif len(spec)>1:
        
            prop = ax.errorbar(spec[energy_unit], spec[flux_unit], 
                               xerr=spec[f"{energy_unit}_err"],
                               yerr=spec[f"{flux_unit}_err"], ls=":", color=clist[i],
                               marker="+", label="{:.1f} days".format(t_ave))
    ax.set_xscale("log")
    ax.legend(ncols=2, fontsize=8)
    ax.grid(which="major")
    ax.grid(which="minor", ls=":", alpha=0.5)

