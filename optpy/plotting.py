import numpy as np
import matplotlib.pyplot as plt
from . import const
from . import utils
import astropy.units as u
from .extinction import host_galaxy_extinction

def set_flux_axis(ax, table):
    if "AB" in table.keys():
        unit = "AB"
        ax.invert_yaxis()
        ax.set_ylabel(r"AB", fontsize=15)
    elif "flux" in table.keys():
        unit = "flux"
        ax.set_yscale("log")
        ax.set_ylabel(r"Flux [Jy]", fontsize=15)
        
    elif "e2dnde" in table.keys():
        unit = "e2dnde"
        ax.set_yscale("log")
        ax.set_ylabel(r"Flux [erg/cm$^2$/s]", fontsize=15)
        
    return ax, unit

def set_time_axis(ax, table, time_unit=None):
    if time_unit is None:
        time_unit = table["time"].unit
        t_conv = 1
        ax.set_xlabel(f"Time since trigger [{time_unit}]", fontsize=15)    
    elif time_unit == "day":
        t_conv = table["time"].unit.to(u.day)
        ax.set_xlabel(f"Time since trigger [{u.day}]", fontsize=15)
    elif time_unit == "second":
        t_conv = table["time"].unit.to(u.second)
        ax.set_xlabel(f"Time since trigger [{u.second}]", fontsize=15)
    ax.set_xscale("log")
    return ax, t_conv

def set_energy_axis(ax, table):
    if "A" in table.keys():
        unit = "A"
        ax.set_xscale("log")
        ax.set_xlabel(r"Wavelength [A]", fontsize=15)
    elif "frequency" in table.keys():
        unit = "frequency"
        table[unit] = table[unit].to(u.Hz)
        table[f"{unit}_err"] = table[f"{unit}_err"].to(u.Hz)
        ax.set_xscale("log")
        ax.set_xlabel(r"Frequency [Hz]", fontsize=15)
    elif "e2dnde" in table.keys():
        unit = "energy"
        ax.set_xscale("log")
        ax.set_xlabel(r"Energy [eV]", fontsize=15)
        
    return ax, unit

def plot_lightcurve(table, time_unit = None, ax = None):
    if ax is None:
        ax = plt.gca()
    

    ax, unit = set_flux_axis(ax, table)
    ax, t_conv = set_time_axis(ax, table, time_unit=time_unit)
    flags = (~np.isnan(table[unit]))*(table[unit]>0)
    table = table[flags]

    for f in set(table["filter"]):
        lc = table[table["filter"]==f]
        
        lc = lc
        if len(lc)>=4:
            if unit == "e2dnde" or unit == "flux":
                from grbpy.external.linearfit import LinearFit
                t = np.logspace(np.log10(lc[0]["time"]/t_conv), np.log10(lc[-1]["time"]/t_conv), 100)
                l = LinearFit(lc["time"]/t_conv, lc[unit], y_err= lc[f"{unit}_err"], logx=True, logy=True)
                l.fit(model="linear")
                prop = ax.errorbar(lc["time"]/t_conv, lc[unit], 
                 yerr=lc[f"{unit}_err"], ls=":", marker="x", color=lc["color"][0],
                 label=r"{} ($\alpha$ = {:.2f}$\pm${:.2f})".format(f, -l.p["m"], l.perr["m"]))
                ax.plot(t, l.model(t), color=prop[0].get_color(), ls="-", lw=0.5)

        else:
            prop = ax.errorbar(lc["time"]/t_conv, lc[unit], color=lc["color"][0], 
                               yerr=lc[f"{unit}_err"], ls=":", marker="x", label="{}".format(f))

    #     ax.errorbar(lc["date-obs"][~flags]/utils.sec2day, lc["ul_3sig"][~flags], 
    #                    yerr = lc["ul_3sig"][~flags]*0.5, c=prop[0].get_color(),
    #                    lolims=True, ls="")
        
    ax.legend(ncols=2, fontsize=8)
    ax.grid(which="major")
    ax.grid(which="minor", ls=":", alpha=0.5)


def plot_spectrum(table, ax=None):
    if ax is None:
        ax = plt.gca()
    
    clist = utils.make_clist(max(table["t_index"])+1, palette='viridis')
    ax, f_unit = set_flux_axis(ax, table)
    ax, e_unit = set_energy_axis(ax, table)

    flags = (~np.isnan(table[f_unit]))*(table[f_unit]>0)
    table = table[flags]

    for i, c in zip(np.unique(table["t_index"]), clist):
        spec = table[table["t_index"] == i]
        spec.sort(e_unit)

        if spec["time"].unit == u.second:
            t_ave = np.average(spec["time"])/const.sec2day
        else:
            t_ave = np.average(spec["time"])
        
        if len(spec)>=4 and f_unit=="flux" and e_unit=="frequency":

            from grbpy.external.linearfit import LinearFit

            eta = host_galaxy_extinction("MW", spec["frequency"].to(u.Hz))*0.15/1.086
            
            e = np.logspace(np.log10(spec[0]["frequency"]), np.log10(spec[-1]["frequency"]), 100)

            l = LinearFit(spec["frequency"]/5e14, 
                      spec["flux"]*np.exp(-eta), x_err=spec["frequency_err"]/5e14,
                      y_err=spec["flux_err"]*np.exp(-eta), logx=True, logy=True)
            l.fit(model="linear")
            prop = ax.errorbar(spec[e_unit], spec[f_unit], 
                               xerr=spec[f"{e_unit}_err"],
                               yerr=spec[f"{f_unit}_err"], ls=":", 
                               marker="x", label=r"{:.1f} days, $\beta$ ={:.2f}$\pm${:.2f}".format(t_ave,-l.p["m"], l.perr["m"]))
            ax.plot(e, l.model(e/(5e14))/np.exp(-host_galaxy_extinction("MW", e)*0.15/1.086), color=prop[0].get_color(), ls="-", lw=0.5)
        elif len(spec)>1:
            prop = ax.errorbar(spec[e_unit], spec[f_unit], 
                               xerr=spec[f"{e_unit}_err"],
                               yerr=spec[f"{f_unit}_err"], ls=":", 
                               marker="x", label="{:.1f} days".format(t_ave))
    ax.set_xscale("log")
    ax.legend(ncols=2, fontsize=8)
    ax.grid(which="major")
    ax.grid(which="minor", ls=":", alpha=0.5)