import numpy as np
from grbpy.external.linearfit import LinearFit
from . import utils
from . import const
from .extinction import host_galaxy_extinction
import astropy.units as u
from astropy.table import Table
from scipy.stats import multivariate_normal

from iminuit import Minuit

class pl_fit:
    def __init__(self, table, z, model="MW", Av=0.15, fix_Av=False, verbose=False):

        self.flux = table["Jy"].value
        flag = ~np.isnan(self.flux)
        self.flux = self.flux[flag]
        
        if len(self.flux)<=3:
            if verbose: print("The number of data points is not enough.")
            self.flag=False
            return
        else:
            self.flag=True

        self.z = z
        self.extinction_model = model
        self.nu = table["Hz"].to(u.Hz).value[flag]
        self.nu_err = table["Hz_err"].value[flag]
        self.flux_err = table["Jy_err"].value[flag]
        self.Av = Av
        self.fix_Av = fix_Av

        self.eta = host_galaxy_extinction(self.extinction_model, nu = self.nu, z = self.z)
        self.pivot = 10**((np.log10(max(self.nu))+np.log10(min(self.nu)))/2.)
        

    def model(self, nu, k, m, Av):
        return 10**k*(nu/self.pivot)**(-m)*np.exp(-self.eta*Av/1.086)

    def likelihood(self, k, m, Av):
        xerr = m/self.nu*self.flux*self.nu_err
        return sum((self.flux-self.model(self.nu, k, m, Av))**2./(self.flux_err**2.+(xerr)**2.))

    def fit(self):
        if not(self.flag):
            return

        minuit=Minuit(self.likelihood, 
            m = -1, 
            k = np.log10(np.median(self.flux)), 
            Av = self.Av, 
            )
        minuit.limits["Av"] = (0, 2)

        minuit.fixed["Av"] = self.fix_Av

        minuit.errordef=1
        fit_result = minuit.migrad()
        minuit.hesse()

        chisq = fit_result.fval
        dof = len(self.flux) - len(fit_result.parameters)

        self.p = fit_result.values
        self.cov = fit_result.covariance
        self.perr = fit_result.errors
        self.stat = [chisq, dof]
        self.minuit = minuit
        self.fit_result = fit_result
        self.valid = fit_result.valid

def spectral_fit(table, z, model="MW", Av=0.15, fix_Av=True, sampling=False, return_fit=False, return_dist=False, **kwargs):
    time_list = np.unique(table["t_index"])
    fit_table = Table(dtype=[("t_index", int), ("k", float), ("beta", float), ("Av", float), 
    ("k_err", float), ("beta_err", float), ("Av_err", float), ("pivot", float), (r"chi2", float), ("dof", int)])

    for time in time_list:
        selected_table = table[table["t_index"] == time]
        l = pl_fit(selected_table, z, model=model, Av=Av, fix_Av=fix_Av)
        if l.flag:
            l.fit()
            fit_result = [time]+list(l.p)+list(l.perr)+[l.pivot]+list(l.stat)
            fit_table.add_row(fit_result)

    if return_dist:
        rv = multivariate_normal(np.asarray([fit_table["k"][0], 
            fit_table["beta"][0], 
            fit_table["Av"][0]]), 
            l.cov, allow_singular=True)
        return fit_table, rv
    elif sampling and (len(time_list) == 1):
        params_sample = np.random.multivariate_normal(np.asarray([fit_table["k"][0], fit_table["beta"][0], fit_table["Av"][0]]), 
                                           l.cov, kwargs.pop("size",10000))
        return fit_table, params_sample
    elif return_fit:
        return fit_table, l
    else:
        return fit_table

def temporal_fit(table, t_conv=1, flux_unit="e2dnde", return_fit=False, **kwargs):
    filter_list = np.unique(table["filter"])
    fit_table = Table(dtype=[("filter", str), ("alpha", float), ("log10(k)", float), 
    ("alpha_err", float), ("log10(k)_err", float), (r"chi2", float), ("dof", int)])

    t_max = kwargs.pop("t_max", None)
    mask = kwargs.pop("mask", None)
    num_pts = kwargs.pop("num_pts", None)


    for filt in filter_list:
        selected_table = table[table["filter"] == filt]

        if num_pts is not None:
            selected_table = selected_table[:num_pts]

        if (num_pts is None) and (t_max is not None):
            selected_table = selected_table[selected_table["time"] < t_max]
        
        if (num_pts is None) and (mask is not None):
            selected_table = selected_table[mask]

        flux = selected_table[flux_unit]
        flag = ~np.isnan(flux)
        flux = flux[flag]

        if len(flux)<=3:
            l = None
            continue

        t = selected_table["time"][flag]*t_conv
        flux_err = selected_table[f"{flux_unit}_err"][flag]

        l = LinearFit(t, flux, y_err= flux_err, logx=True, logy=True)
        l.fit(model="linear")
        fit_result = [filt]+list(l.p)+list(l.perr)+list(l.stat)

        fit_table.add_row(fit_result)

    fit_table["alpha"] *= -1
    
    if return_fit:
        return fit_table, l
    else:
        return fit_table



def get_butterfly(table, model="MW", show_plot=False, ax=None, unabsorbed=True, z=None, Av=0.15, **kwargs):
      
    e_min = min(table["eV"])
    e_max = max(table["eV"])
    E = np.geomspace(e_min, e_max, kwargs.pop("nbins", 101))
    freq = utils.eV2nu(E).value
    
    tab, params_sample = spectral_fit(table, z=z, model=model, Av=Av, sampling=True, **kwargs)

    k_sample = params_sample[:,0]
    m_sample = params_sample[:,1]
    
    eta = host_galaxy_extinction(model, nu=utils.eV2nu(E), z=z)
    tau = eta*tab["Av"]/1.086
    
    const = kwargs.pop("const", 1)
    
    if unabsorbed:
        factor = utils.JyHz2ErgEV()*np.ones(len(tau))*const
    else:
        factor = utils.JyHz2ErgEV()*np.exp(-tau)*const
    
    F = 10**tab["k"]*(freq/tab["pivot"])**(-tab["beta"])*factor/E
    F_sample = np.asarray([10**k_sample*(f/tab["pivot"])**(-m_sample)*fac for f, fac in zip(freq, factor)])
    F_sample = F_sample.T/E

    F_band = np.asarray([[e, f, np.percentile(fs, 16), np.percentile(fs, 84)] for e, f, fs in zip(E, F, F_sample.T)])

    if show_plot:
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()
        fill = kwargs.pop("fill", True)
        
        props = ax.plot(E, E**2*F_band[:,1], **kwargs)
        if fill:
            ax.fill_between(E, E**2*F_band[:,2], E**2*F_band[:,3], 
                             color=props[0].get_color(), alpha=0.3)
        else:
            ax.fill_between(E, E**2*F_band[:,2], E**2*F_band[:,3], 
                             facecolor="white", edgecolor=props[0].get_color(), alpha=0.3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        return ax
    else:
        return F_band, E, F_sample
