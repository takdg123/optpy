import numpy as np
import matplotlib.pyplot as plt

from threeML import PluginPrototype
from threeML.io.logging import setup_logger
from threeML.utils.statistics.likelihood_functions import half_chi2
from astromodels.core.parameter import Parameter
from astromodels.functions.priors import Uniform_prior, Log_uniform_prior, Gaussian
import collections
from typing import Any, Dict, List, Optional, Tuple, Union
import astropy.units as u

from numba import njit

from .extinction import host_galaxy_extinction
from . import utils

from threeML import Powerlaw


log = setup_logger(__name__)

Jy2erg = 1e-23
pflux2Jy = (u.keV.to(u.erg))/(utils.eV2nu(1000).value)/Jy2erg
Jy2eflux = pflux2Jy*Jy2erg*(utils.eV2nu(1000).value)

class OpticalPlugin(PluginPrototype):
    def __init__(self, name, table, model_name, redshift=None, t_index=None, extinction_model="MW", Av=-0.82, **kwargs):
        
        if t_index is not None:
            self.table = table[table["t_index"] == t_index]
        else:
            self.table = table
            
        self.table.sort("eV")
            
        self.x = self.table["eV"].to(u.eV).value * u.eV
        self.xerr = self.table["eV_err"].to(u.eV).value * u.eV

        self.x_Hz = self.table["Hz"].to(u.Hz).value
        self.xerr_Hz = self.table["Hz_err"].to(u.Hz).value
        
        self.x_keV = self.x.to(u.keV).value
        self.xerr_keV = self.table["eV_err"].to(u.keV).value
        
        self.x_erg = self.x.to(u.erg).value
        
        self.y = self.table["e2dnde"].to(u.erg/u.cm**2/u.s).value * u.erg/u.cm**2/u.s
        self.yerr = self.table["e2dnde_err"].to(u.erg/u.cm**2/u.s).value * u.erg/u.cm**2/u.s
        self.y_Jy = self.table["Jy"].to(u.Jy).value * u.Jy
        self.yerr_Jy = self.table["Jy_err"].to(u.Jy).value * u.Jy

        self.redshift = redshift
        self.model_name = model_name
        self._source_name = kwargs.pop("source_name", None)

        if self.redshift is None:
            self._eta = np.zeros(self.get_number_of_data_points())
        else:
            self._eta = host_galaxy_extinction(extinction_model, nu = self.x_Hz, z = self.redshift)
        
        self._full_likelihood = kwargs.pop("full_likelihood", False)
        
        self._nuisance_parameter: Dict[str, Parameter] = collections.OrderedDict()

        self._nuisance_parameter["Av_%s" % name]: Parameter = Parameter(
            "Av_%s" % name,
            -0.82,
            min_value=-4,
            max_value=2,
            delta=0.01,
            free=False,
            desc="Av used in the host galaxy extinction for %s" % name,
        )
        self._nuisance_parameter["cons_%s" % name]: Parameter = Parameter(
             "cons_%s" % name,
            1.0,
            min_value=0.8,
            max_value=1.2,
            delta=0.05,
            free=False,
            desc="Effective area correction for %s" % name,
        )

        super().__init__(name, self._nuisance_parameter)

    def get_number_of_data_points(self):
        return len(self.x)

    def set_model(self, model):
        self._model = model
    
    def get_log_like(self, total=True):

        if self._source_name is not None:
            p = self.get_parameter(self._model[self._source_name].parameters)
        else:
            p = self.get_parameter(self._model.parameters)

        Av = 10**self._nuisance_parameter["Av_%s" % self._name].value
        tau = np.exp(-self._eta*Av/1.086)
        flux = self.get_model(self.x_keV, p = p, energy_unit="keV") * tau

        chi2_ = half_chi2(self.y_Jy, self.yerr_Jy, flux)
        
        assert np.all(np.isfinite(chi2_))

        if total: 
            return np.sum(chi2_) * (-1)
        else:
            return chi2_ * -1
        # stats = kwargs.pop("full_likelihood", self._full_likelihood)
        # if stats:
        #     if self.model_name == "powerlaw":
        #         x, K, index, x0  = symbols('x K index x0')
        #         y = K*(x/x0)**index
        #         yprime = y.diff(x)
        #         f = lambdify([x, K, index, x0], yprime, 'numpy')
        #         xerr = f(self.x_keV, *p)*tau*Fnu_unit*self.x_keV*self.xerr_keV
        #         return half_total_chi2(self.y_Jy, self.yerr_Jy, flux, xerr)
        #     elif self.model_name == "efs_model":
        #         s = 0.80-0.03*(2*p[2]+1)
        #         x, K, xb, low_index, high_index  = symbols('x K xb low_index high_index')
        #         y = K/x*((x/xb)**(s*low_index)+(x/xb)**(s*high_index))**(-1./s)
        #         yprime = y.diff(x)
        #         f = lambdify([x, K, xb, low_index, high_index], yprime, 'numpy')
        #         xerr = f(self.x_keV, *p)*tau*Fnu_unit*self.x_keV*self.xerr_keV
        #         return half_total_chi2(self.y_Jy, self.yerr_Jy, flux, xerr)
        # else:
        #     return half_total_chi2(self.y_Jy, self.yerr_Jy, flux, np.zeros(self.get_number_of_data_points()))

    def get_model(self, x, energy_unit = "eV", p=None, units = "Jy", **kwargs):
        if p is None:
            if self._source_name is not None:
                p = self.get_parameter(self._model[self._source_name].parameters)
            else:
                p = self.get_parameter(self._model.parameters)

        x = (x * getattr(u, energy_unit)).to(u.keV).value
        const = kwargs.pop("const", self._nuisance_parameter["cons_%s" % self._name].value)

        if self.model_name == "powerlaw":
            model = Powerlaw(K = p[0], index = p[1], piv=p[2])
            flux = model(x)
        elif self.model_name == "efs_model":
            flux = efs_model(x, *p)
        elif self.model_name == "efs_model_cutoff":
            flux = efs_model_cutoff(x, *p)


        if units == "Jy":
            return x * flux * pflux2Jy*const
        elif units == "e2dnde":
            return x**2 * flux * u.keV.to(u.erg)*const
        elif units == "dNdE":
            return flux*const
        
    def y_local(self, y):
        Av = 10**self._nuisance_parameter["Av_%s" % self._name].value
        tau = np.exp(-self._eta*Av/1.086)
        return y/tau
    
    def yerr_local(self, yerr):
        Av = self._nuisance_parameter["Av_%s" % self._name].value
        tau = np.exp(-self._eta*Av/1.086)
        return yerr/tau
    
    def inner_fit(self):
        return self.get_log_like()

    def assign_to_source(self, source_name: str) -> None:

        self._source_name = source_name
    
    def use_Av_correction(self,
                          min_value: Union[int, float] = -4,
                          max_value: Union[int, float] = 2) -> None:
        log.info(
            f"{self._name} is using a free Av between {min_value} and {max_value}")
        self._nuisance_parameter["Av_%s" % self._name].free = True
        self._nuisance_parameter["Av_%s" % self._name].bounds = (min_value, max_value)
        self._nuisance_parameter["Av_%s" % self._name].prior = Gaussian(mu=2)

    def fix_Av_correction(self, value: Union[int, float] = 0.15) -> None:
        log.info(
            f"{self._name} is using a fixed Av with {value}")
        self._nuisance_parameter["Av_%s" % self._name].value = value
        self._nuisance_parameter["Av_%s" % self._name].fix = True

    def use_effective_area_correction(self,
                                      min_value: Union[int, float] = 0.8,
                                      max_value: Union[int, float] = 1.2) -> None:
        log.info(
            f"{self._name} is using effective area correction (between {min_value} and {max_value})")
        self._nuisance_parameter["cons_%s" % self._name].free = True
        self._nuisance_parameter["cons_%s" % self._name].bounds = (min_value, max_value)
        self._nuisance_parameter["cons_%s" % self._name].set_uninformative_prior(Uniform_prior)

    def fix_effective_area_correction(self,
                                      value: Union[int, float] = 1) -> None:
        log.info(
            f"{self._name} is using a fixed effective area correction with {value}")
        self._nuisance_parameter["cons_%s" % self._name].value = value
        self._nuisance_parameter["cons_%s" % self._name].fix = True

    
    def get_parameter(self, model):

        if self.model_name == "powerlaw":
            for par in model:
                if "K" in par:
                    tmp_K = model[par].value
                elif "index" in par:
                    tmp_index = model[par].value
                elif "piv" in par:
                    tmp_piv = model[par].value
            return tmp_K, tmp_index, tmp_piv
        elif self.model_name == "efs_model":
            for par in model:
                if "K" in par:
                    tmp_K = model[par].value
                elif "xb" in par:
                    tmp_xb = model[par].value
                elif "low_index" in par:
                    tmp_low_index = model[par].value
                # elif "high_index" in par:
                #     tmp_high_index = model[par].value
            return tmp_K, tmp_xb, tmp_low_index
        elif self.model_name == "efs_model_cutoff":
            for par in model:
                if "K" in par:
                    tmp_K = model[par].value
                elif "xb" in par:
                    tmp_xb = model[par].value
                elif "low_index" in par:
                    tmp_low_index = model[par].value
                elif "log10xc" in par:
                    tmp_cutoff = model[par].value
                elif "high_index" in par:
                    tmp_high_index = model[par].value
                
            return tmp_K, tmp_xb, tmp_low_index, tmp_high_index, tmp_cutoff
        

    def plot(self, ax = None, target="flux", unit="eV"):
        if ax is None:
            ax = plt.gca()

        factor = u.eV.to(getattr(u, unit))
        if target == "flux":
            props = ax.errorbar(self.x*factor, self.y_local(self.y_Jy), yerr=self.yerr_local(self.yerr_Jy), 
                xerr=self.xerr*factor, lw=1, ls="", label="Optical")
            ax.loglog(self.x*factor, self.get_model(self.x_keV, energy_unit="keV"), 
                color=props[0].get_color(), label="Optical Model")
            ax.set_ylabel(r"Flux [Jy]", fontsize=15)
        elif target == "e2dnde":
            props = ax.errorbar(opt.x*factor, self.y_local(self.y), yerr=self.yerr_local(self.yerr), 
                xerr=self.xerr*factor, lw=1, ls="", label="Optical")
            ax.loglog(self.x*factor, self.x*self.get_model(self.x_keV, energy_unit="keV")*eVJy2erg, 
                color=props[0].get_color(), label="Optical Model")
            ax.set_ylabel("Flux [erg/cm$^2$/s]", fontsize=15)
        elif target == "resid":
            ax.errorbar(self.x*factor, self.get_resid(), 
                yerr=1, xerr=self.xerr*factor, lw=1, ls="")
            ax.set_xscale("log")
            ax.set_ylabel(r"$\chi^2$/2", fontsize=15)
        
        if unit == "eV":
            ax.set_xlabel("Energy [eV]", fontsize=15)
            ax.set_xlim(0.8)
        elif unit == "keV":
            ax.set_xlabel("Energy [keV]", fontsize=15)
            ax.set_xlim(0.0008, )
        ax.grid()

    def get_resid(self):
        return np.sign(self.y_local(self.y_Jy).value-self.get_model(self.x_keV, energy_unit="keV"))*np.sqrt(-self.get_log_like(total=False)*2)

@njit(fastmath=True)
def efs_model(x, K, xb, low_index):
    xb = 10**xb
    xx = np.divide(x, xb)
    x1 = np.divide(1, xb)
    p = 2*(low_index-1)+1
    s = 0.80-0.03*p
    high_index = low_index+0.5
    fnu = K*(xx**(s*low_index)+xx**(s*high_index))**(-1./s)
    fnu1 = (x1**(s*low_index)+x1**(s*high_index))**(-1./s)

    return fnu/fnu1

@njit(fastmath=True)
def efs_model_cutoff(x, K, xb, low_index, high_index, log10xc):
    xb = 10**xb
    xx = np.divide(x, xb)
    x1 = np.divide(1, xb)
    p = 2*(low_index-1)+1
    s = 0.80-0.03*p
    #high_index = low_index+0.5
    fnu = K*(xx**(s*low_index)+xx**(s*high_index))**(-1./s)
    fnu1 = (x1**(s*low_index)+x1**(s*high_index))**(-1./s)
    
    xc = 10**log10xc
    cutoff = np.exp(-np.divide(x, xc))
    cutoff[cutoff < 1e-20] = 1e-20
    return fnu/fnu1*cutoff
    