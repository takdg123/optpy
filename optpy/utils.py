import numpy as np
import astropy.units as u


def lam2nu(lam):
	from astropy import constants as const
	c = const.c
	if hasattr(lam, "unit"):
		if lam.unit is not None:
			return c.to(u.Angstrom*u.Hz)/(lam)
	
	return c.to(u.Angstrom*u.Hz)/(lam*u.Angstrom)

def nu2eV(nu):
	from astropy import constants as const
	h = const.h
	if hasattr(nu, "unit"):
		if nu.unit is not None:
			return (h*nu).to(u.eV)
	return (h*nu*u.Hz).to(u.eV)


def eV2nu(eV):
	from astropy import constants as const
	h = const.h
	if hasattr(eV, "unit"):
		if eV.unit is not None:
			return (eV/h).to(u.Hz)
	return (eV*u.eV/h).to(u.Hz)


def lam2eV(lam):
	from astropy import constants as const
	h = const.h
	return (h*lam2nu(lam)).to(u.eV)

def AB2Flux(mag, magerr, units = "cgs"):
    if mag < 0:
        return np.nan, np.nan
    else:
        samples = np.random.normal(loc = mag, scale=magerr, size=10000)
        
        if units == "Jy":
        	flx = (mag*u.ABmag).to(u.Jy)
	        flx_sample = (samples*u.ABmag).to(u.Jy)
	        flx_err = np.std(flx_sample.value)
        	return flx.value, flx_err
        elif units == "cgs":
        	flx = (mag*u.ABmag).to(u.erg/u.s/u.cm**2/u.Hz)
	        flx_sample = (samples*u.ABmag).to(u.erg/u.s/u.cm**2/u.Hz)
	        flx_err = np.std(flx_sample.value)
        	return flx.value, flx_err


def make_clist(n, palette='Spectral'):
    import seaborn as sns
    palette = sns.color_palette(palette, as_cmap=True,)
    palette.reversed

    clist_ = [palette(i) for i in range(palette.N)]
    cstep = int(len(clist_)/n)
    clist = [clist_[i*cstep] for i in range(n)]
    return clist


def make_tbins(time, ttol = 0.05, output="index"):
	time_series = np.unique(time)

	time_bins = []
	temp = []

	for t in time_series:
	    if len(temp) == 0:
	        temp.append(t)
	        continue
	    else:
	        if abs(np.average(temp)/t -1) < ttol:
	            temp.append(t)
	        else:
	            time_bins.append(temp)
	            temp = [t]
	time_bins.append(temp)

	indices = []
	t_center = []

	for t in time:
		for i, bins in enumerate(time_bins):
			if t in bins:
				indices.append(i)
				t_center.append(np.average(bins))
				break

	if output == "index":
		return indices
	elif output == "time":
		return t_center
	elif output == "full":
		return t_center, indices, time_bins