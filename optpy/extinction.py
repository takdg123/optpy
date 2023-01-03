from pathlib import Path
from astropy.io import ascii as asc
from scipy.interpolate import interp1d

SCRIPT_DIR = str(Path(__file__).parent.absolute())

def galactic_extinction(bandpass, system = ["SDSS", "Landolt", "UKIRT", "PS1"]):
    tab = asc.read(f"{SCRIPT_DIR}/data/extinction_calculator.csv")
    tab = tab[tab['Refcode of the publications']=="2011ApJ...737..103S"]
    
    if bandpass == "Ks":
        bandpass = "K"
    
    band = [str(sys) + " " + str(bandpass) for sys in system]
    unique_band = [tab[tab["Bandpass"] == b] for b in band if b in tab["Bandpass"]]
    
    if len(unique_band) >= 1:
        return unique_band[0]
    else:
        return None


def host_galaxy_extinction(model, nu=None, z=None):
    tab = asc.read(f"{SCRIPT_DIR}/data/host_galaxy_extinction.csv")
    eta_model = interp1d(tab[f"nu_{model}"], tab[f"eta_{model}"])
    if (nu is not None) and (z is not None):
        return eta_model(nu/(1.+z))
    else:
        return eta_model
