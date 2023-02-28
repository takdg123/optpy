import astropy.units as u

sec2day = 1./(24*60*60)
Jy2erg = 1e-23

tab_dtype = [('image', '<U62'), 
            ('obs', '<U4'), 
            ('obj', '<U10'), 
            ('ra', '<f8'), 
            ('dec', '<f8'), 
            ('date-obs', '<U10'), 
            ('jd', '<f8'), 
            ('filter', '<U2'), 
            ('stdnumb', '<i8'), 
            ('zp', '<f8'), 
            ('zper', '<f8'), 
            ('seeing', '<f8'), 
            ('skyval', '<f8'), 
            ('skysig', '<f8'), 
            ('ul_3sig', '<f8'), 
            ('ul_5sig', '<f8'), 
            ('mag', '<f8'), 
            ('magerr', '<f8'), 
            ('aper_dia_pix', '<f8')]


filter_dict = dict(
    #   Johnson-Cousine (CFHT)
    B=dict(
        lam=4300.82,
        bdw=941.52,
        c='blue',
    ),
    V=dict(
        lam=5338.40,
        bdw=909.75,
        c='green',
    ),
    R=dict(
        lam=6515.58,
        bdw=1184.89,
        c='red',
    ),
    I=dict(
        lam=8091.02,
        bdw=2024.22,
        c='blueviolet',
    ),
    #   SDSS (PS1)
    g=dict(
        lam=4810.16,
        bdw=1053.08,
        c='dodgerblue',
    ),
    r=dict(
        lam=6155.47,
        bdw=1252.41,
        c='tomato',
    ),
    i=dict(
        lam=7503.03,
        bdw=1206.62,
        c='violet'
    ),
    z=dict(
        lam=8668.36,
        bdw=997.72,
        c='indigo'
    ),
    Y=dict(
        lam=9613.60,
        bdw=638.98,
        c='navy'
    ),
    #   2MASS
    J=dict(
        lam=12350.00,
        bdw=1624.32,
        c='hotpink',
    ),
    H=dict(
        lam=16620.00,
        bdw=2509.40,
        c='pink',
    ),
    Ks=dict(
        lam=21590.00,
        bdw=2618.87,
        c='magenta'
    ),
)
