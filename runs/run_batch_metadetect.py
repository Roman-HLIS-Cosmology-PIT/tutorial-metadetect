import os
from copy import deepcopy
import time
import joblib
from tqdm.notebook import tqdm
import s3fs
import numpy as np
from astropy.io import fits
import galsim
import ngmix
import metadetect
from astropy.wcs import WCS
import galsim.roman as roman
import healpy as hp
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
import sys
import sep




def radec_to_healpix(ra_deg, dec_deg, nside, nest=False):
    """
    Convert RA and DEC (in degrees) to HEALPix pixel index.
    """
    ra_deg = np.asarray(ra_deg)
    dec_deg = np.asarray(dec_deg)

    theta = np.radians(90.0 - dec_deg)  
    phi = np.radians(ra_deg)            

    pix = hp.ang2pix(nside, theta, phi, nest=nest)
    return pix


def match_catalog(detect_ra, detect_dec, catalog_ra, catalog_dec):
    """
    Match detected sources to the nearest entry in a reference catalog.
    Returns:
    indices :Indices in the catalog that best match each detection.
    distances_arcsec : Angular distance to the nearest catalog entry in arcseconds.
    """
    # Create SkyCoord objects
    coords_detect = SkyCoord(detect_ra, detect_dec, unit="deg")
    coords_catalog = SkyCoord(catalog_ra, catalog_dec, unit="deg")

    # Match each detection to the closest catalog source
    idx, d2d, _ = coords_detect.match_to_catalog_sky(coords_catalog)

    return idx, d2d.arcsec
    

def make_mbobs(image,psf, wcs, noise_sigma, img_jacobian):
    psf_img = psf.drawImage(
        nx=PSF_IMG_SIZE,
        ny=PSF_IMG_SIZE,
        wcs=wcs,
    ).array

    # Make NGmix jacobian
    psf_cen = (PSF_IMG_SIZE - 1) / 2
    img_cen = (np.array([IMG_SIZE, IMG_SIZE]) - 1) / 2

    psf_jac = ngmix.Jacobian(
        row=psf_cen,
        col=psf_cen,
        wcs=img_jacobian,
    )
    img_jac = ngmix.Jacobian(
        row=img_cen[0],
        col=img_cen[1],
        wcs=img_jacobian,
    )

    # Make PSF observation
    psf_obs = ngmix.Observation(
        image=psf_img,
        jacobian=psf_jac,
    )

    obs = ngmix.Observation(
            image= image,
            jacobian=img_jac,
            weight=np.ones((IMG_SIZE, IMG_SIZE), dtype=float) / noise_sigma**2,
            psf=psf_obs,
            ormask=np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.int32),
            bmask=np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.int32),
        )
    obslist = ngmix.ObsList()
    obslist.append(obs)
    mbobs = ngmix.MultiBandObsList()
    mbobs.append(obslist)
    return mbobs

def get_block_image(image_path):
    # Read images from preview folder
    with fits.open("s3://" + image_path, fsspec_kwargs={"anon": True}) as hdul:
        f = hdul[0].section[0, 0:1]
        h = hdul[0].header

    return f[0], h
def get_photometry(flux_imcom, band):
    oversample_pix = 0.039
    pix_size = 0.11
    norm_fact = roman.exptime*roman.collecting_area*(pix_size**2/oversample_pix**2) # get flux in same units as truth catalog
    roman_bandpasses = galsim.roman.getBandpasses()
    zp = roman_bandpasses[band].zeropoint
    flux_meas = flux_imcom/norm_fact
    mags_meas = -2.5*np.log10(flux_meas) + zp
    return flux_meas, mags_meas

def _match_meta_fluxes_from_image(image_path, mdet_seed, model, band="H158", usecfg = 'default', keepcols = None):
    """
    Recover the IMCOM image from `aws` and run metadection.
    """
    if usecfg == 'default':
        cfg = deepcopy(METADETECT_CONFIG)
    else: 
        cfg = usecfg
    cfg["model"] = model

    if keepcols is None:
        keepcols = ['flags', 's2n', 'band_flux_flags', 'T', 'T_ratio']
    
    
    # Read images from preview folder
    image, h = get_block_image(image_path)

    # Make WCS from image header
    h.pop("NAXIS3")
    h.pop("NAXIS4")
    h["NAXIS"] = 2
    wcs = galsim.AstropyWCS(header=h)
    img_jacobian = wcs.jacobian(image_pos=galsim.PositionD(h["CRPIX1"], h["CRPIX2"]))

    # Estimate of the image noise. Commented out code extracts the noise using sep
    bkg = sep.Background(image.astype(image.dtype.newbyteorder('=')))
    noise_sigma = bkg.globalrms

    # Make PSF
    psf = galsim.Gaussian(fwhm=PSF_FWHM[band])
    #Create ngmix observations
    mbobs = make_mbobs(image,psf, wcs, noise_sigma, img_jacobian)

    
    # Run metadetect
    res = metadetect.do_metadetect(
        deepcopy(cfg),
        mbobs=mbobs,
        rng=np.random.RandomState(seed=mdet_seed),
    )

    flux_imcom = res['noshear']['pgauss_band_flux']
    flux_meas, mags_meas = get_photometry(flux_imcom, band)

    ## Since we are working on the preview area, we already know what healpy pixel the data falls in, but in case you want
    ## to work with the full data, hp_pix can help you know which healpy pixel your data falls in to fetch appropiate data
    x, y = res["noshear"]["sx_col"], res["noshear"]["sx_row"]
    ra_pos, dec_pos = wcs.toWorld(x,y, units='deg') # convert pixel positions to RA,DEC
    hp_pix = radec_to_healpix(ra_pos, dec_pos, 32)


    # Get matched positions (removed). Doing this outside instead
    #cat_ra, cat_dec = cat_pos['ra'], cat_pos['dec']
    #matched_idx, err_dist = match_catalog(ra_pos, dec_pos, cat_ra, cat_dec)


    buff_mask = (res["noshear"]["sx_col"] > BOUND_SIZE) & (res["noshear"]["sx_col"] < IMG_SIZE-BOUND_SIZE) & \
            (res["noshear"]["sx_row"] > BOUND_SIZE) & (res["noshear"]["sx_row"] < IMG_SIZE-BOUND_SIZE)
    star_mask = (res["noshear"][model+'_'+'T'] < 0.01) #preliminary, needs testing
    star_mask = (~star_mask).astype(int) #convert boolean to 0,1. 0 = galaxy, 1 = star. 

    buff_mask = (~buff_mask).astype(int) # convert boolean to 0,1. 0 = good, 1 = bad. ~buff_mask needed because conversion to bool
                                         # is True = 1 and False = 0 by default
    resultdic  = {
        'ra_meta': ra_pos,
        'dec_meta': dec_pos, 
        'flux_meta': flux_meas, 
        'mag_meta': mags_meas,
        'edge_flag': buff_mask,
        'star_mask': star_mask
    }
    for col in keepcols:
        resultdic[col] = res['noshear'][model+ '_' + col]

    return resultdic

def get_coadd_filenames(coadd_dir, band):

    fs = s3fs.S3FileSystem(anon=True)
    
    band_dir = os.path.join(coadd_dir, band)
    subdirs = fs.ls(band_dir)

    all_img_paths = []
    for subdir in subdirs:
        all_img_paths += fs.ls(subdir)
    return all_img_paths



### Meta config
# Size of the taget Gaussian PSF in IMCOM
PSF_FWHM = {
    "Y106": 0.22,
    "J129": 0.231,
    "H158": 0.242,
    "F184": 0.253,
    "K213": 0.264,
}
# Size of the image used to draw the PSF
PSF_IMG_SIZE = 151#*3

# Size of one IMCOM block
IMG_SIZE = 2688
#IMG_SIZE =1000
# Boundary used to avoid edge effects
# Objects for which the centre is within this distance from the edge will be
# masked out.
BOUND_SIZE = 100

METADETECT_CONFIG = {
    # Shape measurement method
    # wmom: weighted moments
    "model": "pgauss",

    # Size of the weight function for the moments
    'weight': {
        'fwhm': 1.2*5,  # arcsec
    },

    # Metacal settings
    'metacal': {
        'psf': 'fitgauss',
        # Kind of shear applied to the image
        'types': ['noshear', '1p', '1m', '2p', '2m'],
    },

    'sx': {
        # in sky sigma
        # DETECT_THRESH
        'detect_thresh': 5,

        # Minimum contrast parameter for deblending
        # DEBLEND_MINCONT
        'deblend_cont': 0.00001,

        # minimum number of pixels above threshold
        # DETECT_MINAREA: 6
        'minarea': 4,

        'filter_type': 'conv',

        # 7x7 convolution mask of a gaussian PSF with FWHM = 3.0 pixels.
        'filter_kernel': [
            [0.004963, 0.021388, 0.051328, 0.068707, 0.051328, 0.021388, 0.004963],  # noqa
            [0.021388, 0.092163, 0.221178, 0.296069, 0.221178, 0.092163, 0.021388],  # noqa
            [0.051328, 0.221178, 0.530797, 0.710525, 0.530797, 0.221178, 0.051328],  # noqa
            [0.068707, 0.296069, 0.710525, 0.951108, 0.710525, 0.296069, 0.068707],  # noqa
            [0.051328, 0.221178, 0.530797, 0.710525, 0.530797, 0.221178, 0.051328],  # noqa
            [0.021388, 0.092163, 0.221178, 0.296069, 0.221178, 0.092163, 0.021388],  # noqa
            [0.004963, 0.021388, 0.051328, 0.068707, 0.051328, 0.021388, 0.004963],  # noqa
        ]
    },

    # This is for the cutout at each detection
    'meds': {
        'min_box_size': 32,#*20,
        'max_box_size': 32*20,#*20,

        'box_type': 'iso_radius',

        'rad_min': 4,
        'rad_fac': 2,
        'box_padding': 2,
    },

    # check for an edge hit
    'bmask_flags': 2**30,

    'nodet_flags': 2**0,
}



## Some params for the function below
coadd_dir = "nasa-irsa-simulations/openuniverse2024/roman/preview/RomanWAS/images/coadds"
mdet_seed = 42
band = 'H158'
model = 'pgauss'

idx = int(sys.argv[1])


all_img_paths = get_coadd_filenames(coadd_dir, band)
img_filename = all_img_paths[idx]
splt = img_filename.split('/')
block_id = splt[-1].split('_')[2] + '_' + splt[-1].split('_')[3]

## Run Metadetect and make catalog
print('Running Metadetect for Block: ' + block_id )
dict_results =_match_meta_fluxes_from_image(img_filename, mdet_seed, model, band, METADETECT_CONFIG )
df = pd.DataFrame(dict_results)
print('Finished Running')

# create directory for data
row = block_id.split('_')[-1]
dir_ = "output/"  + row + '/'
os.makedirs(dir_, exist_ok=True)

# save catalog
outfile = dir_ + 'catalog_' + block_id +".parquet"
print('Saving Catalog to: ' + outfile)
df.to_parquet(outfile) 




