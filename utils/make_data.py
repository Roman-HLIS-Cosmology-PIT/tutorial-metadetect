import numpy as np

import galsim


def make_data_metacal_simple(
    model="gauss",
    hlr=0.5,
    g1=0.0,
    g2=0.0,
    psf_model="moffat",
    psf_fwhm=0.6,
    image_size=51,
    pixel_scale=0.2,
    noise_sigma=1e-5,
    seed=42,
):
    rng = np.random.RandomState(seed)

    if model == "gauss":
        gal = galsim.Gaussian(half_light_radius=hlr)
    elif model == "exp":
        gal = galsim.Exponential(half_light_radius=hlr)
    gal = gal.shear(g1=g1, g2=g2)

    if psf_model == "gauss":
        psf = galsim.Gaussian(fwhm=psf_fwhm)
    elif psf_model == "moffat":
        psf = galsim.Moffat(fwhm=0.6, beta=3.5)

    obj = galsim.Convolve([gal, psf])

    img = obj.drawImage(
        nx=image_size,
        ny=image_size,
        scale=pixel_scale,
    ).array

    noise = rng.normal(size=(image_size, image_size)) * noise_sigma
    img += noise

    psf_img = psf.drawImage(
        nx=image_size,
        ny=image_size,
        scale=pixel_scale,
    ).array

    return img, psf_img
