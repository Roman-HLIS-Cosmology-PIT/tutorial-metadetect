import time
import joblib
from tqdm.notebook import tqdm

import numpy as np

import ngmix

from make_data import make_data_metacal_simple


##############################
# Shear func without metacal #
##############################


def _shear_cuts(
    arr,
):
    T_psf = 2 * (0.6 / 2.355) ** 2
    assert arr is not None
    msk = (arr["flags"] == 0) & (arr["s2n"] > 10) & (arr["T"] / T_psf > 0.5)
    return msk


def _meas_shear_data(res):
    msk = _shear_cuts(res)
    e1 = np.mean(res["e"][msk, 0])
    e2 = np.mean(res["e"][msk, 1])
    g1, g2 = ngmix.shape.e1e2_to_g1g2(e1, e2)

    dt = [
        ("g1", "f8"),
        ("g2", "f8"),
    ]
    return np.array([(g1, g2)], dtype=dt)


def meas_m_c_cancel(res_raw):
    res = _meas_shear_data(res_raw)
    m = res["g1"] / 0.02 - 1

    c = res["g2"]

    return m[0], c[0]


##############################
# Shear func with metacal #
##############################


def _shear_cuts_mcal(arr, shear_type):
    assert arr is not None
    T_psf = 2 * (0.6 / 2.355) ** 2
    msk = (
        (arr["shear_type"] == shear_type)
        & (arr["flags"] == 0)
        & (arr["s2n"] > 10)
        & (arr["T"] / T_psf > 0.5)
    )
    return msk


def _meas_shear_data_mcal(res):
    msk = _shear_cuts_mcal(res, "noshear")
    e1 = np.mean(res[msk]["e"][:, 0])
    e2 = np.mean(res[msk]["e"][:, 1])
    g1, g2 = ngmix.shape.e1e2_to_g1g2(e1, e2)

    msk = _shear_cuts_mcal(res, "1p")
    e1_1p = np.mean(res[msk]["e"][:, 0])
    e2_1p = np.mean(res[msk]["e"][:, 1])
    g1_1p, g2_1p = ngmix.shape.e1e2_to_g1g2(e1_1p, e2_1p)

    msk = _shear_cuts_mcal(res, "1m")
    e1_1m = np.mean(res[msk]["e"][:, 0])
    e2_1m = np.mean(res[msk]["e"][:, 1])
    g1_1m, g2_1m = ngmix.shape.e1e2_to_g1g2(e1_1m, e2_1m)
    R11 = (g1_1p - g1_1m) / 0.02

    dt = [
        ("g1", "f8"),
        ("g2", "f8"),
        ("R11", "f8"),
    ]
    return np.array([(g1, g2, R11)], dtype=dt)


def meas_m_c_cancel_mcal(res_raw):
    res = _meas_shear_data_mcal(res_raw)
    m = res["g1"] / res["R11"] / 0.02 - 1

    c = res["g2"] / res["R11"]

    return m[0], c[0]


#############
# Main func #
#############


def make_struct(res, obs=None, shear_type=None):
    """
    make the data structure

    Parameters
    ----------
    res: dict
        With keys 's2n', 'e', and 'T'
    obs: ngmix.Observation
        The observation for this shear type
    shear_type: str
        The shear type

    Returns
    -------
    1-element array with fields
    """
    dt = [
        ("flags", "i4"),
        ("shear_type", "U7"),
        ("s2n", "f8"),
        ("e", "f8", 2),
        ("T", "f8"),
        ("Tpsf", "f8"),
    ]
    data = np.zeros(1, dtype=dt)
    if shear_type is None:
        data["shear_type"] = "no_mcal"
        data["flags"] = res["flags"]
        if res["flags"] == 0:
            data["s2n"] = res["s2n"]
            # for moments we are actually measureing e, the elliptity
            data["e"] = res["e"]
            data["T"] = res["T"]
            data["Tpsf"] = 2 * (0.6 / 2.355) ** 2
        else:
            data["s2n"] = np.nan
            data["e"] = np.nan
            data["T"] = np.nan
            data["Tpsf"] = np.nan
    else:
        data["shear_type"] = shear_type
        data["flags"] = res["flags"]
        if res["flags"] == 0:
            data["s2n"] = res["s2n"]
            # for moments we are actually measureing e, the elliptity
            data["e"] = res["e"]
            data["T"] = res["T"]
            data["Tpsf"] = obs.psf.meta["result"]["T"]
        else:
            data["s2n"] = np.nan
            data["e"] = np.nan
            data["T"] = np.nan
            data["Tpsf"] = np.nan

    return data


def _bootstrap_stat(d1, func, seed, nboot=500):
    dim = d1.shape[0]
    rng = np.random.RandomState(seed=seed)
    stats = []
    for _ in tqdm(range(nboot), total=nboot, leave=False):
        ind = rng.choice(dim, size=dim, replace=True)
        stats.append(func(d1[ind]))
    return stats


def _boostrap_m_c(res, func):
    m, c = func(res)
    bdata = _bootstrap_stat(res, func, 14324, nboot=2)
    merr, cerr = np.std(bdata, axis=0)
    return m, merr, c, cerr


def test_shear_meas(ntrial, simu_runner, boot_func):
    rng = np.random.RandomState(seed=116)
    seeds = rng.randint(low=1, high=2**29, size=ntrial)
    mcal_seeds = rng.randint(low=1, high=2**29, size=ntrial)

    tm0 = time.time()

    print("")

    loc = 0
    jobs = [
        joblib.delayed(simu_runner)(
            seeds[loc + i],
            mcal_seeds[loc + i],
        )
        for i in tqdm(range(ntrial), total=ntrial)
    ]
    outputs = joblib.Parallel(n_jobs=-1, verbose=0, backend="loky")(jobs)

    res = np.hstack(np.concatenate(outputs))

    total_time = time.time() - tm0
    print("time per:", total_time / ntrial, flush=True)

    m, merr, c, cerr = _boostrap_m_c(res, boot_func)

    print(
        ("\n\nm [1e-3, 3sigma]: %s +/- %s\nc [1e-5, 3sigma]: %s +/- %s")
        % (
            m / 1e-3,
            3 * merr / 1e-3,
            c / 1e-5,
            3 * cerr / 1e-5,
        ),
        flush=True,
    )


###############
# Simu runner #
###############


def simu_runner_no_mcal(seed, mcal_seed=None):
    model = "exp"
    hlr = 0.3
    psf_model = "moffat"
    psf_fwhm = 0.6
    psf_T = 2 * (psf_fwhm / 2.355) ** 2

    noise_sigma = 1e-5

    img_size = 51
    pixel_scale = 0.2

    shear_g1 = 0.01
    shear_g2 = 0.0

    fitter = ngmix.admom.AdmomFitter()

    gal, psf, shifts = make_data_metacal_simple(
        model=model,
        hlr=hlr,
        g1=shear_g1,
        g2=shear_g2,
        psf_model=psf_model,
        psf_fwhm=psf_fwhm,
        noise_sigma=noise_sigma,
        image_size=img_size,
        pixel_scale=pixel_scale,
        seed=seed,
    )

    weight = np.ones((img_size, img_size)) / noise_sigma**2

    img_center = ((img_size - 1) / 2.0, (img_size - 1) / 2.0)
    psf_center = ((img_size - 1) / 2.0, (img_size - 1) / 2.0)

    img_jacob = ngmix.DiagonalJacobian(
        row=img_center[1] + shifts[1] / pixel_scale,
        col=img_center[0] + shifts[0] / pixel_scale,
        scale=pixel_scale,
    )
    psf_jacob = ngmix.DiagonalJacobian(
        row=psf_center[1],
        col=psf_center[0],
        scale=pixel_scale,
    )

    psf_obs = ngmix.Observation(
        image=psf,
        jacobian=psf_jacob,
    )

    obs = ngmix.Observation(
        image=gal,
        weight=weight,
        jacobian=img_jacob,
        psf=psf_obs,
    )

    res = fitter.go(
        obs,
        psf_T,
    )

    data = make_struct(res=res)

    return data


def simu_runner_mcal(seed, mcal_seed=None):
    model = "exp"
    hlr = 0.3
    psf_model = "moffat"
    psf_fwhm = 0.6

    noise_sigma = 1e-5

    img_size = 51
    pixel_scale = 0.2

    shear_g1 = 0.02
    shear_g2 = 0.0

    gal, psf, shifts = make_data_metacal_simple(
        model=model,
        hlr=hlr,
        g1=shear_g1,
        g2=shear_g2,
        psf_model=psf_model,
        psf_fwhm=psf_fwhm,
        noise_sigma=noise_sigma,
        image_size=img_size,
        pixel_scale=pixel_scale,
        seed=seed,
    )

    weight = np.ones((img_size, img_size)) / noise_sigma**2

    img_center = ((img_size - 1) / 2.0, (img_size - 1) / 2.0)
    psf_center = ((img_size - 1) / 2.0, (img_size - 1) / 2.0)

    img_jacob = ngmix.DiagonalJacobian(
        row=img_center[1] + shifts[1] / pixel_scale,
        col=img_center[0] + shifts[0] / pixel_scale,
        scale=pixel_scale,
    )
    psf_jacob = ngmix.DiagonalJacobian(
        row=psf_center[1],
        col=psf_center[0],
        scale=pixel_scale,
    )

    psf_obs = ngmix.Observation(
        image=psf,
        jacobian=psf_jacob,
    )

    obs = ngmix.Observation(
        image=gal,
        weight=weight,
        jacobian=img_jacob,
        psf=psf_obs,
    )

    rng_mcal = np.random.RandomState(seed=mcal_seed)
    fitter = ngmix.admom.AdmomFitter()
    psf_fitter = ngmix.admom.AdmomFitter()
    guesser = ngmix.guessers.GMixPSFGuesser(
        rng=rng_mcal,
        ngauss=1,
        guess_from_moms=True,
    )

    runner = ngmix.runners.Runner(
        fitter=fitter,
        guesser=guesser,
    )
    psf_runner = ngmix.runners.PSFRunner(
        fitter=psf_fitter,
        guesser=guesser,
    )

    boot = ngmix.metacal.MetacalBootstrapper(
        runner=runner,
        psf_runner=psf_runner,
        types=["noshear", "1p", "1m"],
        psf="fitgauss",
        rng=rng_mcal,
    )

    res, obs_mcal = boot.go(
        obs,
    )

    dlist = []
    for stype in res:
        pdata = make_struct(
            res=res[stype], obs=obs_mcal[stype], shear_type=stype
        )
        dlist.append(pdata)

    return dlist
