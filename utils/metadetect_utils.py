TEST_METADETECT_CONFIG = {
    "model": "wmom",
    "weight": {
        "fwhm": 1.2,  # arcsec
    },
    "metacal": {
        "psf": "fitgauss",
        "types": ["noshear", "1p", "1m", "2p", "2m"],
    },
    "sx": {
        # in sky sigma
        # DETECT_THRESH
        "detect_thresh": 0.8,
        # Minimum contrast parameter for deblending
        # DEBLEND_MINCONT
        "deblend_cont": 0.00001,
        # minimum number of pixels above threshold
        # DETECT_MINAREA: 6
        "minarea": 4,
        "filter_type": "conv",
        # 7x7 convolution mask of a gaussian PSF with FWHM = 3.0 pixels.
        "filter_kernel": [
            [
                0.004963,
                0.021388,
                0.051328,
                0.068707,
                0.051328,
                0.021388,
                0.004963,
            ],  # noqa
            [
                0.021388,
                0.092163,
                0.221178,
                0.296069,
                0.221178,
                0.092163,
                0.021388,
            ],  # noqa
            [
                0.051328,
                0.221178,
                0.530797,
                0.710525,
                0.530797,
                0.221178,
                0.051328,
            ],  # noqa
            [
                0.068707,
                0.296069,
                0.710525,
                0.951108,
                0.710525,
                0.296069,
                0.068707,
            ],  # noqa
            [
                0.051328,
                0.221178,
                0.530797,
                0.710525,
                0.530797,
                0.221178,
                0.051328,
            ],  # noqa
            [
                0.021388,
                0.092163,
                0.221178,
                0.296069,
                0.221178,
                0.092163,
                0.021388,
            ],  # noqa
            [
                0.004963,
                0.021388,
                0.051328,
                0.068707,
                0.051328,
                0.021388,
                0.004963,
            ],  # noqa
        ],
    },
    "meds": {
        "min_box_size": 31,
        "max_box_size": 31,
        "box_type": "iso_radius",
        "rad_min": 4,
        "rad_fac": 2,
        "box_padding": 2,
    },
    # check for an edge hit
    "bmask_flags": 2**30,
    "nodet_flags": 2**0,
}


def get_cutout(img, x, y, stamp_size, return_bounds=False):
    orow = int(y)
    ocol = int(x)
    half_box_size = stamp_size // 2
    maxrow, maxcol = img.shape

    ostart_row = orow - half_box_size + 1
    ostart_col = ocol - half_box_size + 1
    oend_row = orow + half_box_size + 2  # plus one for slices
    oend_col = ocol + half_box_size + 2

    ostart_row = max(0, ostart_row)
    ostart_col = max(0, ostart_col)
    oend_row = min(maxrow, oend_row)
    oend_col = min(maxcol, oend_col)

    cutout_row = y - ostart_row
    cutout_col = x - ostart_col

    if return_bounds:
        return (
            img[ostart_row:oend_row, ostart_col:oend_col],
            cutout_row,
            cutout_col,
            ostart_row,
            ostart_col,
            oend_row,
            oend_col,
        )
    else:
        return (
            img[ostart_row:oend_row, ostart_col:oend_col],
            cutout_row,
            cutout_col,
        )
