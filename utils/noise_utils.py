import numpy as np
from numpy.fft import fft2, ifft2, fftshift


def make_noise_data(
    noise_sigma=1.0,
    image_size=51,
    seed=42,
):
    rng = np.random.RandomState(seed)
    noise = rng.normal(size=(image_size, image_size)) * noise_sigma
    return noise


def compute_noise_correlation(noise_field, size=5):
    """
    Compute the 2D autocorrelation of a noise field and return a central patch.

    Parameters
    ----------
    noise_field : np.ndarray
        2D numpy array representing the noise field.
    size : int
        Size of the central square region of the correlation matrix to return.
        Must be odd to ensure a centered output.

    Returns
    -------
    corr_patch : np.ndarray
        Centered patch (size x size) of the autocorrelation matrix.
    """
    if size % 2 == 0:
        raise ValueError("`size` must be an odd number to center the output.")

    # Remove mean
    noise_zero_mean = noise_field - np.mean(noise_field)

    # Compute autocorrelation using FFT
    fft_noise = fft2(noise_zero_mean)
    power_spectrum = np.abs(fft_noise) ** 2
    corr = fftshift(np.real(ifft2(power_spectrum)))

    # Normalize
    corr /= np.prod(noise_field.shape)

    # Extract central patch
    center_y, center_x = np.array(corr.shape) // 2
    half_size = size // 2
    corr_patch = corr[
        center_y - half_size : center_y + half_size + 1,
        center_x - half_size : center_x + half_size + 1,
    ]

    return corr_patch
