"""Independent constant-tensor heat equation on the periodic unit box.

Analytic Fourier coefficients are integrated over cells, then aliased into the
sampling FFT. This is NOT time integration or the solver's finite-difference
stencil. Array axes are x, y[, z]. The field is normalized using all 3 components
even for a 2D solution. Here temperature means p/rho, not cgs kelvin.
"""
import math

import numpy as np
from scipy.special import erf


def tensor(dimension, gamma, iso, aniso, field):
    b = np.asarray(field, dtype=float)
    if (dimension not in (2, 3) or b.shape != (3,) or
            not np.all(np.isfinite(b)) or
            not all(math.isfinite(v) for v in (gamma, iso, aniso)) or
            gamma <= 1 or iso < 0 or aniso < 0):
        raise ValueError('invalid constant diffusion tensor')
    norm = np.linalg.norm(b)
    bhat = b[:dimension]/norm if norm else np.zeros(dimension)
    return (gamma-1)*(iso*np.eye(dimension)+aniso*np.outer(bhat, bhat))


def periodic_gaussian(shape, time, diffusion, sigma=.1, amplitude=.1,
                      background=1., cutoff=None):
    """Exact periodic, cell-averaged solution to truncation/roundoff accuracy."""
    d = len(shape)
    diffusion = np.asarray(diffusion, dtype=float)
    if (d not in (2, 3) or any(int(n) != n or n < 2 for n in shape) or
            diffusion.shape != (d, d) or not np.all(np.isfinite(diffusion)) or
            not np.allclose(diffusion, diffusion.T, rtol=0, atol=1.e-15) or
            np.linalg.eigvalsh(diffusion).min() < -1.e-15 or
            not all(math.isfinite(v) for v in (time, sigma, amplitude, background)) or
            time < 0 or sigma <= 0):
        raise ValueError('invalid periodic Gaussian arguments')
    if cutoff is None:
        cutoff = math.ceil(math.sqrt(-2*math.log(1.e-18))/(2*math.pi*sigma))+2
    if not isinstance(cutoff, int) or cutoff < 1:
        raise ValueError('invalid Fourier cutoff')
    axes = np.arange(-cutoff, cutoff+1)
    m = np.stack(np.meshgrid(*([axes]*d), indexing='ij'), axis=-1)
    k = 2*math.pi*m
    exponent = -.5*sigma*sigma*np.sum(k*k, axis=-1)
    exponent -= time*np.einsum('...i,ij,...j->...', k, diffusion, k)
    coeff = amplitude*(math.sqrt(2*math.pi)*sigma)**d*np.exp(exponent)
    coeff = coeff.astype(complex)
    for axis, n in enumerate(shape):
        coeff *= np.sinc(m[..., axis]/n)
        coeff *= np.exp(2j*math.pi*m[..., axis]*(.5/n-.5))
    folded = np.zeros(shape, dtype=complex)
    indices = tuple((m[..., a] % n).ravel() for a, n in enumerate(shape))
    np.add.at(folded, indices, coeff.ravel())
    values = np.fft.ifftn(folded)*math.prod(shape)
    if np.max(np.abs(values.imag)) > 1.e-13*max(1., abs(amplitude)):
        raise ValueError('non-real Gaussian Fourier reference')
    return background+values.real


def separable_gaussian(shape, sigma=.1, amplitude=.1, background=1.):
    """Independent real-space cell integral, including periodic images."""
    values = np.ones(shape)
    for a, n in enumerate(shape):
        centers = (np.arange(n)+.5)/n
        factor = np.zeros(n)
        # At the narrow widths used here omitted images are below roundoff.
        for image in range(-4, 5):
            offset = centers-.5-image
            factor += sigma*math.sqrt(math.pi/2)*n*(
                erf((offset+.5/n)/(math.sqrt(2)*sigma)) -
                erf((offset-.5/n)/(math.sqrt(2)*sigma)))
        axis_shape = [1]*len(shape)
        axis_shape[a] = n
        values *= factor.reshape(axis_shape)
    return background+amplitude*values


def metrics(observed, expected, initial, background=1., amplitude=.1):
    if (observed.shape != expected.shape or observed.shape != initial.shape or
            not all(np.all(np.isfinite(v)) for v in (observed, expected, initial))):
        raise ValueError('nonfinite or mismatched Gaussian fields')
    excess = expected-background
    error = observed-expected
    result = dict(l1=float(np.sum(np.abs(error))/np.sum(np.abs(excess))),
                  l2=float(np.linalg.norm(error.ravel())/np.linalg.norm(excess.ravel())),
                  linf_over_initial_amplitude=float(np.max(np.abs(error))/amplitude),
                  heat_error_over_initial_excess=float(
                      abs(np.sum(observed-initial))/np.sum(initial-background)),
                  peak=float(np.max(observed)), reference_peak=float(np.max(expected)))
    coords = np.meshgrid(*[(np.arange(n)+.5)/n-.5 for n in observed.shape], indexing='ij')
    cov = []
    for field in (observed-background, excess):
        weight = np.sum(field)
        centers = [np.sum(field*x)/weight for x in coords]
        cov.append([[float(np.sum(field*(x-centers[a])*(y-centers[b]))/weight)
                     for b, y in enumerate(coords)] for a, x in enumerate(coords)])
    # Periodic wrapped moments: compare like-for-like to sampled analytic fields,
    # not to the infinite-domain sigma^2 I + 2 D t when tails reach boundaries.
    result.update(covariance=cov[0], reference_covariance=cov[1])
    return result
