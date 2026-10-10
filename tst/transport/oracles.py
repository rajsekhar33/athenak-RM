"""Independent constant-background transport references; no simulation imports.

AthenaK uses T=p/rho and q=-rho*alpha*grad(T). Thus chi=(gamma-1)*alpha.
alpha_aniso denotes an additional parallel coefficient above alpha_iso.
Hyperviscosity symbols apply only to the specified scalar/composed operator.
"""
import cmath
import math


def nonnegative(value):
    if not math.isfinite(value) or value < 0:
        raise ValueError("coefficient must be finite and nonnegative")
    return value


def vector(value):
    value = tuple(value)
    if len(value) != 3 or not all(math.isfinite(x) for x in value):
        raise ValueError("expected a finite three-vector")
    return value


def conduction_rate(k, field, gamma, alpha_iso, alpha_aniso):
    k, field = vector(k), vector(field)
    nonnegative(alpha_iso)
    nonnegative(alpha_aniso)
    if not math.isfinite(gamma) or gamma <= 1:
        raise ValueError("gamma must exceed one")
    bnorm = math.hypot(*field)
    # The constant-coefficient prototype defines zero anisotropic flux at B=0.
    parallel = sum(ki*(bi/bnorm) for ki, bi in zip(k, field)) if bnorm else 0.0
    return (gamma-1)*(alpha_iso*sum(x*x for x in k)+alpha_aniso*parallel**2)


def hyperviscous_rate(k, nu4, spacing=None):
    k = vector(k)
    nonnegative(nu4)
    if spacing is None:
        laplacian = sum(x*x for x in k)
    else:
        spacing = vector(spacing)
        if min(spacing) <= 0:
            raise ValueError("spacing must be positive")
        laplacian = sum(4*math.sin(ki*dx/2)**2/dx**2
                        for ki, dx in zip(k, spacing))
    return nu4*laplacian**2


def sound_roots(kmag, sound_speed, longitudinal_rate):
    """exp(i k.x-i omega t), omega^2+i D omega-c_s^2 k^2=0."""
    for value in (kmag, sound_speed, longitudinal_rate):
        nonnegative(value)
    root = cmath.sqrt((kmag*sound_speed)**2-longitudinal_rate**2/4)
    return root-0.5j*longitudinal_rate, -root-0.5j*longitudinal_rate


def read_metrics(path):
    lines = [line for line in path.read_text().splitlines()
             if line.strip() and not line.startswith("#")]
    if len(lines) != 1:
        raise ValueError("expected exactly one diagnostic row")
    values = [float(value) for value in lines[0].split()]
    if len(values) != 7 or not all(math.isfinite(v) for v in values):
        raise ValueError("invalid or non-finite transport diagnostic")
    return dict(zip(("time", "mean_temp", "cosine", "sine", "min_temp",
                     "max_temp", "total_energy"), values))


def check_mode(metrics, rate, time, amplitude=0.01, temp0=1.0,
               initial_energy=2.0, relative_tolerance=0.01):
    """Absolute-to-initial-amplitude errors stay meaningful for decaying modes.

    The pilot evolves <= one e-fold; this is not a long-time rate-fitting checker.
    Missing keys fail loudly; malformed/nonfinite values never count as acceptance.
    """
    expected = amplitude*math.exp(-nonnegative(rate)*time)
    failures = []
    if not all(math.isfinite(value) for value in metrics.values()):
        return ["non-finite metric"]
    if abs(metrics["time"]-time) > 1.e-12*max(1.0, abs(time)):
        failures.append("final time")
    if abs(metrics["cosine"]-expected) > max(1.e-12, abs(amplitude)*relative_tolerance):
        failures.append("cosine amplitude")
    if abs(metrics["sine"]) > max(1.e-12, abs(amplitude)*1.e-8):
        failures.append("spurious phase")
    if abs(metrics["mean_temp"]-temp0) > 1.e-10*max(1.0, abs(temp0)):
        failures.append("mean temperature conservation")
    if abs(metrics["total_energy"]-initial_energy) > 1.e-10*max(1.0, initial_energy):
        failures.append("total energy conservation")
    if metrics["min_temp"] < temp0-abs(amplitude)-1.e-10:
        failures.append("new minimum")
    if metrics["max_temp"] > temp0+abs(amplitude)+1.e-10:
        failures.append("new maximum")
    return failures
