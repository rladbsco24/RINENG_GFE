"""Cache invariant scattered-wave geometry without changing force numerics.

The adapter preserves the original term multiplication and accumulation order.
It changes neither the physical model nor root-search or quadrature settings.
"""
from __future__ import annotations

from collections import OrderedDict
import math
from weakref import WeakKeyDictionary

import numpy as np

from hat_revision_pipeline import exact_validator as _validator

_MAX_GRIDS_PER_EVALUATOR = 16
_BASIS_CACHES = WeakKeyDictionary()


def _scattered_pressure_cached(self, offsets_m, coefficients):
    offsets = np.asarray(offsets_m, dtype=float).reshape(-1, 3)
    # These quantities fully determine every cached geometry-dependent value.
    # Scatter amplitudes remain outside the cache and are read on each call.
    key = (
        self.k_rad_m,
        self.sphere.radius_m,
        tuple(self.lm),
        offsets.shape,
        offsets.tobytes(),
    )
    cache = _BASIS_CACHES.setdefault(self, OrderedDict())
    basis = cache.get(key)
    if basis is None:
        radius = np.linalg.norm(offsets, axis=1)
        if np.any(radius <= self.sphere.radius_m):
            raise ValueError("scattered pressure requested on or inside the sphere")
        theta = np.arccos(np.clip(offsets[:, 2] / radius, -1.0, 1.0))
        phi = np.mod(np.arctan2(offsets[:, 1], offsets[:, 0]), 2.0 * math.pi)
        kr = self.k_rad_m * radius
        hankel = {}
        harmonics = []
        for ell, m in self.lm:
            if ell not in hankel:
                hankel[ell] = _validator._hankel1(ell, kr)
                hankel[ell].setflags(write=False)
            harmonic = _validator.sph_harm_y(ell, m, theta, phi)
            harmonic.setflags(write=False)
            harmonics.append(harmonic)
        basis = (hankel, tuple(harmonics))
        cache[key] = basis
        if len(cache) > _MAX_GRIDS_PER_EVALUATOR:
            cache.popitem(last=False)
    else:
        cache.move_to_end(key)
    hankel, harmonics = basis
    result = np.zeros(len(offsets), dtype=np.complex128)
    for index, (ell, m) in enumerate(self.lm):
        # Do not combine factors or replace this loop with a matrix product:
        # the original floating-point multiplication/summation order matters.
        result += (
            coefficients[index]
            * self.scatter[ell]
            * hankel[ell]
            * harmonics[index]
        )
    return result


def install():
    """Install the process-local equivalent calculation once; return status."""
    evaluator = _validator.PartialWaveForceEvaluator
    if evaluator._scattered_pressure is _scattered_pressure_cached:
        return False
    evaluator._scattered_pressure = _scattered_pressure_cached
    return True
