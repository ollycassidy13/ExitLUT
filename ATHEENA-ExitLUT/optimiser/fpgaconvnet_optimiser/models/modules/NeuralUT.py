# New file added in 2026 by Oliver Cassidy, Marta Andronic and George A. Constantinides for ExitLUT (exit-decision model).
# Part of the ATHEENA / fpgaConvNet optimiser, licensed under the GNU GPL v3 (see optimiser/LICENSE).
import numpy as np
from dataclasses import dataclass, field
from fpgaconvnet_optimiser.models.modules import Module


@dataclass
class NeuralUT(Module):
    threshold: float = 0.0

    _LUT_COUNT:  int = field(default=1241, init=False, repr=False)
    _FF_COUNT:   int = field(default=460,  init=False, repr=False)
    _DSP_COUNT:  int = field(default=0,    init=False, repr=False)
    _BRAM_COUNT: int = field(default=0,    init=False, repr=False)

    def rate_in(self):
        return 1.0

    def rate_out(self):
        return 1.0 / self.channels

    def pipeline_depth(self):
        return 12

    def rsc(self, coef=None):
        return {
            "LUT":  self._LUT_COUNT,
            "FF":   self._FF_COUNT,
            "BRAM": self._BRAM_COUNT,
            "DSP":  self._DSP_COUNT,
        }

    def functional_model(self, data):
        data = np.atleast_2d(data).astype(np.float64)
        e = np.exp(data - data.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        return (probs.max(axis=1) >= self.threshold).astype(np.float32)
