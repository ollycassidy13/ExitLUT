# Modified in 2026 by Oliver Cassidy, Marta Andronic and George A. Constantinides for ExitLUT: adds utilisation and rate models.
# Part of the ATHEENA / fpgaConvNet optimiser, licensed under the GNU GPL v3 (see optimiser/LICENSE).
import numpy as np
import math
import os
import sys
from dataclasses import dataclass, field

from fpgaconvnet_optimiser.models.modules import Module

@dataclass
class Squeeze(Module):
    coarse_in: int
    coarse_out: int

    def __post_init__(self):
        # load the resource model coefficients
        self.rsc_coef["LUT"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/squeeze_lut.npy"))
        self.rsc_coef["FF"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/squeeze_ff.npy"))
        self.rsc_coef["BRAM"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/squeeze_bram.npy"))
        self.rsc_coef["DSP"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/squeeze_dsp.npy"))

    def utilisation_model(self):
        return {
            # LUT: [coarse_in, coarse_out, channels]
            # NNLS fit on 13 rsc55 Vivado synthesis measurements (per-tile)
            "LUT":  np.array([float(self.coarse_in), float(self.coarse_out), float(self.channels)]),
            "FF":   np.array([float(self.coarse_in), float(self.coarse_out), float(self.channels)]),
            "DSP":  np.array([1]),
            "BRAM": np.array([1]),
        }

    def rate_in(self):
        channel_groups = math.ceil(self.channels / self.coarse_in)
        return self.channels / (self.coarse_in * channel_groups)

    def rate_out(self):
        channel_groups = math.ceil(self.channels / self.coarse_out)
        return self.channels / (self.coarse_out * channel_groups)

    def module_info(self):
        # get the base module fields
        info = Module.module_info(self)
        # add module-specific info fields
        info["coarse_in"] = self.coarse_in
        info["coarse_out"] = self.coarse_out
        # return the info
        return info

    def lcm(a, b):
        return abs(a*b) // math.gcd(a, b)

    def functional_model(self, data):
        # check input dimensionality
        assert data.shape[0] == self.rows                       , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols                       , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels//self.coarse_in   , "ERROR: invalid channel dimension"
        assert data.shape[3] == self.coarse_in                  , "ERROR: invalid coarse dimension"

        out = np.ndarray((
            self.rows,
            self.cols,
            self.channels//self.coarse_out,
            self.coarse_out),dtype=float)

        out = np.reshape(data,(self.rows,self.cols,self.channels))
        out = np.reshape(data,(self.rows,self.cols,self.channels//self.coarse_out,self.coarse_out))

        return out


