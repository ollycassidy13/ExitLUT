# Modified in 2026 by Oliver Cassidy, Marta Andronic and George A. Constantinides for ExitLUT: registers the NeuralUT module.
# Part of the ATHEENA / fpgaConvNet optimiser, licensed under the GNU GPL v3 (see optimiser/LICENSE).
"""
These are the basic building blocks of the accelerator.
"""

from .Module import Module
from .Accum import Accum
from .BatchNorm import BatchNorm
from .Conv import Conv
from .Fork import Fork
from .Glue import Glue
from .Pool import Pool
from .ReLU import ReLU
from .SlidingWindow import SlidingWindow
from .Squeeze import Squeeze
from .Bias import Bias
#EE modules
from .Buffer import Buffer
from .ReduceMax import ReduceMax
from .Compare import Compare
from .Exponential import Exponential
#from .Div import Div
from .SoftMaxSum import SoftMaxSum
from .ExitMerge import ExitMerge
from .NeuralUT import NeuralUT
