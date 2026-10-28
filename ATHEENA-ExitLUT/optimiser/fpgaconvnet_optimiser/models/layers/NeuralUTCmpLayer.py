# New file added in 2026 by Oliver Cassidy, Marta Andronic and George A. Constantinides for ExitLUT (exit-decision model).
# Part of the ATHEENA / fpgaConvNet optimiser, licensed under the GNU GPL v3 (see optimiser/LICENSE).
import numpy as np
import pydot
from typing import List

from fpgaconvnet_optimiser.models.modules import Fork
from fpgaconvnet_optimiser.models.modules.NeuralUT import NeuralUT
from fpgaconvnet_optimiser.models.layers import Layer


class NeuralUTCmpLayer(Layer):
    def __init__(
            self,
            rows: int,
            cols: int,
            channels: int,
            ctrledges: List[str],
            threshold: float,
            coarse_in: int = 1,
            coarse_out: int = 1,
            data_width: int = 16
        ):
        super().__init__(rows, cols, channels, coarse_in, coarse_out,
                         data_width=data_width)

        self._ctrledges = ctrledges
        self._threshold = threshold
        self._coarse_in = coarse_in
        self._coarse_out = coarse_out
        self.ctrl_out_size = len(ctrledges)

        self.modules['neuralut']  = NeuralUT(self.rows_in(), self.cols_in(),
                                             self.channels_in(),
                                             threshold=self._threshold)
        self.modules['fork_out']  = Fork(self.rows_out(), self.cols_out(),
                                         self.channels_in(), 1,
                                         self.ctrl_out_size)
        self.update()


    def rows_out(self):
        return 1

    def cols_out(self):
        return 1

    def channels_out(self):
        return self.ctrl_out_size


    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def ctrledges(self) -> List[str]:
        return self._ctrledges

    @property
    def coarse_in(self) -> int:
        return self._coarse_in

    @property
    def coarse_out(self) -> int:
        return self._coarse_out

    @coarse_in.setter
    def coarse_in(self, val: int) -> None:
        self._coarse_in = val
        self.update()

    @coarse_out.setter
    def coarse_out(self, val: int) -> None:
        self._coarse_out = val
        self.update()


    def layer_info(self, parameters, batch_size=1):
        Layer.layer_info(self, parameters, batch_size)
        parameters.channels_out  = self.channels_out() + 1
        parameters.threshold     = self.threshold
        parameters.ctrl_out_size = self.ctrl_out_size + 1
        parameters.ctrledges.extend(self.ctrledges)


    def update(self):
        self.modules['neuralut'].rows     = self.rows_in()
        self.modules['neuralut'].cols     = self.cols_in()
        self.modules['neuralut'].channels = int(self.channels_in() / self.coarse_in)

        self.modules['fork_out'].rows     = self.rows_out()
        self.modules['fork_out'].cols     = self.cols_out()
        self.modules['fork_out'].channels = int(self.channels_in() / self.coarse_in)


    def resource(self):
        assert self.coarse_in == 1, (
            f"NeuralUTCmpLayer hardware is fixed at coarse_in=1; got {self.coarse_in}")
        return self.modules['neuralut'].rsc()


    def pipeline_depth(self):
        return self.modules['neuralut'].pipeline_depth()


    def visualise(self, name):
        cluster = pydot.Cluster(name, label=name)
        for i in range(self.coarse_in):
            cluster.add_node(pydot.Node(
                "_".join([name, "neuralut", str(i)]), label="neuralut"))
            cluster.add_node(pydot.Node(
                "_".join([name, "fork_o", str(i)]), label="fork_o"))
            cluster.add_edge(pydot.Edge(
                "_".join([name, "neuralut", str(i)]),
                "_".join([name, "fork_o", str(i)])))

        nodes_in  = ["_".join([name, "neuralut", str(i)])
                     for i in range(self.coarse_in)]
        nodes_out = ["_".join([name, "fork_o", str(i)])
                     for i in range(len(self.ctrledges))]
        return cluster, nodes_in, nodes_out


    def functional_model(self, data, batch_size=1):
        data = data.astype(np.float32)

        batched_flag = len(data.shape) > 3
        if batched_flag:
            assert data.shape[0] == batch_size,           "ERROR: invalid batch"
            assert data.shape[1] == self.rows_in(),       "ERROR: invalid rows"
            assert data.shape[2] == self.cols_in(),       "ERROR: invalid cols"
            assert data.shape[3] == self.channels_in(),   "ERROR: invalid channels"
        else:
            assert data.shape[0] == self.rows_in(),       "ERROR: invalid rows"
            assert data.shape[1] == self.cols_in(),       "ERROR: invalid cols"
            assert data.shape[2] == self.channels_in(),   "ERROR: invalid channels"

        if batched_flag:
            logits = data.reshape(batch_size, -1)
        else:
            logits = data.reshape(1, -1)

        exit_bits = self.modules['neuralut'].functional_model(logits)

        out = np.zeros((batch_size, self.ctrl_out_size))
        for b in range(batch_size):
            for c in range(self.ctrl_out_size):
                out[b][c] = float(exit_bits[b] if b < len(exit_bits) else 0.0)
        return out
