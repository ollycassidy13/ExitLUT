# Modified in 2026 by Oliver Cassidy, Marta Andronic and George A. Constantinides for ExitLUT: adds an explicit resource model.
# Part of the ATHEENA / fpgaConvNet optimiser, licensed under the GNU GPL v3 (see optimiser/LICENSE).
from fpgaconvnet_optimiser.models.layers import Layer
from fpgaconvnet_optimiser.tools.resource_model import bram_stream_resource_model

import pydot
import numpy as np

from fpgaconvnet_optimiser.models.modules import Squeeze

class SqueezeLayer(Layer):
    def __init__(
            self,
            rows: int,
            cols: int,
            channels: int,
            coarse_in: int,
            coarse_out: int,
            data_width: int = 16,
        ):

        # initialise parent class
        super().__init__(rows, cols, channels, coarse_in, coarse_out,
                data_width=data_width)

        # initialise modules
        self.modules["squeeze"] = Squeeze(self.rows, self.cols, self.channels, self.coarse_in,
                                          self.coarse_out)

    def layer_info(self,parameters,batch_size=1):
        Layer.layer_info(self, parameters, batch_size)

    def resource(self):
        squeeze_rsc = self.modules['squeeze'].rsc()
        return {
            "LUT":  squeeze_rsc['LUT'],
            "FF":   squeeze_rsc['FF'],
            "DSP":  squeeze_rsc['DSP'],
            "BRAM": bram_stream_resource_model(self.buffer_depth, self.data_width) * self.streams_in(),
        }

    def update(self):
        self.modules["squeeze"].rows = self.rows
        self.modules["squeeze"].cols = self.cols
        self.modules["squeeze"].channels = self.channels
        self.modules["squeeze"].coarse_in = self.coarse_in
        self.modules["squeeze"].coarse_out = self.coarse_out

    def visualise(self,name):
        cluster = pydot.Cluster(name,label=name)

        # add squeeze module
        cluster.add_node(pydot.Node( "_".join([name,"squeeze"]), label="squeeze" ))

        # get nodes in and out
        nodes_in  = [ "_".join([name,"squeeze"]) for i in range(self.streams_in()) ]
        nodes_out = [ "_".join([name,"squeeze"]) for i in range(self.streams_out()) ]

        # return module
        return cluster, nodes_in, nodes_out

#what goes in must come out
    def functional_model(self,data,batch_size=1):

        batched_flag=False
        print(data.shape)
        if len(data.shape) > 3:
            batched_flag=True
            assert data.shape[1] == self.rows_in()    , "ERROR (data): invalid row dimension"
            assert data.shape[2] == self.cols_in()    , "ERROR (data): invalid column dimension"
            assert data.shape[3] == self.channels_in(), "ERROR (data): invalid channel dimension"
        else:
            assert data.shape[0] == self.rows_in()    , "ERROR (data): invalid row dimension"
            assert data.shape[1] == self.cols_in()    , "ERROR (data): invalid column dimension"
            assert data.shape[2] == self.channels_in(), "ERROR (data): invalid channel dimension"

        # return output featuremap
        if batched_flag:
            data = np.moveaxis(data, -1, 1)
            print(data.shape)
        else:
            data = np.moveaxis(data, -1, 0)
            # FIXME clean up use of batch size here
            data = np.repeat(data[np.newaxis,...], batch_size, axis=0)

        return data
