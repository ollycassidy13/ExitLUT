#!/bin/bash
# Regenerates fpgaconvnet_pb2.py from fpgaconvnet.proto. Only needed if the .proto changes.
# Requires protoc 3.15.x (e.g. `sudo apt install protobuf-compiler`, or download it from
# https://github.com/protocolbuffers/protobuf/releases/tag/v3.15.8). The checked-in
# fpgaconvnet_pb2.py was generated with 3.15.8 and matches protobuf==3.20.3 at runtime.
protoc --python_out=. fpgaconvnet.proto
