"""Frozen D9 initialization and D8-matched work definition."""

from benchmarks.donut_tensor_d8 import POLICY, REPRESENTATION as D8_REPRESENTATION, limits

ARMS=('D9-C','D9-T')
PROTOCOL='donut-kurt-paired-tensor-d9-v1'
REPRESENTATION=dict(D8_REPRESENTATION, initial_width=.5, initialization={
    'name':'projected_world_x_rank_one_v1','major_width':.52,'minimum_minor_width':.48})
