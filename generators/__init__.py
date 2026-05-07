from .branch_fan_in_aggregation import materialize as materialize_branch_fan_in_aggregation
from .linear_elementwise_chain import materialize as materialize_linear_elementwise_chain

REGISTRY = {
    "branch_fan_in_aggregation": materialize_branch_fan_in_aggregation,
    "linear_elementwise_chain": materialize_linear_elementwise_chain,
}
