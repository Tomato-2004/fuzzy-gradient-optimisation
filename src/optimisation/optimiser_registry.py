from .adam_optimizer import AdamOptimizer
from .ga_optimizer import GAOptimizer
from .pso_optimizer import PSOOptimizer
from .de_optimizer import DEOptimizer
from .cmaes_optimizer import CMAESOptimizer
from .sgd_optimizer import SGDOptimizer
from .rmsprop_optimizer import RMSPropOptimizer
OPTIMISERS = {
    "adam": AdamOptimizer,
    "sgd": SGDOptimizer,
    "rmsprop": RMSPropOptimizer,
    "ga": GAOptimizer,
    "pso": PSOOptimizer,
    "de": DEOptimizer,
    "cmaes": CMAESOptimizer,
}
