import itertools
from typing import Dict, List, Any

class GridSearch:
    """Simple grid search utility for iterating over parameter grids."""
    def __init__(self, param_grid: Dict[str, List[Any]]):
        self.param_grid = param_grid
        
    def generate_configs(self):
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))
