class BenchmarkSuite:
    """Automated benchmark runner for different evaluation metrics."""
    def __init__(self, runner, test_datasets: list):
        self.runner = runner
        self.test_datasets = test_datasets
        
    def run_generalization_test(self):
        """Tests the model on held-out or out-of-distribution datasets."""
        pass
        
    def run_robustness_test(self):
        """Tests the model against various levels of channel noise."""
        pass
