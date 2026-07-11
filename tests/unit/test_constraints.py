import unittest
import torch
from communication.constraints import CommunicationConstraints

class TestCommunicationConstraints(unittest.TestCase):
    def setUp(self):
        self.constraints = CommunicationConstraints(vocab_size=10, max_length=5)
        
    def test_enforce_length(self):
        symbols = torch.randint(0, 10, (2, 10)) # batch_size=2, length=10
        truncated = self.constraints.enforce_length(symbols)
        self.assertEqual(truncated.size(1), 5)
        
    def test_enforce_vocab(self):
        symbols = torch.tensor([[1, 5, 12, -2]])
        clipped = self.constraints.enforce_vocab(symbols)
        self.assertTrue((clipped >= 0).all())
        self.assertTrue((clipped < 10).all())

if __name__ == '__main__':
    unittest.main()
