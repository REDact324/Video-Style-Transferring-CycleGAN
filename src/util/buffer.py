import random
import torch

# Build up buffer for data feeding
class ReplayBuffer():
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data_ = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data_) < self.max_size:
                self.data_.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data_[i].clone())
                    self.data_[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)
