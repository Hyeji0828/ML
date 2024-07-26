# Data Loader

class DataLoader:
    def __init__(self, test_ratio=0.1, val_ratio=0.1):
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.val_ratio = 1 - (test_ratio + val_ratio)

    def train_loader(self):
        pass

    def val_loader(self):
        pass

# Data Split
# K-fold
# Straitified K-fold
