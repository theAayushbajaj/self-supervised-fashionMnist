from torch.utils.data.dataset import Subset
from sklearn.model_selection import train_test_split

def create_subset(dataset, subset_size=0.1):
    indices = list(range(len(dataset)))
    subset_indices, _ = train_test_split(indices, test_size=(1 - subset_size), random_state=42, stratify=dataset.targets)
    subset = Subset(dataset, subset_indices)
    
    return subset