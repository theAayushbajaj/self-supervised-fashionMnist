from torch.utils.data.dataset import Subset
from sklearn.model_selection import train_test_split

def create_subset(dataset, subset_size=0.1):
    # Generate a list of indices from 0 to len(dataset)
    indices = list(range(len(dataset)))
    
    # Use train_test_split to split indices into two parts
    subset_indices, _ = train_test_split(indices, test_size=(1 - subset_size), random_state=42, stratify=dataset.targets)
    
    # Create a subset of the original dataset
    subset = Subset(dataset, subset_indices)
    
    return subset