import numpy as np
from scipy.spatial import KDTree


class PatchSampler:
    """
    Spatial patch sampler for MAE
    Samples a fixed number of spatially connected spots
    """
    def __init__(self, sampling_method: str = 'nearest', fixed_n_spots: int = 128):
        """
        Args:
            sampling_method: 'nearest' for spatial patches, 'random' for random sampling
            fixed_n_spots: Fixed number of spots to sample (required for MAE)
        """
        self.sampling_method = sampling_method
        self.fixed_n_spots = fixed_n_spots

    def sample_nearest_patch(self, coords, num_samples):
        """Sample nearest neighbors as a spatial patch"""
        num_samples = min(len(coords), num_samples)

        if num_samples == len(coords):
            return np.arange(len(coords))

        # Build a KDTree for efficient nearest neighbor searches
        tree = KDTree(coords)
        
        # Randomly choose a center point for the patch
        center_idx = np.random.randint(0, len(coords))
        center_coord = coords[center_idx]
        
        # Find the nearest num_samples points including the center
        _, idx_nearest = tree.query(center_coord, k=num_samples)
        
        return idx_nearest
    
    def sample_random_patch(self, coords, num_samples):
        """Random sampling without spatial constraint"""
        num_samples = min(len(coords), num_samples)
        return np.random.choice(len(coords), size=num_samples, replace=False)

    def __call__(self, coords):
        """
        Sample spots from coordinates
        Args:
            coords: numpy array of shape [n_spots, 2]
        Returns:
            indices: numpy array of selected spot indices
        """
        # Always use fixed number of spots for MAE
        total_samples = min(self.fixed_n_spots, len(coords))
        
        if self.sampling_method == 'nearest':
            return self.sample_nearest_patch(coords, total_samples)
        elif self.sampling_method == 'random':
            return self.sample_random_patch(coords, total_samples)
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")