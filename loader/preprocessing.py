import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import rotate
from scipy.ndimage import shift
from scipy.ndimage import zoom

import torch

from utils import get_series_properties


INPUT_DIM = 224
MEAN = 58.09
STDDEV = 49.73


def normalize_slice(exam_slice: np.ndarray, eps=1e-11) -> np.ndarray:
    """Normalize a single slice

    args:
        exam_slice: numpy array containing a slice, of shape (length, width)
        eps: small value to avoid division by zero

    returns:
        numpy array containing a normalized slice, of shape (length, width)
    """
    mean = exam_slice.mean().item()
    std = exam_slice.std().item()

    return (exam_slice - mean) / (std + eps)


def normalize_slice_by_slice(vol: np.ndarray) -> np.ndarray:
    """Normalize a volume slice by slice, using the mean and std of each slice and
    not the entire volume.

    args:
        vol: numpy array containing a volume, of shape (num_slices, length, width)

    returns:
        numpy array containing a normalized volume, of shape (num_slices, length, width)
    """
    for slice_idx in range(vol.shape[0]):
        vol[slice_idx, :, :] = normalize_slice(vol[slice_idx, :, :])
    return vol


def reshape_slice(slice_data: np.ndarray, series_key, zero_pad=True):
    series_props = get_series_properties(series_key)
    crop_size = series_props["crop_size"]
    slice_shape = (slice_data.shape[0], slice_data.shape[1])
    crop_dims = (
        (slice_shape[0] - crop_size[0]) // 2,
        (slice_shape[1] - crop_size[1]) // 2,
    )

    if slice_shape[0] > crop_size[0]:
        slice_data = slice_data[
            crop_dims[0] : crop_dims[0] + crop_size[0],
            :,
        ]
    elif zero_pad:
        slice_data = np.pad(
            slice_data,
            (
                (-crop_dims[0], -crop_dims[0]),
                (0, 0),
            ),
        )
    if slice_shape[1] > crop_size[1]:
        slice_data = slice_data[
            :,
            crop_dims[1] : crop_dims[1] + crop_size[1],
        ]
    elif zero_pad:
        slice_data = np.pad(
            slice_data,
            (
                (0, 0),
                (-crop_dims[1], -crop_dims[1]),
            ),
        )

    return slice_data


def reshape_volume(vol, series_key, zero_pad=True):
    series_props = get_series_properties(series_key)
    crop_size = series_props["crop_size"]
    num_slices = series_props["num_slices"]

    if zero_pad:
        center_cropped_vol = np.zeros((crop_size[0], crop_size[1], vol.shape[-1]))
        for slice_idx in range(vol.shape[-1]):
            center_cropped_vol[:, :, slice_idx] = reshape_slice(
                vol[:, :, slice_idx], series_key, zero_pad=True
            )

        if center_cropped_vol.shape[-1] < num_slices:
            slices_to_add = [
                (num_slices - center_cropped_vol.shape[-1]) // 2,
                num_slices
                - (
                    center_cropped_vol.shape[-1]
                    + ((num_slices - center_cropped_vol.shape[-1]) // 2)
                ),
            ]
            depth_adjusted_vol = np.pad(
                center_cropped_vol,
                (
                    (0, 0),
                    (0, 0),
                    (slices_to_add[0], slices_to_add[1]),
                ),
            )
        elif center_cropped_vol.shape[-1] > num_slices:
            slices_to_keep = (
                (center_cropped_vol.shape[-1] - num_slices) // 2,
                ((center_cropped_vol.shape[-1] - num_slices) // 2) + num_slices,
            )
            depth_adjusted_vol = center_cropped_vol[
                :,
                :,
                slices_to_keep[0] : slices_to_keep[1],
            ]
        else:
            depth_adjusted_vol = center_cropped_vol

    else:
        center_cropped_vol = np.zeros(
            (
                np.min([vol.shape[0], crop_size[0]]),
                np.min([vol.shape[1], crop_size[1]]),
                vol.shape[-1],
            )
        )
        for slice_idx in range(vol.shape[-1]):
            center_cropped_vol[:, :, slice_idx] = reshape_slice(
                vol[:, :, slice_idx], series_key, zero_pad=False
            )

        if center_cropped_vol.shape[-1] > num_slices:
            slices_to_keep = (
                (center_cropped_vol.shape[-1] - num_slices) // 2,
                ((center_cropped_vol.shape[-1] - num_slices) // 2) + num_slices,
            )
            depth_adjusted_vol = center_cropped_vol[
                :,
                :,
                slices_to_keep[0] : slices_to_keep[1],
            ]
        else:
            depth_adjusted_vol = center_cropped_vol

        target_shape = (crop_size[0], crop_size[1], num_slices)
        scale_factors = np.array(target_shape) / np.array(depth_adjusted_vol.shape)
        depth_adjusted_vol = zoom(depth_adjusted_vol, scale_factors)

    rotated_vol = np.moveaxis(depth_adjusted_vol, -1, 0)

    return rotated_vol


def scale_volume(vol, series_key):
    series_props = get_series_properties(series_key)
    pixel_range = series_props["pixel_range"]
    vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol))
    vol = vol * (pixel_range[1] - pixel_range[0]) + pixel_range[0]
    return vol


def augment_slice(
    exam_slice: np.ndarray,
    flip_prob=0.5,
    rotate_prob=0.5,
    shear_prob=0.5,
    filter_prob=0.5,
) -> np.ndarray:
    """Augment a single slice

    args:
        exam_slice: numpy array containing a slice, of shape (length, width)
        flip_prob: probability of flipping the slice
        rotate_prob: probability of rotating the slice
        shear_prob: probability of shearing the slice
        filter_prob: probability of applying a gaussian filter to the slice

    returns:
        numpy array containing an augmented slice, of shape (length, width)
    """
    augmented_slice = exam_slice.copy()

    # flip
    if np.random.random() < flip_prob:
        augmented_slice = np.fliplr(augmented_slice)

    # rotate
    if np.random.random() < rotate_prob:
        angle = np.random.uniform(-180, 180)
        augmented_slice = rotate(augmented_slice, angle, reshape=False)

    # shear
    if np.random.random() < shear_prob:
        shear_range = np.random.uniform(-0.2, 0.2, size=2)
        augmented_slice = shift(augmented_slice, shear_range)

    # gaussian filter
    if np.random.random() < filter_prob:
        sigma = np.random.uniform(0.5, 2.0)
        augmented_slice = gaussian_filter(augmented_slice, sigma)

    return augmented_slice


def augment_slice_by_slice(vol: np.ndarray) -> np.ndarray:
    """Augment a volume slice by slice

    args:
        vol: numpy array containing a volume, of shape (num_slices, length, width)

    returns:
        numpy array containing an augmented volume, of shape (num_slices, length, width)
    """
    for slice_idx in range(vol.shape[0]):
        vol[slice_idx, :, :] = augment_slice(vol[slice_idx, :, :])
    return vol


def normalize_volume(vol: np.ndarray) -> np.ndarray:
    """Normalize a volume, just a wrapper for normalize_slice.

    args:
        vol: numpy array containing a volume, of shape (num_slices, length, width)

    returns:
        numpy array containing a normalized volume, of shape (num_slices, length, width)
    """
    return normalize_slice(vol)


def augment_volume(vol: np.ndarray, flip_prob=0.5, shear_prob=0.5) -> np.ndarray:
    """Augment a volume

    args:
        vol: numpy array containing a volume, of shape (num_slices, length, width)

    returns:
        numpy array containing an augmented volume, of shape (num_slices, length, width)
    """
    augmented_vol = vol.copy()
    # flip the volume
    if np.random.random() < flip_prob:
        for i in range(vol.shape[0]):
            augmented_vol[i, :, :] = np.fliplr(vol[i, :, :])

    # shear the volume
    vol = augmented_vol.copy()
    if np.random.random() < shear_prob:
        shear_range = np.random.uniform(-0.1, 0.1, size=2)
        for i in range(vol.shape[0]):
            augmented_vol[i, :, :] = shift(vol[i, :, :], shear_range)

    return augmented_vol


def augment_volume_with_monai(vol: np.ndarray) -> np.ndarray:
    """Augment a volume using MONAI

    args:
        vol: numpy array containing a volume, of shape (num_slices, length, width)

    returns:
        numpy array containing an augmented volume, of shape (num_slices, length, width)
    """

    from monai.transforms import (
        Compose,
        RandFlip,
        RandRotate,
        RandZoom,
        RandGaussianSmooth,
        RandGaussianNoise,
        RandAdjustContrast,
        RandBiasField,
        RandAffine,
    )
        
    # Define MONAI augmentations
    transforms = Compose([
        RandFlip(spatial_axis=[2], prob=0.5),  # Flip along the first axis
        RandRotate(range_x=5, range_y=5, range_z=5, prob=0.5),  # Rotate in 3D
        RandZoom(min_zoom=0.95, max_zoom=1.05, prob=0.5),  # Scale/zoom
        RandGaussianSmooth(sigma_x=(0.2, 0.8), sigma_y=(0.2, 0.8), sigma_z=(0.5, 1.0), prob=0.5),  # Smooth
        RandGaussianNoise(mean=0.0, std=0.01, prob=0.5),  # Add noise
    ])

    # Apply augmentations
    augmented_vol = transforms(vol[np.newaxis, ...])  # Add batch dim for MONAI transforms
    return augmented_vol[0]  # Remove batch dim


def augment_volume_with_torchio(vol: np.ndarray, flip_prob=0.5, shear_prob=0.5) -> np.ndarray:
    """Augment a volume using TorchIO

    args:
        vol: numpy array containing a volume, of shape (num_slices, length, width)
        flip_prob: probability of flipping the volume
        shear_prob: probability of shearing the volume

    returns:
        numpy array containing an augmented volume, of shape (num_slices, length, width)
    """

    import torchio as tio
    from torchio import Subject, ScalarImage, Compose, RandomFlip, RandomAffine, RandomBlur

    # Define TorchIO transformations
    transforms = Compose([
        RandomFlip(axes=(0, 1), flip_probability=flip_prob),  # Flip along axes
        RandomAffine(scales=(0.9, 1.1), degrees=(0, 10), isotropic=False),  # Shear/scale/rotate
        RandomBlur(std=(0.5, 2.0)),  # Gaussian blur
    ])
    
    # Wrap volume in TorchIO Subject
    subject = Subject(image=ScalarImage(tensor=torch.from_numpy(vol).unsqueeze(0)))
    
    # Apply transformations
    transformed = transforms(subject)
    augmented_vol = transformed['image'].numpy().squeeze(0)

    return augmented_vol


def preprocess_volume(
    vol: np.ndarray,
    series_key,
    normalize=True,
    augment=False,
    slice_by_slice=False,
    zero_pad=True,
    config=None,
) -> np.ndarray:
    vol = reshape_volume(vol, series_key, zero_pad=zero_pad)

    if normalize:
        vol = normalize_slice_by_slice(vol) if slice_by_slice else normalize_volume(vol)
        if augment:
            if config["data"]["augment_type"] == "slice_by_slice":
                vol = augment_slice_by_slice(vol) if slice_by_slice else augment_volume(vol)
            if config["data"]["augment_type"] == "monai":
                vol = augment_volume_with_monai(vol)
            elif config["data"]["augment_type"] == "torchio":
                vol = augment_volume_with_torchio(vol)
            else:
                raise ValueError(f"Unsupported augment_type: {config['data']['augment_type']}")

    return vol
