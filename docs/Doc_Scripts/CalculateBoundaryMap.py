import numpy as np
import tifffile
import argparse
import os
from scipy import ndimage


def compute_boundary_map(label_img, thickness=3):
    boundary = np.zeros_like(label_img, dtype=bool)
    
    # Create structuring element for dilation
    struct = np.ones((thickness, thickness, thickness), dtype=bool)
    
    for axis in range(3):
        print(f"Processing axis {axis}")
        for shift in [-1, 1]:
            print(f"Processing shift {shift}")
            slicer1 = [slice(None)] * 3
            slicer2 = [slice(None)] * 3
            
            if shift == -1:
                slicer1[axis] = slice(1, None)
                slicer2[axis] = slice(0, -1)
            else:
                slicer1[axis] = slice(0, -1)
                slicer2[axis] = slice(1, None)
                
            diff = label_img[tuple(slicer1)] != label_img[tuple(slicer2)]
            
            # Apply to both sides of the boundary
            boundary[tuple(slicer1)] |= diff
            boundary[tuple(slicer2)] |= diff
    
    # Dilate the boundary by the specified thickness
    if thickness > 1:
        boundary = ndimage.binary_dilation(boundary, structure=struct, iterations=thickness//2).astype(np.uint8)
    
    return boundary.astype(np.uint8)


def remove_small_components(boundary_map, min_size=100):
    """
    Remove connected components smaller than min_size voxels.
    
    Parameters:
    -----------
    boundary_map : np.ndarray
        Binary boundary map (0s and 1s)
    min_size : int
        Minimum component size in voxels. Components smaller than this will be removed.
    
    Returns:
    --------
    np.ndarray
        Filtered boundary map with small components removed
    """
    print(f"Removing components smaller than {min_size} voxels...")
    
    # Label connected components
    labeled_components, num_components = ndimage.label(boundary_map)
    
    print(f"Found {num_components} connected components")
    
    # Count voxels in each component
    component_sizes = np.bincount(labeled_components.ravel())
    
    # Identify components to keep (size >= min_size)
    # Skip component 0 (background)
    keep_components = component_sizes >= min_size
    keep_components[0] = False  # Always remove background
    
    # Create filtered map
    filtered_map = keep_components[labeled_components]
    
    num_removed = num_components - np.sum(keep_components)
    print(f"Removed {num_removed} small components, kept {np.sum(keep_components)}")
    
    return filtered_map.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Compute boundary map from 3D label image (multiclass supported).")
    parser.add_argument("input_path", type=str, help="Path to input 3D label .tif image")
    parser.add_argument("--thickness", type=int, default=3, help="Boundary thickness (default: 3)")
    parser.add_argument("--min_component_size", type=int, default=0, 
                        help="Minimum component size in voxels. Components smaller than this will be removed (default: 0, no filtering)")
    
    args = parser.parse_args()
    
    print(f"Computing boundary map with thickness {args.thickness}...")
    label_img = tifffile.imread(args.input_path)
    boundary_map = compute_boundary_map(label_img, thickness=args.thickness)
    
    # Remove small components if threshold is specified
    if args.min_component_size > 0:
        boundary_map = remove_small_components(boundary_map, min_size=args.min_component_size)
    
    output_path = os.path.splitext(args.input_path)[0] + f"_boundary_t{args.thickness}.tif"
    if args.min_component_size > 0:
        output_path = os.path.splitext(args.input_path)[0] + f"_boundary_t{args.thickness}_minsize{args.min_component_size}.tif"
    
    tifffile.imwrite(output_path, boundary_map)
    print(f"Boundary map saved to: {output_path}")


if __name__ == "__main__":
    main()
