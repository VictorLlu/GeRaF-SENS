# Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

def grid_num(max_val: float, min_val: float, res: float) -> int:
    """
    Calculates the number of grid points based on the specified resolution.

    This function determines how many points fit between `min_val` and `max_val` 
    using a given resolution `res`. The result is the number of grid points that 
    can be created from `min_val` to `max_val` inclusively.

    Parameters:
    ----------
    max_val : float
        The upper bound of the grid.
    min_val : float
        The lower bound of the grid.
    res : float
        The resolution or step size between grid points.

    Returns:
    -------
    int
        The total number of grid points, including both endpoints.
    """
    if res <= 0:
        raise ValueError("Resolution 'res' must be greater than zero.")
    
    return int(round((max_val - min_val) / res)) + 1
