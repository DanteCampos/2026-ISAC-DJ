import numpy as np

def db2lin(db):
    return 10**(db/10)

def lin2db(lin):
    return 10*np.log10(lin)

def polar_to_cartesian(ranges, angles):
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    return x, y

def angle_relative_to_boresight(x1, y1, x2, y2, boresight):
    """"
    Calculate angle between (x1,y1) and (x2,y2) relative to the boresight direction (in radians)
    """
    # Calculate the angle between two points relative to the boresight
    angle_to_point = np.arctan2(y2 - y1, x2 - x1)
    relative_angle = angle_to_point - boresight
    # Normalize the angle to the range [-pi, pi]
    relative_angle = (relative_angle + np.pi) % (2 * np.pi) - np.pi
    return np.abs(relative_angle)