import xarray as xr
import pandas as pd
import numpy as np
import os
import warnings


def earth_radius(lat):
    '''
    calculate radius of Earth assuming oblate spheroid
    defined by WGS84
    
    Input
    ---------
    lat: vector or latitudes in degrees  
    
    Output
    ----------
    r: vector of radius in meters
    
    Notes
    -----------
    WGS84: https://earth-info.nga.mil/GandG/publications/tr8350.2/tr8350.2-a/Chapter%203.pdf

    Code from Kevin Schwarzwald https://github.com/ks905383
    '''
    from numpy import deg2rad, sin, cos

    # define oblate spheroid from WGS84
    a = 6378137
    b = 6356752.3142
    e2 = 1 - (b**2/a**2)
    
    # convert from geodecic to geocentric
    # see equation 3-110 in WGS84
    lat = deg2rad(lat)
    lat_gc = np.arctan( (1-e2)*np.tan(lat) )

    # radius equation
    # see equation 3-107 in WGS84
    r = (
        (a * (1 - e2)**0.5) 
         / (1 - (e2 * np.cos(lat_gc)**2))**0.5 
        )

    return r

def area_grid(lat, lon):
    """
    Calculate the area of each grid cell
    Area is in square meters
    
    Input
    -----------
    lat: vector of latitude in degrees
    lon: vector of longitude in degrees
    
    Output
    -----------
    area: grid-cell area in square-meters with dimensions, [lat,lon]
    
    Notes
    -----------
    Based on the function in
    https://github.com/chadagreene/CDT/blob/master/cdt/cdtarea.m

    Code from Kevin Schwarzwald https://github.com/ks905383
    """
    from numpy import meshgrid, deg2rad, gradient, cos
    from xarray import DataArray

    xlon, ylat = meshgrid(lon, lat)
    R = earth_radius(ylat)

    dlat = deg2rad(gradient(ylat, axis=0))
    dlon = deg2rad(gradient(xlon, axis=1))

    dy = dlat * R
    dx = dlon * R * cos(deg2rad(ylat))

    area = dy * dx

    xda = DataArray(
        area,
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
        attrs={
            "long_name": "area_per_pixel",
            "description": "area per pixel",
            "units": "m^2",
        },
    )
    return xda

def area_mean(ds):
    """
    Calculate area-weighted mean for all variables in an xarray dataset.

    Parameters:
    ds (xarray.Dataset): The input dataset.
    Returns:
    xarray.Dataset: A dataset with the weighted means of the original variables.

    Code from Kevin Schwarzwald https://github.com/ks905383
    """
    
    # Calculate area in each pixel
    weights = area_grid(ds.lat,ds.lon)

    # Remove nans, to make weight sum have the right magnitude
    weights = weights.where(~np.isnan(ds))
    
    # Calculate mean
    weighted_means = ((ds*weights).sum(('lat','lon'))/weights.sum(('lat','lon')))
    
    # Return 
    return weighted_means
    
    # ds = ds.copy()
    
    # lat = ds[get_lat_name(ds)]
    
    # # compute the weights based on the latitude
    # weights = np.cos(np.deg2rad(lat))
    
    # # normalize weights to sum to 1 along the latitude dimension
    # weights = weights / weights.sum()
    
    # # expand dimensions of weights to match the dimensions of the dataset variables
    # weights = weights.broadcast_like(ds)

    # # apply weights and calculate  weighted mean
    # #weighted_means = (ds * weights).sum(dim=['lat', 'lon'])
    # weighted_means = (ds * weights).mean(dim=['lat', 'lon'])
    
    return weighted_means

def global_mean(ds):
    '''
    simplified and fast area weighting, relies on built in xarray functionaility 
    Parmeters:
    ds (xarray.Dataset): input dataset

    Returns:
    weighted means (xarray.Dataset): dataset with area weighted means for all variables in input dataset
    - maintains attributes
    
    '''
    weights = np.cos(np.deg2rad(ds.lat))

    weighted_means = ds.weighted(weights).mean(dim = ['lat','lon'], keep_attrs = True)
    return weighted_means

def spi(ds):
    '''
    Calculates Standardized Precipitation Index (SPI)
    returns dataset ds

    for more information on SPI: https://climatedataguide.ucar.edu/climate-data/standardized-precipitation-index-spi 
    '''

    return ds