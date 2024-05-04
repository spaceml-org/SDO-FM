import numpy as np
import os
import blosc
from loguru import logger

num_warnings = 0
max_warnings = 100

def load_blosc_file(blosc_cache, event_date, setname, elements):
    datestr = event_date.strftime("%Y-%m-%d_%H:%M:%S")
    elemsstr = "-".join(elements)
    
    file_name = f"{blosc_cache}/{datestr}__{setname}__{elemsstr}.blosc"
    
    if os.path.isfile(file_name):
        try:
            with open(file_name, "rb") as f:
                arr = blosc.unpack_array(f.read())
            return arr
        except Exception as e:
            if num_warnings<max_warnings:
                logger.warning(f"exception reading {file_name}, {e}")
                num_warnings += 1

    return None


def save_blosc_file(blosc_cache, event_date, setname, elements, array, dtype=np.float32):
    datestr = event_date.strftime("%Y-%m-%d_%H:%M:%S")
    elemsstr = "-".join(elements)
    
    file_name = f"{blosc_cache}/{datestr}__{setname}__{elemsstr}.blosc"
    with open(file_name, "wb") as f:
        f.write(blosc.pack_array(array.astype(dtype)))

def load_aia(blosc_cache, aia_data, align_data_row_element, wavelengths):

    # attempts to read from blosc cache first
    if blosc_cache is not None:
        r = load_blosc_file(blosc_cache, align_data_row_element.name, 'aia', wavelengths)
        if r is not None:
            return r

    r = []
    for wavelength in wavelengths:
        idx_wavelength = align_data_row_element[f"idx_{wavelength}"]
        year = str(align_data_row_element.name.year)

        img = aia_data[year][wavelength][idx_wavelength, :, :]
        r.append(img)
    r = np.r_[r]

    # saves to blosc cache
    if blosc_cache is not None:
        save_blosc_file(blosc_cache, align_data_row_element.name, 'aia', wavelengths, r)

    return r

def load_hmi(blosc_cache, hmi_data, align_data_row_element, components):
    # attempts to read from blosc cache first
    if blosc_cache is not None:
        r = load_blosc_file(blosc_cache, align_data_row_element.name, 'hmi', components)
        if r is not None:
            return r

    r = []
    for component in components:
        idx_components = align_data_row_element[f"idx_{component}"]
        year = str(align_data_row_element.name.year)

        img = hmi_data[year][component][idx_components, :, :]
        r.append(img)
    r = np.r_[r]

    # saves to blosc cache
    if blosc_cache is not None:
        save_blosc_file(blosc_cache, align_data_row_element.name, 'hmi', components, r)

    return r