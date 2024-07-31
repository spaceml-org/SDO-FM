import glob
from functools import partial

import astropy.units as u
import webdataset as wds
from aiapy.calibrate.util import get_pointing_table
from multiprocess import Pool, Manager
from sdomlpy.calibration import calibrate
from sunpy.map import Map
from tqdm import tqdm
import time 
import logging
from datetime import datetime
from pathlib import Path

WAVELENGTHS = ['0094', '0131', '0171', '0193', '0211', '0304', '0335', '1600', '1700']

logging.basicConfig(filename="calibration.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
logger = logging.getLogger('calibrate')

def run(one_datetime_paths):
    arrs, metas = {}, {}
    for fits_path in one_datetime_paths:
        try:
            # e.g. 'AIA20231001_000300' from the path, collect only the wavelength
            wav = fits_path.split("/")[-1].split(".")[0].split("_")[-1]

            scaled_image_array, metadata = calibrate(
                fits_path,
                scale=512,
                target_sun_area=976.0,
                source="SDOML",
                file_stamp="name",
                # correction_table_path="/home/walsh/sdomlpy/sdomlpy/misc/calibration_table.dat",
                # pointing_table=pointing_table,
            )
            if scaled_image_array is None or metadata is None:
                continue
            arrs[wav + '.npz'] = scaled_image_array
            metas[wav + '.json'] = metadata
        except Exception as e:
            logger.error("Failed to process", fits_path, "\n", e)
        
    return arrs, metas

def _wrapper(enum_iterable, function, **kwargs):
    print(enum_iterable)
    day = kwargs['day_lookup']
    output_dir = kwargs['output_dir']

    with wds.TarWriter(
        Path(output_dir) / f"sdomlv2b-live.aia.512pix.3min-{day}.tar"
    ) as sink:
        # iterate per CPU subset
        for mem_bound_subset in enum_iterable[1]:
            t0 = time.time()
            
            # process the a single datetime
            arrs, meta = function(mem_bound_subset)
            
            logger.info(f"Datetime subset {s} took {time.time()-t0:.2f} to process")
            time_to_work = t0 - time.time()
            t1 = time.time()
            sink.write(
                {
                    "__key__": f"{meta[WAVELENGTHS[0]]['T_OBS'][:-4]}",
                    **arrs,
                    **meta
                }
            )
            logger.info(f"Datetime subset {s} took {time_to_work:.2f} to process and {time.time()-t1:.2f} to save")
    return enum_iterable[0], True


if __name__ == "__main__":
    print("Starting conversion")

    files = glob.glob("/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/*")
    files = sorted(files)

    logger.info(f"There are {len(files)} files to process")

    # map = Map(files[0])
    # logger.info(f"Collecting pointing table for {map.date - 12 * u.h} to {datetime.today()}")
    # pointing_table = get_pointing_table(
    #     map.date - 12 * u.h, datetime.today()
    # )

    days = {}
    # files = ["/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_000300_0094.fits", "/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_000300_0131.fits"]
    files = [
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_000300_0094.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_000300_0131.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_000300_0171.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_000300_0193.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_000300_0211.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_000300_0304.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_000300_0335.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_000300_1600.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_000300_1700.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_000600_0094.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_000600_0131.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_000600_0171.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_000600_0193.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_000600_0211.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_000600_0304.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_000600_0335.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_000600_1600.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_000600_1700.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_000900_0094.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_000900_0131.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_000900_0171.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_000900_0193.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_000900_0211.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_000900_0304.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_000900_0335.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_000900_1600.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_000900_1700.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_001200_0094.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_001200_0171.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_001200_0304.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_001200_1600.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_001200_1700.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_001800_0131.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_001800_0193.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_001800_0211.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_001800_0335.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_002100_0094.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_002100_0131.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_002100_0171.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_002100_0193.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_002100_0211.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_002100_0304.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_002100_0335.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_002100_1600.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_002100_1700.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_002400_0171.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_002400_0304.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_002700_0094.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_002700_0131.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_002700_0171.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_002700_0193.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_002700_0211.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_002700_0304.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_002700_0335.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_002700_1600.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_002700_1700.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_003000_0094.fits',
        '/mnt/us-fdlx-ard-calibrated-synoptic/nrt/fits/2023/10/01/H0000/AIA20231001_003000_1700.fits',
    ]
    for f in files:
        s = f.split("/")
        day = f"{s[-5]}{s[-4]}{s[-3]}"
        if day not in days:
            days[day] = []
        days[day].append(f)

    # MAX_MEM_CHUNK_SIZE = 36

    # Chunk the days into memory chunks for processing
    # manager = Manager()
    days_chunked = {} # manager.dict()
    for day in days:
        if day not in days_chunked:
            days_chunked[day] = []
        chunked = []
        chunk = []
        count = 0
        all_datetimes = {}
        for f in days[day]:
            dt = '_'.join(f.split("/")[-1].split(".")[0].split("_")[0:2])
            if dt not in all_datetimes:
                all_datetimes[dt] = []
            all_datetimes[dt].append(f)

            # chunk.append(f)
            # count += 1
            # if count % MAX_MEM_CHUNK_SIZE == 0:
            #     chunked.append(chunk)
            #     chunk = []
        # if len(chunk) > 0:
        #     chunked.append(chunk)
        # days_chunked[day].append(list(all_datetimes.values()))
        for k, v in all_datetimes.items():
            print(len(v))
            days_chunked[day].append(v)
        # days_chunked[day].append(chunk)

    nprocs = 240       
    
    logger.info(f"Number of days: {len(days_chunked)}, with {len(chunked)} memory chunks each")

    try:
        with Pool(nprocs) as pool:
            results = []
            results = [None] * len(days_chunked)

            func = partial(_wrapper, function=run, day_lookup = list(days_chunked.keys()))

            with tqdm(total=len(days_chunked)) as pbar:
                for i, r in pool.imap_unordered(func, enumerate(days_chunked.values())):
                    assert r == True
                    count += 1
                    pbar.update()

    except Exception as e:
        print(e)
        raise e
    finally:  # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()
