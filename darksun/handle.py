"""
IROS output data handling.
"""

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import pickle

from bloodmoon.io import _exists_valid
from darksun.data import DataLoader

__all__ = [
    "save_database", "save_sky", "save_pickle",
    "load_database", "load_sky", "load_pickle",
]


def _make_column(
    name: str,
    data: np.array,
    frmt: str,
    unit: str = "",
) -> fits.Column:
    """
    Creates a FITS table column with the specified parameters.

    Args:
        name (str):
            Name of the column.
        data (np.array):
            Data to be stored in the column.
        frmt (str):
            FITS format of the column data.
        unit (str, optional (default="")):
            Physical unit of the column data.

    Returns:
        column (fits.Column): FITS column instance.
    """
    column = fits.Column(
        name=f"{name.upper()}",
        array=data,
        format=frmt,
        unit=unit,
    )
    return column


def _make_bintable(
    name: str,
    columns: list[fits.Column],
    header: fits.Header = None,
) -> fits.BinTableHDU:
    """
    Creates a FITS binary table HDU.

    Args:
        name (str):
            Name of the binary table.
        columns (list[fits.Column]):
            List of FITS Column objects.
        header (fits.Header, optional (default=None)):
            FITS Header object for the table.

    Returns:
        table (fits.BinTableHDU): FITS binary table HDU.
    """
    table = fits.BinTableHDU.from_columns(
        columns=columns,
        header=header,
        name=f"{name.upper()}",
    )
    return table

"""
                           @
                          @@@
                         @@@@@
                        @@@@@@@
                       @@@@@@@@@
                      @@@@@@@@@@@
                     @@@@@@@@@@@@@
                    @@@@@@@@@@@@@@@
                   @@@@@@@@@@@@@@@@@
                  @                 @
                 @@@               @@@
                @@@@@             @@@@@
               @@@@@@@           @@@@@@@
              @@@@@@@@@         @@@@@@@@@  
             @@@@@@@@@@@       @@@@@@@@@@@
            @@@@@@@@@@@@@     @@@@@@@@@@@@@
           @@@@@@@@@@@@@@@   @@@@@@@@@@@@@@@
          @@@@@@@@@@@@@@@@@ @@@@@@@@@@@@@@@@@
"""

def save_database(
    database: dict,
    sdlA: DataLoader,
    sdlB: DataLoader,
    save_to: str | Path,
) -> None:
    """
    Saves the input database with the structure described
    in `Log` (look at the `data` module) to a FITS file.

    Args:
        database (dict):
            Database with the analysis-stored data.
        sdlA (DataLoader):
            SDL instance for WFM camera A.
        sdlB (DataLoader):
            SDL instance for WFM camera B.
        save_to (str | Path):
            Directory path to save the FITS file.
    """
    print("# Saving data...")
    # HDU list and Primary Header
    hdu_list = fits.HDUList([])
    primary_hdu = fits.PrimaryHDU()
    hdu_list.append(primary_hdu)

    # BinTables
    for camID, sdl in zip(database.keys(), (sdlA, sdlB)):
        datacam = database[camID]
        columns = [
            _make_column(
                key, datacam[key]["data"], datacam[key]["format"], datacam[key]["unit"],
            )
            for key in list(datacam.keys())
        ]
        table_hdu = _make_bintable(camID, columns, sdl.header)
        hdu_list.append(table_hdu)

    # save data
    hdu_list.writeto(save_to, output_verify="fix+ignore")
    hdu_list.close()
    print("# Saving completed!")


def save_sky(
    sky: np.array,
    snr: np.array,
    sdl: DataLoader,
    save_to: str | Path,
    wcs: WCS = None,
) -> None:
    """
    Saves the given sky array and its significance (SNR) as FITS Image files,
    including optional World Coordinate System (WCS) information if provided.

    Args:
        sky (np.array):
            Sky data array to be saved.
        snr (np.array):
            Sky significance array.
        sdl (DataLoader):
            DataLoader instance providing additional metadata.
        save_to (str | Path):
            File path or directory where the FITS image will be saved.
        wcs (WCS, optional (default=None)):
            World Coordinate System instance, which can be used to
            include coordinate information in the FITS header.
    """
    print("# Saving sky...")
    # HDU list and Primary Header
    hdu_list = fits.HDUList([])
    primary_hdu = fits.PrimaryHDU()
    hdu_list.append(primary_hdu)

    # Images for data
    for img, name in zip(
        [np.int32(sky), np.float32(snr)],
        ["sky", "snr"],
    ):
        image_hdu = fits.ImageHDU(
            data=img,
            header=sdl.header,
            name=name.upper(),
        )
        if wcs: image_hdu.header.update(wcs.to_header())
        hdu_list.append(image_hdu)
    
    hdu_list.writeto(save_to, output_verify="fix+ignore")
    hdu_list.close()
    print("# Saving completed!")


def save_pickle(data: object, save_to: str | Path) -> None:
    """
    Saves data in pickle format.

    Args:
        data (object):
            Data to save.
        save_to (str | Path):
            Path to the directory for saving the pickle file.
    """
    print("# Saving data...")
    with open(save_to, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("# Saving completed!")

"""
                                                           @       @  
                                                          @@@@    @@@ 
                                                        @@@@@@@@@@@@@ 
                                                       @@@@@@@@@@@@@@@
                                                       @@@@@@@@@@@@@@@
                                                       @@@@@@@@@@@@@@@
                                            @@@@@@@@@@@@@@@@@@@@@@@@@ 
                                          @@@@@@@@@@@@@@@@@@@@@@@@@@  
                                        @@@@@@@@@@@@@@@@@@@@@@@@@@@@  
                                       @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 
                                      @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 
                                      @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 
                                     @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 
                                   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  
       @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@   
       @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@      
"""

def load_database(filepath: str | Path) -> dict:
    """
    Loads the specified database to a dict having the
    structure described in `Log` (within `data` module).

    Args:
        filepath (str | Path): Path to the FITS file.

    Returns:
        database (dict): Container with collected data.
    """
    def load_data(filepath: Path) -> dict:
        """Opens FITS file and stores data in a dict."""
        with fits.open(filepath) as hdul:
            hdus = (dict(hdul[1].header), dict(hdul[2].header))
            hdus_data = (hdul[1].data, hdul[2].data)
        data = {
            hdu["EXTNAME"].lower(): {
                hdu[f"TTYPE{idx}"].lower(): {
                    "data": hdu_data.field(idx - 1),
                    "format": hdu[f"TFORM{idx}"],
                    "unit": hdu[f"TUNIT{idx}"] if f"TUNIT{idx}" in hdu.keys() else "",
                }
                for idx in range(1, len(hdus_data[0][0]) + 1)
            }
            for hdu, hdu_data in zip(hdus, hdus_data)
        }
        return data
    
    if not isinstance(filepath, Path):
        filepath = Path(filepath)
    if _exists_valid(filepath):
        print("# Loading data...")
        data = load_data(filepath)
        print("# Loading completed!")
        return data


def load_sky(filepath: str | Path) -> tuple[np.array]:
    """
    Loads sky and its SNR from FITS.

    Args:
        filepath (str | Path): Path to the FITS file.

    Returns:
        output (tuple):
            - sky (np.array): 2D array for the sky.
            - snr (np.array): sky significance.
    """
    def load_data(filepath: Path) -> tuple[np.array]:
        """Open FITS and store Images in 2D-array."""
        with fits.open(filepath) as hdu:
            sky, snr = hdu[1].data, hdu[2].data
        return sky, snr
    
    if not isinstance(filepath, Path):
        filepath = Path(filepath)
    if _exists_valid(filepath):
        print("# Loading data...")
        sky, snr = load_data(filepath)
        print("# Loading completed!")
        return sky, snr


def load_pickle(filepath: str | Path) -> object:
    """
    Loads data from pickle file.

    Args:
        filepath (str | Path):
            Path to the pickle file.
    
    Returns:
        output (object): Loaded object.
    """
    if _exists_valid(filepath):
        print("# Loading data...")
        with open(filepath, "rb") as handle:
            data = pickle.load(handle)
        print("# Loading completed!")
        return data


# end