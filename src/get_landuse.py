"""Grabbed current landuse WMS tiles and convert it into geopackage."""

import os
import warnings

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
from tqdm import tqdm
# import urllib3
from shapely.ops import transform
from pyproj import Transformer
import requests
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from shapely.geometry import mapping, shape, Polygon
from shapely.geometry.multipolygon import MultiPolygon


warnings.filterwarnings("ignore", category=UserWarning, module="rasterio")


class WMSImageProcessor:
    """class doctring here"""

    def __init__(self, 
                #  grid_extents: list, 
                #  grid_ids: list,  
                 table: dict, 
                 width:int, 
                 height:int, 
                 res: float,
                 epsg: int= 3826,
                 folder_name: str = "test"):
        """Initializes the instance
        Args:
          grid_extents:
            the list of the grid polygons.
          # grid_ids:
          #   the list of the grid ids.
          table:
            the rule of the mapping code.
          folder_name:
            the output folder
          width:
            width of each grid
          height:
            height of each grid
        """
        # self.grid_extents = grid_extents
        # self.grid_ids = grid_ids
        self.width = width
        self.height = height
        self.table = table
        self.res = res
        self.epsg = epsg
        self.folder_name = folder_name
      

    def initialize(self):
        """Create the output folder.

        the folder with the prefix _raw for the original tif data grabbed from nlsc.
        the folder with the prefic _gpkg for the vectorized data, each layer representing each grid.
        """
        folders = [f"../data/processed/{self.folder_name}_raw",
                   f"../data/processed/{self.folder_name}_gpkg"]
        for folder in folders:
            if not os.path.exists(folder):
                os.mkdir(folder)

    def download_image(self,
                       grid_extent: list,
                      #  grid_id: int | str,
                      #  width: int = 250,
                      #  height: int = 250,
                      #  res:float = 1, 
                      #  epsg=3826
                       ) -> None:
        """fetch WMS tiles from nlsc

        Args:
          grid_extent:
            the polygon of the grid
          grid_id:
            the id of the grid
          width:
            the width of the raw tif, default is 250.
          height:
            the height of the raw tif, default is 250.
          epsg:
            the crs of the raw tif, default is epsg:3826.

        Returns:
          None.

        """

        url = (
            "https://wms.nlsc.gov.tw/wms?&SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap"\
            f"&BBOX={grid_extent[3][0]},{grid_extent[3][1]},{grid_extent[1][0]},{grid_extent[1][1]}"\
            f"&SRS=EPSG:{self.epsg}&WIDTH={self.width}&HEIGHT={self.height}&LAYERS=LUIMAP&STYLES=&FORMAT=image/png"\
            "&DPI=96&MAP_RESOLUTION=96&FORMAT_OPTIONS=96&TRANSPARENT=TRUE"
          )

        # http = urllib3.PoolManager()
        # response = http.request("GET", url)
        response = requests.get(url)
        if response.status_code == 200:    # TODO: change to use try and except
          # with open(r"../data/test.tif", "wb") as f:
          #     f.write(response.content)
            
          # Read in memory
          with MemoryFile(response.content) as memfile:
            with memfile.open() as src:
                band_r = src.read(1)
                band_g = src.read(2)
                band_b = src.read(3)
                x_size, y_size = src.res
          
          code, meta = self._preprocessing(grid_extent, 
                                          #  self.grid_ids, 
                                           band_r, band_g, band_b, x_size, y_size)
          return code, meta
          # print("Image saved successfully.")
        else:
          print("Failed to download image.")
          
        

    def _preprocessing(self, 
                       grid_extent: list, 
                      #  grid_id: int | str, 
                       band_r:list, band_g:list, band_b:list, x_size, y_size) -> np.ndarray | dict:
        """convert raw tif band into landuse code

        Args:
          grid_extent:
            the polygon of the grid.
          grid_id:
            the id of the grid.

        Returns:
          code:
            the mapping of the code.
          meta:
            the meta data of the raw with the transform affine.
        """
        # with rasterio.open(f"../data/processed/{self.folder_name}_raw/{grid_id}.tif") as src:
        #     band_r = src.read(1)
        #     band_g = src.read(2)
        #     band_b = src.read(3)
        #     x_size, y_size = src.res
        code = self._match_landuse_code(band_r, band_g, band_b).astype("int32")

        # building new geotiff
        meta = {
            "driver":
                "GTiff",
            "dtype":
                "int32",
            "nodata":
                None,
            "width":
                self.width,
            "height":
                self.height,
            "count":
                1,
            "crs":
                CRS.from_epsg(self.epsg),
            "transform":    # give the tif with geospatial location
                rasterio.transform.from_origin(
                    float(grid_extent[0][0]),
                    float(grid_extent[0][1]), x_size*self.res, y_size*self.res),    # upper left and pixel size
        }

        # print("Image processed successfully.")
        return code, meta

    def _match_landuse_code(self, band_r: int, band_g: int, band_b: int) -> np.ndarray:
        """Convert [R,G,B] of each pixel to landuse code

        Args:
          band_r:
            pixel value of the R band of the raw tif
          band_g:
            pixel value of the G band of the raw tif
          band_b:
            pixel value of the B band of the raw tif

        Returns:
          result:
            the mapping result.
        """

        # Mapping RGB and code
        codes = np.array(list(self.table.values()))
        colors = np.stack((band_r, band_g, band_b), axis=-1)

        distances = np.linalg.norm(colors[:, :, None] - codes[None, None], axis=-1)
        min_distances = np.min(distances, axis=-1)
        indices = np.argmin(distances, axis=-1)

        result = np.zeros_like(indices)

        for n, key in enumerate(self.table.keys()):
            mask = (min_distances == 0) & (indices == n)
            result[mask] = key

        return result

    def convert2shp(self, code: np.ndarray, meta: dict, 
                    grid_id: int | str
                    ) -> None:
        """Vertorize the raw tif

        Args:
          code:
            the mapping code.
          meta:
            the meta data of the raw tif
          grid_id:
            the id of the grid

        Returns:
          None
        """

        # Vectorize the raster data
        shapes = list(rasterio.features.shapes(code, transform=meta["transform"]))
        gpkg_schema = {
            "geometry": "MultiPolygon",
            "properties": {
                "code": "int",
            },
        }
        none_vector = []
        with fiona.open(r"../data/test2.gpkg",
                        "w",
                        "GPKG",
                        gpkg_schema,
                        crs="epsg:3826",
                        layer=f"{grid_id}"
                        ) as gpkg:
            for geom, value in shapes:
                if value != 0:
                    polygons = [shape(geom)]
                    multipolygon = MultiPolygon(polygons)
                    gpkg.write({
                        "geometry": mapping(multipolygon),
                        "properties": {
                            "code": int(value),
                        },
                    })
                else:
                    none_vector.append(shape(geom))
            multipolygon = MultiPolygon(none_vector)
            gpkg.write({
                "geometry": mapping(multipolygon),
                "properties": {
                    "code": 0,
                },
            })
        # print("Image converted successfully.")

    def wms_to_shp(self, grid_ids, grid_extents):
        """The process of coverting wms to shp"""
        # self.initialize()
        grid_extents = grid_extents.apply(lambda geom:list(geom.exterior.coords)).to_list()
      
        for (grid_extent, grid_id) in tqdm(zip(grid_extents, grid_ids), total = len(grid_ids)):
            # STEP1: Download WMS from nlsc
            code, meta = self.download_image(grid_extent)

            # STEP2: Convert RGB into the land code
            # code, meta = self.preprocessing(grid_extent_split, grid_id)

            # STEP3: Vectorize the raster data and save as a geopackage
            self.convert2shp(code, meta, grid_id)


def create_grid(extent, side_length):
    
    # Example
    extent = Polygon([
            [
              121.48070616727443,
              25.09496550087492
            ],
            [
              121.48070616727443,
              24.983447165281433
            ],
            [
              121.5945476252482,
              24.983447165281433
            ],
            [
              121.5945476252482,
              25.09496550087492
            ],
            [
              121.48070616727443,
              25.09496550087492
            ]
          ]
        )
    
    transformer_4326_3826 = Transformer.from_crs(4326, 3826, always_xy=True)
    transformer_3826_4326 = Transformer.from_crs(3826, 4326, always_xy=True)

    extent_3826 = transform(transformer_4326_3826.transform, extent)
    xmin, ymin, xmax, ymax = extent_3826.bounds


    cols = list(np.arange(xmin, xmax + side_length, side_length))
    rows = list(np.arange(ymin, ymax + side_length, side_length))

    polygons = []
    for x in cols[:-1]:
        for y in rows[:-1]:
            polygons.append(Polygon([(x,y), (x+side_length, y), (x+side_length, y+side_length), (x, y+side_length)]))

    grid = gpd.GeoDataFrame({'geometry':polygons}, crs=3826)

    grid_str = list(map(lambda x: x.wkt.replace("POLYGON ((", "").replace("))", ""), list(grid.geometry)))
    grid_polygons = list(map(lambda x: x.split(", "), grid_str))

    # test_grid_ids = grid["gridid"][:3]
    test_grid_extents = grid_polygons[:1]

    return grid_polygons


if __name__ == "__main__":
    table_df = pd.read_csv(r"mapping_code/109年國土利用現況調查成果.csv")

    # # Setting mapping codes
    # mapping_table = {
    #     value[0]: [value[1], value[2], value[3]]
    #     for value in zip(table_df["類別代碼"], table_df["R"], table_df["G"], table_df["B"])
    # }

    # grid = gpd.read_file(r"grids.geojson").iloc[:3]

    # wms = WMSImageProcessor(
    #     # grid_extents=test_grid_extents,
    #     width=2000,
    #     height=2000,
    #     res = 1,
    #     # grid_ids=test_grid_ids,
    #     table=mapping_table
    # )
    # wms.wms_to_shp(grid['id'].to_list(), grid['geometry'])
