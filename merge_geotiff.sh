#!/bin/bash
gdal_merge.py -o summer.tiff  -separate 2019-07-01.tiff 2019-07-04.tiff 2019-08-03.tiff 2019-08-05.tiff 2019-08-08.tiff 2019-08-15.tiff 2019-08-30.tiff 2020-07-03.tiff 2020-07-05.tiff 2020-07-13.tiff 2020-08-04.tiff 2020-08-09.tiff 2020-08-12.tiff 2020-08-17.tiff aspect_COP.tif slope_COP.tif wetnessindex_COP.tif sink_COP.tif