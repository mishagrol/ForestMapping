#!/bin/bash
gdal_merge.py -o terrain.tiff -separate aspect_COP.tif slope_COP.tif wetnessindex_COP.tif sink_COP.tif