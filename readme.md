To run the annotation tool, just update the --hoi_path parameter to where you mounted the meva dataset
The filter_by and filter_val params can be used to select specific cameras, locations, etc
For example: filter_by=location filter_val=bus will give all the videos on the bus station

So basically, just run UI.py --hoi_path /path/to/meva --filter_by ... --filter_val ...