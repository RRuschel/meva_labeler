To run the annotation tool, just update the --hoi_path parameter to where you mounted the meva dataset
The filter_by and filter_val params can be used to select specific cameras, locations, etc
For example: filter_by=location filter_val=bus will give all the videos on the bus station


### Usage

#### Requirements Install
```
pip install -r requirements.txt
```
#### Python3 Tkinter Install
```
sudo apt-get install python3-tk
```

#### Run Command
```
python UI.py --hoi_path <root_path_to_meva_data> --filter_by <e.g. camera OR location, etc> --filter_val <G109 OR bus>
```
Sample command:
Camera
```
python UI.py --hoi_path ./data/meva --filter_by camera --filter_val G109
```
Location
```
python UI.py --hoi_path ./data/meva --filter_by location --filter_val bus
```
