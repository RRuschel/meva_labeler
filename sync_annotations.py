import argparse
from pathlib import Path
from collections import defaultdict
import ast
from natsort import natsorted
from MEVA import build
import random
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed


def read_correspondence_dict(file_location):
    correspondence_dict = defaultdict(list)
    try:
        with open(file_location, 'r') as f:
            for line in f:
                try:
                    key, value = line.split(':')
                    parsed_val = ast.literal_eval(value.strip())
                    if len(parsed_val) == 0:
                        continue
                    if parsed_val[0] == 'skipped':
                        continue
                
                    correspondence_dict[key] = parsed_val
                except ValueError as ve:
                    print(f'Error on read_correspondence_dict: {ve}')
                    continue
        return correspondence_dict
    except FileNotFoundError as e:
        print(f'Error on read_correspondence_dict: {e}')


def read_video_id_mapping(file_location):
    id_map = defaultdict(dict)
    try:
        with open(file_location, 'r') as f:
            for line in f:
                try:
                    first, second, vals = line.split(':')
                    parsed_vals = ast.literal_eval(vals.strip())
                    id_map[first][second] = parsed_vals
                    #id_map[second][first] = [val[::-1] for val in parsed_vals]
                except ValueError as ve:
                    print(f'Error on video_id_mapping: {ve}')
                    continue
    except FileNotFoundError as e:
        print(f'{video_id_mapping} not found, creating a new file')

    return id_map


class UniqueObjects():
    def __init__(self, video_id_mapping):
        self.objects = []
        self.populate(video_id_mapping)

    def populate(self, video_id_mapping):
        paired_cameras = natsorted(video_id_mapping.keys())

        for camera in paired_cameras:
            id_map = video_id_mapping[camera]
            for new_camera, id_pairs in id_map.items():
                for id_pair in id_pairs:
                    self.try_add(camera, id_pair[0], new_camera, id_pair[1])
    
    def __len__(self):
        return len(self.objects)
    
    def add(self, camera, time, obj_id):
        self.objects.append({
            'camera': [camera],
            'time': [time],
            'obj_id': [obj_id]
        })

    def try_add(self, shows_up, obj_id, update_location, update_id):
        _, start_time, _, _, camera = shows_up.split('.')
        _, start_time_new, _, _, camera_new = update_location.split('.')
        start_time = start_time[:-3]
        start_time_new = start_time_new[:-3]

        if obj_id < 0:
            self.add(camera_new, start_time_new, update_id)
            return

        index, _ = self.get(camera, start_time, obj_id)

        if index == -1:
            self.add(camera, start_time, obj_id)
            index = len(self.objects) - 1

        if update_id < 0:
            return

        self.objects[index]['camera'].append(camera_new)
        self.objects[index]['obj_id'].append(update_id)
        self.objects[index]['time'].append(start_time_new)

    def get(self, camera, start_time, obj_id):
        for idx, obj in enumerate(self.objects):
            for _camera, _start, _id in zip(obj['camera'], obj['time'], obj['obj_id']):
                if _camera == camera and _start == start_time and _id == obj_id:
                    return idx, obj
        return -1, None
    
    def insert_class(self, camera, start_time, obj_id, obj_class):
        idx, obj = self.get(camera, start_time, obj_id)
        if idx == -1:
            return
        if hasattr(obj, 'class'):
            assert obj['class'] != obj_class, f'Error in inserting class, class already exists - {obj_class} != {obj["class"]}' 
        obj['class'] = obj_class

    def __iter__(self):
        for idx, obj in enumerate(self.objects):
            yield idx, obj


def update_annotation(cat_id_pairs, c_dict, video_name):
    updated_cat_id_pairs = {}
    
    if video_name not in c_dict:
        print(f'No correspondence dict for {video_name}')
        return cat_id_pairs
    
    c_list = c_dict[video_name]

    for new_idx, _c_list in enumerate(c_list):
        for entry in _c_list:
            updated_cat_id_pairs[entry] = new_idx

    if len(updated_cat_id_pairs) != len(cat_id_pairs):
        print(f'Error in updating cat_id_pairs for {video_name}')
        original_keys = set(cat_id_pairs.keys())
        updated_keys = set(updated_cat_id_pairs.keys())
        print(f'Missing: {original_keys - updated_keys} - Consider checking the annotations manually')
        
    return updated_cat_id_pairs


def generate_random_colors(n_colors):
    colors = []
    for i in range(n_colors):
        colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    return colors


def dump_files_threaded(files, dataset, correspondence_dict, unique_map, output_path='./'):
    random_colors = generate_random_colors(len(unique_map))
    with ThreadPoolExecutor(max_workers=len(files)) as executor: # Adjust max_workers based on your system
        futures = [executor.submit(dump_frame_with_unique_id, unique_map, file, dataset, correspondence_dict, random_colors, output_path) for file in files]

        # Wait for all futures to complete
        for future in as_completed(futures):
            future.result() # You can handle exceptions here if needed


def dump_frame_with_unique_id(unique_map, file, dataset, correspondence_dict, random_colors, output_path='./'):
    tmp_output_path = Path(output_path) / file
    tmp_output_path.mkdir(parents=True, exist_ok=True)
    video, boxes, actions, cat_id_pairs = dataset[file]
    date, start, end, _, camera, _, _ = video.name.split('.')
    start = start[:-3]
    updated_cat_id_pairs = update_annotation(cat_id_pairs, correspondence_dict, file)
    new_cat_id_pairs = {}
    for k, v in cat_id_pairs.items():
        try:
            new_value = updated_cat_id_pairs[k]
            new_cat_id_pairs[new_value] = list(v.keys())[0]
        except Exception as e:
            print(f'Exeption {e} in {k} {v} @ dump_frame_with_unique_id')
            continue

    video_reader = cv2.VideoCapture(str(video))
    video_length = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_counter = 0
    while frame_counter < video_length:
        frame, ret = video_reader.read()

        for action in actions:
            for actor_id, actor_info in action.actors_time_dict.items():
                if min(actor_info['timespan']) <= frame_counter <= max(actor_info['timespan']):
                    box_idx = frame_counter - min(actor_info['timespan'])
                    unique_idx, _ = unique_map.get(camera, start, updated_cat_id_pairs[actor_id])
                    assert unique_idx != -1, f'Error in getting unique object for {action.name} {actor_id}'
                    color = random_colors[unique_idx]
                    box = actor_info['boxes'][box_idx]
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'ID: {unique_idx}', (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imwrite(str(tmp_output_path / f'frame_{frame_counter:06d}.jpg'), frame)
    video_reader.release()


def update_annotations(files, dataset, correspondence_dict):
    data_dict = {}
    for file in files:
        new_cat_id_pairs = {}
        video, boxes, actions, cat_id_pairs = dataset[file]
        #dataset_return_buffer[file] = (video, boxes, actions, cat_id_pairs)
        updated_cat_id_pairs = update_annotation(cat_id_pairs, correspondence_dict, file)
        for k, v in cat_id_pairs.items():
            try:
                new_value = updated_cat_id_pairs[k]
                new_cat_id_pairs[new_value] = list(v.keys())[0]
            except Exception as e:
                print(f'Exeption {e} in {k} {v} @ update_annotations')
                continue
        
        data_dict[file] = {
            'video': video,
            'boxes': boxes,
            'actions': actions,
            'cat_id_pair': cat_id_pairs,
            'new_cat_id_pairs': new_cat_id_pairs,
            'updated_cat_id_pair': updated_cat_id_pairs
        }
    return data_dict


def generate_GPT_files(files, dataset, data_dict, unique_map):
    for file in files:
        _, _, actions, _ = dataset[file]
        with open(f'GPT_{file}.txt', 'w+') as f:
            date, start_time ,end_time, location, camera_id = file.split('.')
            start_time = start_time[:-3]
            f.write(f'Date: {date} - Start time: {start_time} - End time: {end_time} - Location: {location} - Camera ID: {camera_id}\n')

            for action in actions:
                f.write(f'{action.name}\n')
                time_dict = action.actors_time_dict
                
                for actor, time in time_dict.items():
                    try:
                        updated_actor = data_dict[file]['updated_cat_id_pair'][actor]
                        unique_idx, unique_obj = unique_map.get(camera_id, start_time, updated_actor)
                        assert unique_idx != -1, f'Error in getting unique object for {action.name} {actor}'
                        timespan = time['timespan']
                        _start, _end = min(timespan), max(timespan)
                        f.write(f'ID: {unique_idx} - Start frame: {_start} - End frame: {_end}\n')
                    except Exception as e:
                        print(f'Error {e} in {action.name} {actor}')
    
    all_ids = []
    for idx, file in enumerate(files):
        date, start_time, end_time, location, camera_id = file.split('.')
        if idx == 0:
            first_time = start_time
        all_ids.append(camera_id)

    final_name = '.'.join([first_time] + all_ids + [end_time])
    with open(f'GPT_{final_name}.txt', 'w+') as f:
        for idx, obj in unique_map:
            f.write(f"ID: {idx} - {obj['class']}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hoi_path', help='Path to HOI file', default='/home/raphael/Documents/skywalker_6/raphael/meva')
    parser.add_argument('--dataset_file', type=str, default='MEVA')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--filter_by', nargs='+', help='Filter by', default=['date'])
    parser.add_argument('--filter_val', nargs='+', help='Filter value', default=['2018-03-05'])
    parser.add_argument('--camera_pairs', nargs='+', help='Camera pairs to perform matching', default=['G331', 'G331'])
    args = parser.parse_args()

    correspondence_dict_path = Path(args.hoi_path) / 'correspondence_dict.txt'
    correspondence_dict = read_correspondence_dict(correspondence_dict_path)

    for value in args.filter_val:
        correspondence_dict = {k: v for k, v in correspondence_dict.items() if value in k}


    video_id_mapping = Path(args.hoi_path) / 'video_id_mapping.txt'
    video_id_mapping = read_video_id_mapping(video_id_mapping)

    unique_map = UniqueObjects(video_id_mapping)

    dataset = build(image_set='train', args=args)


    files = [
        '2018-03-05.13-10-01.13-15-01.bus.G331',
        '2018-03-05.13-15-01.13-20-01.bus.G331',
        '2018-03-05.13-15-00.13-20-00.bus.G506'
    ]

    dump_files_threaded(files, dataset, correspondence_dict, unique_map, output_path='./')

    data_dict = update_annotations(files, dataset, correspondence_dict)

    for video, data in data_dict.items():
        date, start, end, _, camera = video.split('.')
        start = start[:-3]
        for _id, obj_category in data['new_cat_id_pairs'].items():

            unique_map.insert_class(camera, start, _id, obj_category)

    generate_GPT_files(files, dataset, data_dict, unique_map)

    

