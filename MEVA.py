from torch.utils.data import Dataset
from pathlib import Path
import cv2
from natsort import natsorted
from collections import defaultdict
import random
from typing import Union
import torch
from tqdm import tqdm
import yaml
from box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy, union_box
from functools import total_ordering
from functools import lru_cache

@total_ordering
class activity:
    def __init__(self, name, confidence, actors=None, boxes=None):
        self.name = name
        self.confidence = confidence
        self.actors = [] if actors is None else actors
        self.boxes = defaultdict(list) if boxes is None else boxes

    def add_actor(self, actor):
        self.actors.extend(actor)

    def __repr__(self):
        result = f"Actions: {self.name}\n"
        result += f"Confidence: {self.confidence}\n"
        result += f"Actors: {self.actors}\n"
        result += '---\n'
        return result

    def organize(self, ret=False):
        actors_time_dict = {}
        for actor in self.actors:
            actors_time_dict[actor.id] = {}
            actors_time_dict[actor.id]['timespan'] = set(actor.timespan.range_iter())
            actors_time_dict[actor.id]['ptr'] = actor
        self.actors_time_dict = actors_time_dict
        if ret:
            return self

    def check_add_box(self, box):
        if box.id1 in self.actors_time_dict.keys():
            if box.ts in self.actors_time_dict[box.id1]['timespan']:
                self.boxes[box.ts].append(box)
                return True
        return False
    
    def __min__(self):
        return min(self.boxes.keys())
    
    def min(self):
        return min(self.boxes.keys())
    
    def __max__(self):
        return max(self.boxes.keys())
    
    def __eq__(self, other):
        return self.__min__() == other.__min__()
    
    def __lt__(self, other):
        return self.__min__() < other.__min__()
    


class actor:
    def __init__(self, id, timespan, category=None):
        self.id = id
        self.timespan = timespan
        self.category = category

    def __repr__(self):
        result = f"Actor: {self.id}\n"
        result += f"Stamp Type: {self.timespan.type}\n"
        result += f"Start: {self.timespan.start}\n"
        result += f"End: {self.timespan.end}\n"
        return result

    # def update_category(self, category):
        
class timespan:
    def __init__(self, _type, start, end):
        self.type = _type
        self.start = start
        self.end = end

    def range_iter(self):
        return range(self.start, self.end)

class BBox:
    def __init__(self, x0, y0, x1, y1, id0, id1, ts):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.id0 = id0
        self.id1 = id1
        self.ts = ts

    def __repr__(self):
        result = f"Box: {self.x0} {self.y0} {self.x1} {self.y1}\n"
        result += f"ID: {self.id0} {self.id1}\n"
        result += f"Timestamp: {self.ts}\n"
        return result

def parse_activity_yaml(file_path):
    with open(file_path, 'r') as file:
        # Load the YAML content
        activities = yaml.safe_load(file)

    acts_list = []
    # Print the parsed data
    for aa in activities:
        for k, v in aa.items():
            if k == 'meta' and 'empty' in v:
                ##print('Found "empty" in v:', v)
                return acts_list
            for k2, v2 in v.items():
                if k2 == 'act2':
                    acts = [activity(k3, v3) for k3, v3 in v2.items()]
                    assert len(acts) == 1, 'More than one activity'
                    acts = acts[0]
                elif k2 == 'actors':
                    actors = []
                    for entry in v2:
                        _id = entry['id1']
                        _time = entry['timespan']
                        assert len(_time) == 1, 'More than one timespan'	
                        for _type, _span in _time[0].items():
                            _start = _span[0]
                            _end = _span[1]
                            actors.append(actor(_id, timespan(_type, _start, _end)))

        acts.add_actor(actors)
        acts_list.append(acts)
    return acts_list

def parse_types_yaml(file_path):
    with open(file_path, 'r') as file:
    # Load the YAML content
        types = yaml.safe_load(file)

    #cat_id_pairs = []
    cat_id_pairs = {}
    for entry in types:
        for k, v in entry.items():
            if k == 'meta' and 'empty' in v:
                #print('Found "empty" in v:', v)
                return cat_id_pairs
            assert k == 'types', 'Not types'
            assert len(v) == 2, 'More than 2 entries'
            for _type, values in v.items():
                if _type == 'cset3':
                    category = values
                elif _type == 'id1':
                    id = values
        #cat_id_pairs.append((category, id))
        cat_id_pairs[id] = category

    return cat_id_pairs

def parse_geom_yaml(file_path):
    with open(file_path, 'r') as file:
        # Load the YAML content
        geom = yaml.safe_load(file)
    boxes = []
    for entry in geom:
        for k, v in entry.items():
            if k == 'meta' and 'empty' in v:
                #print('Found "empty" in v:', v)
                return boxes
            #assert k == 'geom', 'Not geom'
            for key, value in v.items():
                if key == 'g0':
                    x0, y0, x1, y1 = map(int, value.split())
                    #assert x0 < x1, 'x0 >= x1'
                    #assert y0 < y1, 'y0 >= y1'
                elif key == 'id0':
                    id0 = value
                elif key == 'id1':
                    id1 = value
                elif key == 'ts0':
                    ts = value
            #try:
            bbox = BBox(x0, y0, x1, y1, id0, id1, ts)
            boxes.append(bbox)
            # except:
            #     print('Insufficient data')
    return boxes

class MEVA(Dataset):
    def __init__(self, data_path: Path, annotation_path: Path, args, transforms=None, output_format=None):
        super().__init__()
        self.data_path = data_path
        self.annotation_path = annotation_path
        self.debug = args.debug
        self.annotations = self.find_annotation(data_path, annotation_path)
        self.file_names = list(self.annotations.keys())
        self.transforms = transforms
        
        self.output_format = output_format

        for filter_cat, filter_val in zip(args.filter_by, args.filter_val):
            filter_dict = self.filter_files()
            self.file_names = filter_dict[filter_cat][filter_val]
        self.annotations = {k: v for k, v in self.annotations.items() if k in self.file_names}
        # if args.filter_by:
        #     for filter_criteria in args.filter_val:
        #         self.filter_dict = self.filter_files()
        #         self.file_names = self.filter_dict[args.filter_by][args.filter_val]
        #     self.annotations = {k: v for k, v in self.annotations.items() if k in self.file_names}
        

        self.MEVA_activities = [
            'hand_interacts_with_person',
            'person_abandons_package',
            'person_carries_heavy_object',
            'person_closes_facility_door',
            'person_closes_trunk',
            'person_closes_vehicle_door',
            'person_embraces_person',
            'person_enters_scene_through_structure',
            'person_enters_vehicle',
            'person_exits_scene_through_structure',
            'person_exits_vehicle',
            'person_interacts_with_laptop',
            'person_loads_vehicle',
            'person_opens_facility_door',
            'person_opens_trunk',
            'person_opens_vehicle_door',
            'person_picks_up_object',
            'person_purchases',
            'person_puts_down_object',
            'person_reads_document',
            'person_rides_bicycle',
            'person_sits_down',
            'person_stands_up',
            'person_steals_object',
            'person_talks_on_phone',
            'person_talks_to_person',
            'person_texts_on_phone',
            'person_transfers_object',
            'person_unloads_vehicle',
            'vehicle_drops_off_person',
            'vehicle_makes_u_turn',
            'vehicle_picks_up_person',
            'vehicle_reverses',
            'vehicle_starts',
            'vehicle_stops',
            'vehicle_turns_left',
            'vehicle_turns_right'
        ]
    
        self.MEVA_objects = ['person',
            'vehicle',
            'bicycle',
            'bag',
            'receptacle',
            'other']

        self.num_verbs = len(self.MEVA_activities)
        self.num_objects = len(self.MEVA_objects)

    def __len__(self):
        return len(self.file_names)
    
    def find_annotation(self, data_root_path, anno_root_path) -> dict:
        annotations_files = list(anno_root_path.rglob('*.yml'))
        annotations_files = natsorted(annotations_files)

        act_files = annotations_files[::3]
        geo_files = annotations_files[1::3]
        type_files = annotations_files[2::3]

        anno_dict = defaultdict()

        assert len(act_files) == len(geo_files) == len(type_files), 'The number of annotation files does not match'

        for act_file, geo_file, type_file in zip(act_files, geo_files, type_files):
            # Hack to deal with . or -* in the name file - FFS
            act = act_file.name[:-len('.activities.yml')]
            geo = geo_file.name[:-len('.geom.yml')]
            _type = type_file.name[:-len('.types.yml')]

            if self.debug:
                # Remove geo files that have more than 1mb:
                if geo_file.stat().st_size > 1000000:
                    #print(f'File {geo_file} has more than 1mb')
                    continue

            assert act == geo == _type, 'The annotation files do not match'

            try:
                name_parts = act.split('.')
                date = name_parts[0]
                time = name_parts[1].split('-')[0]

                video_path = data_root_path / date / time / (act+'.r13.avi')

                if not video_path.exists():
                    end_time = name_parts[2].split('-')[0]

                    video_path = data_root_path / date / end_time / (act+'.r13.avi')
                    if not video_path.exists():
                        print(f'{video_path} does not exist')
                        continue

                anno_dict[act] = {'video_path': video_path, 'act_file': act_file, 'geo_file': geo_file, 'type_file': type_file}
            except Exception as e:
                print(f'Error processing {act} - {e}')
                continue
        
        return anno_dict

    @lru_cache(maxsize=10)
    def __getitem__(self, index: Union[int, str]) -> tuple:
        if type(index) == int:
            index = self.file_names[index]
        print(f'Getting {index}')

        file_dict = self.annotations[index]
        video_path = file_dict['video_path']
        act_file = file_dict['act_file']
        geo_file = file_dict['geo_file']
        type_file = file_dict['type_file']

        #print('Reading boxes')
        boxes = parse_geom_yaml(geo_file)
        #print('Reading types')
        cat_id_pairs = parse_types_yaml(type_file)
        #print('Reading actions')
        acts_list = parse_activity_yaml(act_file)
        #print('Organizing actions')
        acts_list = [act.organize(True) for act in acts_list]
        # Old code, ngl I don't remember what it does
        for box in boxes:
            for act in acts_list:
                if act.check_add_box(box):
                    break
        # Add new code to fix the issue with the boxes
        for box in boxes:
            for action in acts_list:
                if box.id1 in action.actors_time_dict.keys():
                    if box.ts in action.actors_time_dict[box.id1]['timespan']:
                        if 'boxes' not in action.actors_time_dict[box.id1]:
                            action.actors_time_dict[box.id1]['boxes'] = [box]
                        else:
                            action.actors_time_dict[box.id1]['boxes'].append(box)
        print(f'Done!')

        if self.output_format == 'DDS':
            if len(acts_list) == 0:  # If there's no activities, return empty dict
                return video_path, {}
            
            #print('Converting to DDS')
            frame_dict = self.MEVA2DDS(acts_list, cat_id_pairs)
            return video_path, frame_dict

        return video_path, boxes, acts_list, cat_id_pairs

    def filter_files(self):
        location_dict = defaultdict(list)
        camera_dict = defaultdict(list)
        start_time_dict = defaultdict(list)
        date_dict = defaultdict(list)
        for file_name in self.file_names:
            date, start_time, end_time, location, camera = file_name.split('.')
            location_dict[location].append(file_name)
            camera_dict[camera].append(file_name)
            start_time_dict[start_time[:-3]].append(file_name)
            date_dict[date].append(file_name)
        return {'location': location_dict, 'camera': camera_dict, 'start_time': start_time_dict, 'date': date_dict}

    def MEVA2DDS(self, acts_list, cat_id_pairs):
        # DDS format:
        DDS_keys = ['orig_size',
        'size',
        'filename',
        'boxes',
        'labels',
        'iscrowd',
        'area',
        'obj_labels',
        'verb_labels',
        'sub_labels',
        'sub_boxes',
        'obj_boxes',
        'union_boxes']

        frame_dict = defaultdict(dict)
        for action in acts_list:
            for frame_number, box_info in action.boxes.items():
                if frame_number not in frame_dict:
                    for key in DDS_keys:
                        frame_dict[frame_number][key] = []

                if len(box_info) == 1:
                    box = box_info[0]    
                    bbox = [box.x0, box.y0, box.x1, box.y1]
                    category = list(cat_id_pairs[box.id1].keys())[0]
                    frame_dict[frame_number]['sub_labels'].append(self.MEVA_objects.index(category))
                    frame_dict[frame_number]['sub_boxes'].append(bbox)
                    frame_dict[frame_number]['boxes'].append(bbox)
                    frame_dict[frame_number]['boxes'].append(bbox)
                    frame_dict[frame_number]['obj_boxes'].append(bbox)
                    category = list(cat_id_pairs[box.id1].keys())[0]
                    frame_dict[frame_number]['obj_labels'].append(self.MEVA_objects.index('other'))
                    frame_dict[frame_number]['verb_labels'].extend([self.MEVA_activities.index(action.name)])
                    #frame_dict[frame_number]['union_boxes'].append(union_box(bbox, bbox))
                else:
                    actor_class = [list(cat_id_pairs[box.id1].keys())[0] for box in box_info]
                    main_actor = action.name.split('_')[0]

                    if main_actor == 'hand': # WTF is this garbage
                        main_actor = 'person'

                    actors_idx = [i for i, actor in enumerate(actor_class) if main_actor in actor]
                    object_idx = [i for i in range(len(actor_class)) if i not in actors_idx]

                    if len(actor_class) == len(actors_idx) and len(object_idx) == 0: # Person(s) doing something to person
                        object_idx = [actors_idx[-1]]
                        actors_idx = actors_idx[:-1]

                        if type(actors_idx) is not list:
                            actors_idx = [actors_idx]
                        
                    # Apparently there's no order guarantee on the annotations FFS
                    # PAINFUL
                    #assert len(object_idx) == 1, 'PQP ein' # Apparently one person can have 2 associated objects on the same action
                    # if len(object_idx) > 1:
                    #     print(f'Found {len(object_idx)} objects for {action.name} at frame {frame_number}')
                    #     continue
                    # Ok but no frame with 4 items

                    if len(actors_idx) >= len(object_idx): # 1 actor 1+ objects
                        for idx in actors_idx:
                            person_box = box_info[idx]
                            person_bbox = [person_box.x0, person_box.y0, person_box.x1, person_box.y1]
                            frame_dict[frame_number]['boxes'].append(person_bbox)
                            category = actor_class[idx]
                            frame_dict[frame_number]['sub_labels'].append(self.MEVA_objects.index(category))
                            frame_dict[frame_number]['sub_boxes'].append(person_bbox)

                            object_box = box_info[object_idx[0]]
                            object_bbox = [object_box.x0, object_box.y0, object_box.x1, object_box.y1]
                            frame_dict[frame_number]['boxes'].append(object_bbox)
                            category = actor_class[object_idx[0]]
                            frame_dict[frame_number]['obj_labels'].append(self.MEVA_objects.index(category))
                            frame_dict[frame_number]['obj_boxes'].append(object_bbox)

                            frame_dict[frame_number]['verb_labels'].extend([self.MEVA_activities.index(action.name)])
                            #frame_dict[frame_number]['union_boxes'].append(union_box(person_bbox, object_bbox))
                    else: # Since the max number of boxes is 3, this will go when there's 2 objects and 1 actor
                        for idx in object_idx:
                            actor_box = box_info[actors_idx[0]]
                            bbox = [actor_box.x0, actor_box.y0, actor_box.x1, actor_box.y1]
                            frame_dict[frame_number]['boxes'].append(bbox)
                            category = actor_class[actors_idx[0]]
                            frame_dict[frame_number]['sub_labels'].append(self.MEVA_objects.index(category))
                            frame_dict[frame_number]['sub_boxes'].append(bbox)

                            object_box = box_info[idx]
                            bbox = [object_box.x0, object_box.y0, object_box.x1, object_box.y1]
                            frame_dict[frame_number]['boxes'].append(bbox)
                            category = actor_class[idx]
                            frame_dict[frame_number]['obj_labels'].append(self.MEVA_objects.index(category))
                            frame_dict[frame_number]['obj_boxes'].append(bbox)

                            frame_dict[frame_number]['verb_labels'].extend([self.MEVA_activities.index(action.name)])
                            #frame_dict[frame_number]['union_boxes'].append(union_box(person_bbox, object_bbox))

                assert len(frame_dict[frame_number]['sub_labels']) == len(frame_dict[frame_number]['obj_labels']) == len(frame_dict[frame_number]['verb_labels']), 'Not the same number of labels'
                #for _ in range(max(1, len(box_info)-1)):
                #frame_dict[frame_number]['verb_labels'].extend([self.MEVA_activities.index(action.name)] * len(frame_dict[frame_number]['sub_labels']))
        
        #print('Casting to tensor and making final ajustments')
        for frame_number in frame_dict.keys():
            verb_one_hot = []
            for value in frame_dict[frame_number]['verb_labels']:
                zeros = [0] * self.num_verbs
                zeros[value] = 1
                verb_one_hot.append(zeros)
            #verb_one_hot = [1 if idx in frame_dict[frame_number]['verb_labels'] else 0 for idx in range(self.num_verbs)]
            frame_dict[frame_number]['verb_labels'] = verb_one_hot
            frame_dict[frame_number] = [{k: torch.tensor(v) for k, v in frame_dict[frame_number].items() if k != 'filename'}][0]
            frame_dict[frame_number]['union_boxes'] = union_box(frame_dict[frame_number]['sub_boxes'], frame_dict[frame_number]['obj_boxes'])
            
            frame_dict[frame_number]['verb_labels'] = frame_dict[frame_number]['verb_labels'].float()
            for key, value in frame_dict[frame_number].items():
                if 'boxes' in key:
                    frame_dict[frame_number][key] = box_xyxy_to_cxcywh(value)

        return frame_dict


def dump_video_with_anno(video_path, boxes, acts_list, cat_id_pairs, output_path):
    print(f'Dumping {video_path} with {len(acts_list)} activities at {output_path}')
    output_path.mkdir(parents=True, exist_ok=True)

    video_file = cv2.VideoCapture(str(video_path))
    frame_counter = 0
    while True:
        print(f'{frame_counter:06d}')
        ret, frame = video_file.read()
        if ret:
            frame_counter += 1
        else:
            break

        for action in acts_list:
            if min(list(action.boxes.keys())) <= frame_counter <= max(list(action.boxes.keys())):
                boxes = action.boxes[frame_counter]
                for box in boxes:
                    bbox_top_left, bbox_bottom_right = (box.x0, box.y0), (box.x1, box.y1)
                    cv2.rectangle(frame, bbox_top_left, bbox_bottom_right, (0, 0, 255), 2)
                    cv2.putText(frame, action.name, bbox_top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(str(output_path / f'frame_{frame_counter:06d}.jpg'), frame)
    video_file.release()

def dump_video_with_DDS_anno(video_path, DDS_dict, output_path, MEVA_activities):
    output_path.mkdir(parents=True, exist_ok=True)

    video_file = cv2.VideoCapture(str(video_path))
    frame_counter = 0
    while True:
        print(f'{frame_counter:06d}')
        ret, frame = video_file.read()
        if ret:
            frame_counter += 1
        else:
            break

        if frame_counter not in DDS_dict:
            continue

        info = DDS_dict[frame_counter]
        #for info in DDS_dict[frame_counter]:
        for sub_box, obj_box, action in zip(info['sub_boxes'], info['obj_boxes'], info['verb_labels']):
            bbox_top_left, bbox_bottom_right = (sub_box[0], sub_box[1]), (sub_box[2], sub_box[3])
            cv2.rectangle(frame, bbox_top_left, bbox_bottom_right, (0, 0, 255), 2)
            cv2.putText(frame, MEVA_activities[action], bbox_top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            bbox_top_left, bbox_bottom_right = (obj_box[0], obj_box[1]), (obj_box[2], obj_box[3])
            cv2.rectangle(frame, bbox_top_left, bbox_bottom_right, (255, 0, 0), 2)
        cv2.imwrite(str(output_path / f'frame_{frame_counter:06d}.jpg'), frame)

    video_file.release()


def build(image_set, args):
    # TODO: Build test set
    if image_set != 'train':
        raise NotImplementedError('Only train set is supported')
    base_path = Path(args.hoi_path)
    anno_root_path = base_path / 'meva-data-repo/annotation/DIVA-phase-2/MEVA/kitware-meva-training'
    data_root_path = base_path / 'data'

    dataset = MEVA(data_root_path, anno_root_path, args)
    return dataset
    

if __name__ == '__main__':
    anno_root_path = Path('/data/raphael/dds_skywalker/meva/meva-data-repo/annotation/DIVA-phase-2/MEVA')
    data_root_path = Path('/data/raphael/dds_skywalker/meva/data/')

    test_dataset = MEVA(data_root_path, anno_root_path)

    files = ['2018-03-05.13-10-00.13-15-00.bus.G340',
             '2018-03-05.13-10-00.13-15-00.hospital.G341',
             '2018-03-05.13-10-01.13-15-01.bus.G331',
             '2018-03-05.13-15-00.13-20-00.school.G424',
             '2018-03-05.13-15-01.13-20-01.bus.G331',
             '2018-03-05.13-20-00.13-25-00.bus.G506',
             '2018-03-05.13-20-00.13-25-00.admin.G326']
    
    export_dict = {}
    for file in files:
        video_path, boxes, acts_list, cat_id_pairs = test_dataset[file]
        export_dict[file] = {}
        for action in acts_list:
            #actors = [actor.id for actor in action.actors]
            timestamp = {k: {"timestamp" :[min(v2), max(v2)]} for k, v in action.actors_time_dict.items() for k2, v2 in v.items() if k2 == 'timespan'}
            name = action.name
            if name in export_dict[file]:
                export_dict[file][name].append(timestamp)
            else:
                export_dict[file][name] = [{"actors": timestamp}]

        id_pairs = {k: k2 for k, v in cat_id_pairs.items() for k2 in v.keys()}
        export_dict[file]['id_pairs'] = id_pairs

    import yaml
    with open('meva_test.yaml', 'w') as file:
        yaml.dump(export_dict, file)


    
    # video_path, boxes, acts_list, cat_id_pairs = test_dataset['2018-03-09.10-10-00.10-15-00.school.G330']
    # frame_dict = MEVA2DDS(acts_list, cat_id_pairs)
    # output_path = Path(f'/media/hdd6/raphael/meva/{video_path.name}')
    # dump_video_with_DDS_anno(video_path, frame_dict, output_path)
    # for _ in range(20):
    #     video_path, boxes, acts_list, cat_id_pairs = test_dataset[random.randint(0, len(test_dataset)-1)]
    #     if len(acts_list) == 0:
    #         continue
    #     #video_path, boxes, acts_list, cat_id_pairs = test_dataset['2018-03-12.10-40-01.10-45-01.school.G419']
    #     #output_path = Path('/media/hdd6/raphael/meva/2018-03-07.17-35-01.17-40-01.school.G328')
    #     #dump_video_with_anno(video_path, boxes, acts_list, cat_id_pairs, output_path)
    #     frame_dict = MEVA2DDS(acts_list, cat_id_pairs)
    #     output_path = Path(f'/media/hdd6/raphael/meva/{video_path.name}')
    #     dump_video_with_DDS_anno(video_path, frame_dict, output_path)

    # for video_path, _, acts_list, cat_id_pairs in tqdm(test_dataset):
    #     idk = MEVA2DDS(acts_list, cat_id_pairs)


            