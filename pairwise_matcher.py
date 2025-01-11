import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
from pathlib import Path
from natsort import natsorted
import argparse
from MEVA import build
from collections import defaultdict
from PIL import ImageDraw
import random
import ast
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--hoi_path', help='Path to HOI file', default='/home/raphael/Documents/skywalker_6/raphael/meva')
parser.add_argument('--dataset_file', type=str, default='MEVA')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
# parser.add_argument('--filter_by', nargs='+', help='Filter by', default=['date', 'start_time'])
# parser.add_argument('--filter_val', nargs='+', help='Filter value', default=['2018-03-05', '13-15'])
parser.add_argument('--filter_by', nargs='+', help='Filter by', default=[])
parser.add_argument('--filter_val', nargs='+', help='Filter value', default=[])
parser.add_argument('--max_cols', type=int, default=2)
parser.add_argument('--camera_pairs', nargs='+', help='Camera pairs to perform matching', default=['G331', 'G506'])
args = parser.parse_args()

class ZoomableImage:
    def __init__(self, _orig_image, _bbox, _category_frame, _row, _col, actor_id):
        self.is_zoomed = False
        self.orig_img = copy.deepcopy(_orig_image)
        self.bbox = _bbox
        self.actor_id = actor_id  # Store actor_id
        max_size = (600, 600)
        PADDING = 20

        max_x, max_y, = self.orig_img.size
        padded_box = [_bbox[0] - PADDING, _bbox[1] - PADDING, _bbox[2] + PADDING, _bbox[3] + PADDING]
        # padded_box[0] = max(0, padded_box[0])
        # padded_box[1] = max(0, padded_box[1])
        padded_box[0] = min(max_x, padded_box[0])
        padded_box[1] = min(max_y, padded_box[1])
        padded_box[2] = min(max_x, padded_box[2])
        padded_box[3] = min(max_y, padded_box[3])

        if padded_box[0] >= padded_box[2] or padded_box[1] >= padded_box[3]:
            print(f'Invalid bbox: {padded_box}')
        
        self.zoomed_image  = self.orig_img.crop(padded_box).copy()
        scale = 1
        if self.zoomed_image.size[0] < self.zoomed_image.size[1]:
            scale = self.zoomed_image.size[0] / self.zoomed_image.size[1]
            new_width = int(max_size[0] * scale)
            max_size = (new_width, max_size[1])
            self.zoomed_image = self.zoomed_image.resize((new_width, max_size[1]), resample = Image.BOX)
        else:
            scale = self.zoomed_image.size[1] / self.zoomed_image.size[0]
            new_height = int(max_size[1] * scale)
            max_size = (max_size[0], new_height)
            self.zoomed_image = self.zoomed_image.resize((max_size[0], new_height), resample = Image.BOX)

        # Draw rectangle on the original image
        draw = ImageDraw.Draw(self.orig_img)
        draw.rectangle(_bbox, outline=(255, 0, 0), width=5)

        # Convert bbox to respect with zoomed_image
        x0 = PADDING * 2
        y0 = PADDING * 2
        x1 = max_size[0] - PADDING * 2
        y1 = max_size[1] - PADDING * 2
        draw = ImageDraw.Draw(self.zoomed_image)
        draw.rectangle((x0, y0, x1, y1), outline=(255, 0, 0), width=3)

        # Resize images for conformity
        self.orig_img.thumbnail((600, 600))
        self.zoomed_image.thumbnail((self.orig_img.size[1], self.orig_img.size[1]))
        self.orig_photo = ImageTk.PhotoImage(self.orig_img)
        self.zoomed_photo = ImageTk.PhotoImage(self.zoomed_image)   

        # Image label
        self.label = tk.Label(_category_frame, image=self.orig_photo)
        self.label.bind("<Button-1>", lambda e: self.toggle_zoom(e))  # Left click to zoom
        self.label.bind("<Button-3>", lambda e: self.add_actor_id_to_entry())  # Right click to add actor_id
        self.label.grid(row=_row, column=_col, padx=5, pady=5)

    def toggle_zoom(self, _event):
        """Toggles zoom in on an image."""
        if self.is_zoomed:
            self.show_original_image()
        else:
            self.show_zoomed_image()
        self.is_zoomed = not self.is_zoomed    

    def show_original_image(self):
        """Shows the original image."""
        self.label.configure(image=self.orig_photo)
        self.label.image = self.orig_photo    

    def show_zoomed_image(self):
        """Shows the zoomed image."""
        self.label.configure(image=self.zoomed_photo)
        self.label.image = self.zoomed_photo
    
    def add_actor_id_to_entry(self):
        """Adds the actor ID to the annotation entry."""
        current_text = annotation_entry.get()
        if current_text:
            annotation_entry.insert(tk.END, f", {self.actor_id}")
        else:
            annotation_entry.insert(tk.END, str(self.actor_id))

MAX_COLS = args.max_cols

def on_frame_configure(canvas):
    '''Reset the scroll region to encompass the inner frame'''
    canvas.configure(scrollregion=canvas.bbox("all"))

def get_videos_iterator(all_videos):
    """Yield videos one by one."""
    if not all_videos:
        messagebox.showinfo('No videos', 'No videos found')
        root.quit()
    for video in all_videos:
        yield video


def update_annotation(cat_id_pairs, c_dict, video_name):
    updated_cat_id_pairs = {}
    
    if video_name not in c_dict:
        print(f'No correspondence dict for {video_name}')
        return None
    
    c_list = ast.literal_eval(c_dict[video_name].strip())

    for new_idx, _c_list in enumerate(c_list):
        for entry in _c_list:
            updated_cat_id_pairs[entry] = new_idx

    if len(updated_cat_id_pairs) != len(cat_id_pairs):
        print(f'Error in updating cat_id_pairs for {video_name}')
        original_keys = set(cat_id_pairs.keys())
        updated_keys = set(updated_cat_id_pairs.keys())
        print(f'Missing: {original_keys - updated_keys} - Consider checking the annotations manually')
        
    return updated_cat_id_pairs


def load_video(filename, actions, categories, frame_files):
    global correspondence_dict
    actions = sorted(actions)
    all_frames = set()
    for action in actions:
        frame_ids = list(action.boxes.keys())
        all_frames.update(frame_ids)
    all_frames = sorted(list(all_frames))

    categories = update_annotation(categories, correspondence_dict, filename)

    if not categories or set(categories.keys()) == set('skipped'):
        print(f'No categories found for {filename}')
        return None

    cat = set(categories.values())

    new_dict = defaultdict(list)

    for action in actions:
        for actor_id in action.actors_time_dict.keys():
            if categories[actor_id] in cat:
                random_box = random.choice(action.actors_time_dict[actor_id]['boxes'])
                random_idx = random_box.ts
                first_frame = all_frames.index(random_idx)

                frame = Image.open(str(frame_files[first_frame]))
                box = (random_box.x0, random_box.y0, random_box.x1, random_box.y1)
                # draw = ImageDraw.Draw(frame)
                # draw.rectangle(box, outline=(255, 0, 0), width=5)

                new_dict[filename].append((frame, random_idx, categories[actor_id], box))
                cat.remove(categories[actor_id])

    # for k,v in new_dict.items():
    #     new_dict[k] = sorted(v, key=lambda x: x[1])

    return new_dict


def process_next_video():
    global first_dict, second_dict, pairs_iterator
    
    try:
        first_file, second_file = next(pairs_iterator)

        first_frames = list((frames_folder / first_file).glob('*.jpg'))
        second_frames = list((frames_folder / second_file).glob('*.jpg'))
        first_frames = natsorted(first_frames, key=lambda x: x.stem)
        second_frames = natsorted(second_frames, key=lambda x: x.stem)

        _, _, actions1, categories1 = dataset[first_file]
        _, _, actions2, categories2 = dataset[second_file]

        if not actions1 or not actions2:
            print('No actions found @ process_next_video')
            # write_annotation()
            return process_next_video()
        
        first_dict = load_video(first_file, actions1, categories1, first_frames)
        second_dict = load_video(second_file, actions2, categories2, second_frames)

        if not first_dict or not second_dict:
            return process_next_video()

        load_images(canvas, canvas_r, left_frame, right_frame)

    except StopIteration:
        print('No more videos')
        #write_annotation()
        root.quit()
        return


def load_images(canvas, canvas_r, left_frame, right_frame):
    global first_dict, second_dict, left_name, right_name, images_first, images_second, labels_first, labels_second
    
    if not first_dict and not second_dict:
        print('No actions found @ load_images')
        write_annotation()
        return process_next_video()
    
            # Clear the current grid
    for widget in left_frame.winfo_children():
        widget.destroy()

    for widget in right_frame.winfo_children():
        widget.destroy()
    
    if first_dict:
        images_first = []
        labels_first = []
        _row = 0
        _col = 0
        left_name = list(first_dict.keys())[0]
        category_frame_left = tk.LabelFrame(left_frame, text=left_name, padx=5, pady=5)
        category_frame_left.grid(row=_row, column=0, columnspan=5, sticky='ew', padx=5, pady=5)
        #for frame, random_idx, actor_id in first_dict[left_name]:
        for frame, random_idx, actor_id, box in first_dict[left_name]:
            #frame.thumbnail((600, 600))
            #img = ImageTk.PhotoImage(frame)
            img = ZoomableImage(frame, box, category_frame_left, _row, _col, actor_id)
            images_first.append(img.orig_photo)
            labels_first.append(img.label)

            # label = tk.Label(left_frame, image=img)
            # label.grid(row=_row, column=_col, padx=5, pady=5)
            # labels_first.append(label)

            caption_label = tk.Label(category_frame_left, text=f'Actor {actor_id} - Frame {random_idx}')
            caption_label.grid(row=_row+1, column=_col, padx=5, pady=5)

            _col+=1
            if _col >= MAX_COLS:
                _col = 0
                _row+=2

    if second_dict:
        images_second = []
        labels_second = []
        _row = 0
        _col = 0
        right_name = list(second_dict.keys())[0]
        category_frame_right = tk.LabelFrame(right_frame, text=right_name, padx=5, pady=5)
        category_frame_right.grid(row=_row, column=0, columnspan=5, sticky='ew', padx=5, pady=5)
        #for frame, random_idx, actor_id in second_dict[right_name]:
        for frame, random_idx, actor_id, box in second_dict[right_name]:
            #frame.thumbnail((600, 600))
            #img = ImageTk.PhotoImage(frame)
            img = ZoomableImage(frame, box, category_frame_right, _row, _col, actor_id)
            images_second.append(img.orig_photo)
            labels_second.append(img.label)

            # label = tk.Label(right_frame, image=img)
            # label.grid(row=_row, column=_col, padx=5, pady=5)
            # labels_second.append(label)

            caption_label = tk.Label(category_frame_right, text=f'Actor {actor_id} - Frame {random_idx}')
            caption_label.grid(row=_row+1, column=_col, padx=5, pady=5)

            _col+=1
            if _col >= MAX_COLS:
                _col = 0
                _row+=2

    canvas.configure(scrollregion=canvas.bbox("all"))
    canvas_r.configure(scrollregion=canvas_r.bbox("all"))


def read_correspondence_dict(file_location):
    correspondence_dict = defaultdict(list)
    try:
        with open(file_location, 'r') as f:
            for line in f:
                try:
                    key, value = line.split(':')
                    correspondence_dict[key] = value
                except ValueError as ve:
                    print(f'Error on read_correspondence_dict: {ve}')
                    continue
        return correspondence_dict
    except FileNotFoundError as e:
        print(f'Error on read_correspondence_dict: {e}')


def write_annotation():
    global video_id_mapping
    try:
        with open(video_id_mapping, 'a') as f:
            for k, v in id_map_dict.items():
                f.write(f'{k}:{v}\n')
        print(f'Annotation saved to {video_id_mapping}')
    except Exception as e:
        print(f'Error on write_annotation: {e}')


def is_digit_or_negative(s):
    # Remove leading and trailing whitespace
    s = s.strip()
    # Check if the string is just a digit (positive number)
    if s.isdigit():
        return True
    # Check if the string is a negative number
    elif s.startswith('-') and s[1:].isdigit():
        return True
    return False

def submit_annotation(event=None):
    global id_map_dict, first_dict, second_dict, left_name, right_name
    annotation_indexes = annotation_entry.get().split(',')
    annotation_indexes = [int(index.strip()) for index in annotation_indexes if is_digit_or_negative(index.strip())]
    print(f'Annotation indexes: {annotation_indexes}')
    if len(annotation_indexes) != 2:
        print('Invalid annotation')
        return
    id_map_dict[f'{left_name}:{right_name}'].append(annotation_indexes)
    id_left, id_right = annotation_indexes[0], annotation_indexes[1]

    if id_left >= 0:
        idx_pop_left = [idx for idx, (_, _, actor_id) in enumerate(first_dict[left_name]) if actor_id == id_left]
        assert len(idx_pop_left) == 1, f'Error: {idx_pop_left}'
        first_dict[left_name].pop(idx_pop_left[0])
        if len(first_dict[left_name]) == 0:
            first_dict.clear()
    if id_right >= 0:
        idx_pop_right = [idx for idx, (_, _, actor_id) in enumerate(second_dict[right_name]) if actor_id == id_right]
        assert len(idx_pop_right) == 1, f'Error: {idx_pop_right}'
        second_dict[right_name].pop(idx_pop_right[0])
        if len(second_dict[right_name]) == 0:
            second_dict.clear()

    annotation_entry.delete(0, tk.END)

    load_images(canvas, canvas_r, left_frame, right_frame)


def skip_pair(event=None):
    global id_map_dict, left_name, right_name
    print(f'Skipping pair: {left_name}:{right_name}')
    id_map_dict[f'{left_name}:{right_name}'] = []
    write_annotation()
    process_next_video()


if __name__ == '__main__':

    ### Initial UI Setup

    root = tk.Tk()
    root.title('Annotation Tool')

    # Create a canvas and a scrollbar attached to the canvas
    canvas = tk.Canvas(root)
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)  # This frame will contain your images and captions

    # Configure canvas
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    scrollbar.pack(side="left", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    # This is the magic that makes the frame inside the canvas scrollable
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

    scrollable_frame.bind("<Configure>", lambda event, canvas=canvas: on_frame_configure(canvas))

    # Now, use `scrollable_frame` as your `current_frame` for adding images
    left_frame = scrollable_frame

    past_frames = []
    # Create a label to display the current frame
    left_frame_label = tk.Label(left_frame)
    left_frame_label.pack()

    # Create a right frame and label
    canvas_r = tk.Canvas(root)
    scrollbar_r = ttk.Scrollbar(root, orient="vertical", command=canvas_r.yview)
    scrollable_frame_r = tk.Frame(canvas_r)  # This frame will contain your images and captions

    # Configure canvas
    canvas_r.configure(yscrollcommand=scrollbar_r.set)
    canvas_r.bind('<Configure>', lambda e: canvas_r.configure(scrollregion=canvas_r.bbox("all")))
    scrollbar_r.pack(side="right", fill="y")
    canvas_r.pack(side="right", fill="both", expand=True)

    # This is the magic that makes the frame inside the canvas scrollable
    canvas_r.create_window((0, 0), window=scrollable_frame_r, anchor="nw")

    scrollable_frame_r.bind("<Configure>", lambda event, canvas_r=canvas_r: on_frame_configure(canvas_r))

    right_frame = scrollable_frame_r
    right_frame_label = tk.Label(right_frame, text='Left Frame')
    right_frame_label.pack()

    # Add an entry widget for annotations
    annotation_frame = tk.Frame(root)
    annotation_label = tk.Label(annotation_frame, text="Enter the unique ids of the objects to remove (comma separated)")
    annotation_entry = tk.Entry(annotation_frame)
    submit_button = tk.Button(annotation_frame, text="Submit Annotation", command=submit_annotation)
    root.bind('<Return>', lambda e: submit_annotation())  # Bind the return key to submit the annotation
    root.bind('<Escape>', skip_pair)  # Bind the escape key to skip the pair
    annotation_frame.pack(side='bottom', fill='x', expand=False)  # Pack the annotation frame at the bottom
    annotation_label.pack(side='top', fill='x', expand=False)  # Pack the label inside the annotation frame
    annotation_entry.pack(side='top', fill='x', expand=False)  # Pack the entry below the label
    submit_button.pack(side='top', fill='x', expand=False)  # Pack the submit button below the entry

    # Some Global Variables
    correspondence_dict = read_correspondence_dict(Path(args.hoi_path) / 'correspondence_dict.txt') 
    filename = None
    class_dict = defaultdict(list)
    right_name = None
    left_name = None
    id_map_dict = defaultdict(list)
    video_id_mapping = Path(args.hoi_path) / 'video_id_mapping.txt'
    #video_id_mapping = Path('video_id_mapping.txt')

    #frames_folder = Path('/home/raphael/Documents/skywalker_6/raphael/meva/frames')
    frames_folder = Path(args.hoi_path) / 'frames'
    dataset = build(image_set='train', args=args)
    frame_files = []
    pairs = []

    first_dict = {}
    second_dict = {}

    assert len(args.camera_pairs) <= 2, 'Error: Camera pairs must be a single pair (e.g. G331 G332) or a single camera (e.g. G331 G331)'
    camera_1, camera_2 = args.camera_pairs
    left_camera_files = natsorted(dataset.filter_files()['camera'][camera_1])
    right_camera_files = natsorted(dataset.filter_files()['camera'][camera_2])

    if camera_1 == camera_2:
        for i in range(len(left_camera_files)-1):
            first_file = left_camera_files[i]
            second_file = left_camera_files[i+1]

            date1, start1, end1, _, _ = first_file.split('.')
            date2, start2, end2, _, _ = second_file.split('.')

            start1 = start1[:-3]
            end1 = end1[:-3]
            start2 = start2[:-3]
            end2 = end2[:-3]

            if date1 != date2:
                continue

            if end1 == start2:
                pairs.append((first_file, second_file))

    else:
        for first_file, second_file in zip(left_camera_files, right_camera_files):
            pairs.append((first_file, second_file))


    try:
        with open(video_id_mapping, 'r') as f:
            for line in f:
                try:
                    first, second, _ = line.split(':')
                    for idx, pair in enumerate(pairs):
                        if pair[0] == first and pair[1] == second:
                            pairs.pop(idx)
                            break
                except ValueError as ve:
                    print(f'Error on video_id_mapping: {ve}')
                    continue
    except FileNotFoundError as e:
        print(f'{video_id_mapping} not found, creating a new file')
        

    pairs_iterator = get_videos_iterator(pairs)
    process_next_video()
    root.mainloop()





