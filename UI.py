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


parser = argparse.ArgumentParser()
parser.add_argument('--hoi_path', help='Path to HOI file')
parser.add_argument('--dataset_file', type=str, default='MEVA')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
parser.add_argument('--filter_by', nargs='+', help='Filter by', default=[])
parser.add_argument('--filter_val', nargs='+', help='Filter value', default=[])
args = parser.parse_args()

output_file = Path(args.hoi_path) / 'correspondence_dict.txt'

assert len(args.filter_by) == len(args.filter_val), 'Filter by and filter value must have the same length'

class ZoomableImage:
    def __init__(self, _orig_image, _bbox, _category_frame, _row, _col):
        self.is_zoomed = False
        self.orig_img = _orig_image
        self.bbox = _bbox
        max_size = (600, 600)
        PADDING = 20

        padded_box = (_bbox[0] - PADDING, _bbox[1] - PADDING, _bbox[2] + PADDING, _bbox[3] + PADDING)
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
        self.label.bind("<Button-1>", lambda e: self.toggle_zoom(e))
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

def write_annotation():
    try:
        with open(output_file, 'a') as f:
            for k, v in correspondence_dict.items():
                f.write(f'{k}: {v}\n')
        
        print('Annotation saved')
        correspondence_dict.clear()
    except Exception as e:
        print(f'Error on write_annotation: {e}')


def on_frame_configure(canvas):
    '''Reset the scroll region to encompass the inner frame'''
    canvas.configure(scrollregion=canvas.bbox("all"))


def get_videos_iterator(all_videos):
    """Yield videos one by one."""
    for video in all_videos:
        yield video



def load_images(canvas, current_frame):
    global image_labels, images
    # Check if there are no more images to display
    if not class_dict or all(len(v) == 0 for v in class_dict.values()):
        write_annotation()
        process_next_video()  # Load more data if empty
        return  # Exit this function to avoid executing the rest of its code prematurely

    # Clear the current grid
    for widget in current_frame.winfo_children():
        widget.destroy()
    image_labels = []
    images = []

    row = 0
    col = 0
    if class_dict:
        category = list(class_dict.keys())[0]
        img_actor_pairs = class_dict[category]
        # Create a frame for each category within the scrollable frame
        category_frame = tk.LabelFrame(current_frame, text=category, padx=5, pady=5)
        category_frame.grid(row=row, column=0, columnspan=5, sticky='ew', padx=5, pady=5)

        col = 0  # Column index for images within a category
        row = 1  # Start image row within the category frame
        for img, actor_id, frame_idx, bbox in img_actor_pairs:
            z_img = ZoomableImage(img, bbox, category_frame, row, col)
            images.append(z_img.orig_photo)  # Keep a reference!
            image_labels.append(z_img.label)  # Keep a reference!

            # Caption label with actor_id
            caption_label = tk.Label(category_frame, text=f'Actor ID: {actor_id} - Frame: {frame_idx}')
            caption_label.grid(row=row+1, column=col, padx=5, pady=5)

            col += 1
            if col >= 4:  # Adjust based on how many images per row you want
                col = 0
                row += 2  # Adjust row increment to accommodate image and caption
    else:
        messagebox.showinfo("End of list", "No more images to display")
        with open(output_file, 'a') as f:
            for key, value in correspondence_dict.items():
                f.write(f'{key}: {value}\n')
        root.quit()
    
    # Update the scrollable region to encompass the updated frame
    canvas.configure(scrollregion=canvas.bbox("all"))


def submit_annotation(event=None):
    # Function to handle the submission of an annotation
    annotation_indexes = annotation_entry.get().split(',')
    print(f'Annotation indexes: {annotation_indexes}')
    annotation_indexes = [int(index.strip()) for index in annotation_indexes if index.strip().isdigit()]
    correspondence_dict[filename].append(annotation_indexes)
    first_entry = list(class_dict.keys())[0]
    for id_to_remove in annotation_indexes:
        id_to_remove = int(id_to_remove)
        for i, (_, actor_id, _) in enumerate(class_dict[first_entry]):
            if actor_id == id_to_remove:
                class_dict[first_entry].pop(i)

    if not class_dict[first_entry]:
        class_dict.pop(first_entry)

    annotation_entry.delete(0, tk.END)  # Clear the entry widget

    load_images(canvas, current_frame)
    # Here you would save the annotation or process it as needed


def process_next_video():
    global class_dict  # Ensure class_dict is accessible and modifiable
    global filename
    class_dict.clear()  # Optionally clear the existing data if you want to start fresh

    files = get_updated_file_list()
    video_iterator = get_videos_iterator(files)
    
    try:
    #for current_file in all_videos:
        filename = next(video_iterator)
        frame_files = list((frames_folder / filename).glob('*.jpg'))
        frame_files = natsorted(frame_files, key=lambda x: x.stem)
        _, _, actions, categories = dataset[filename]

        if not actions:
            print(f'No actions found for {filename}')
            correspondence_dict[filename] = []
            write_annotation()
            return process_next_video()

        actions = sorted(actions)
        all_frames = set()
        for action in actions:
            frame_ids = list(action.boxes.keys())
            all_frames.update(frame_ids)
        all_frames = sorted(list(all_frames))

        for action in actions:
            #random_idx = random.choice(list(action.boxes.keys()))
            #for actor_id, bbox in zip(action.actors_time_dict.keys(), action.boxes[random_idx]):
            for actor_id in action.actors_time_dict.keys():
                random_box = random.choice(action.actors_time_dict[actor_id]['boxes'])
                random_idx = random_box.ts
                first_frame = all_frames.index(random_idx)
                actor_class = list(categories[actor_id].keys())[0]

                frame = Image.open(str(frame_files[first_frame]))
                box = (random_box.x0, random_box.y0, random_box.x1, random_box.y1)
                # draw = ImageDraw.Draw(frame)
                # draw.rectangle(box, outline=(255, 0, 0), width=5)

                class_dict[actor_class].append((frame, actor_id, random_idx, box))

        for k, v in class_dict.items():
            class_dict[k] = sorted(v, key=lambda x: x[2])
        # After loading new data, refresh the UI
        load_images(canvas, current_frame)
    except StopIteration:
        messagebox.showinfo("End of list", "No more videos to process")
        root.quit()


def get_updated_file_list():
    global dataset
    all_videos = dataset.file_names
    seen_videos = set()
    try:
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    key, value = line.split(':')
                    seen_videos.add(key)
                except ValueError as ve:
                    print(f'Error on get_updated_file_list: {ve}')
                    continue

                #_correspondence_dict[key] = value
    except FileNotFoundError as e:
        print(f'Error on get_updated_file_list: {e}')

    all_videos = natsorted([folder for folder in all_videos if folder not in seen_videos])
    return all_videos


def skip_video():
    correspondence_dict[filename] = ['skipped']
    write_annotation()
    process_next_video()


# Initialize main application window

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
scrollbar.pack(side="right", fill="y")
canvas.pack(side="left", fill="both", expand=True)

# This is the magic that makes the frame inside the canvas scrollable
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

scrollable_frame.bind("<Configure>", lambda event, canvas=canvas: on_frame_configure(canvas))

# Now, use `scrollable_frame` as your `current_frame` for adding images
current_frame = scrollable_frame

# Initialize a list to store past frame thumbnails
past_frames = []
# video_iterator = get_videos_iterator(all_videos)e = None  # Placeholder for the current image


# Create a label to display the current frame
current_frame_label = tk.Label(current_frame)
current_frame_label.pack()

# Add an entry widget for annotations
annotation_frame = tk.Frame(root)
annotation_label = tk.Label(annotation_frame, text="Enter the unique ids of the objects to remove (comma separated)")
annotation_entry = tk.Entry(annotation_frame)
submit_button = tk.Button(annotation_frame, text="Submit Annotation", command=submit_annotation)
root.bind('<Return>', lambda e: submit_annotation())  # Bind the return key to submit the annotation
annotation_frame.pack(side='bottom', fill='x', expand=False)  # Pack the annotation frame at the bottom
annotation_label.pack(side='top', fill='x', expand=False)  # Pack the label inside the annotation frame
annotation_entry.pack(side='top', fill='x', expand=False)  # Pack the entry below the label
submit_button.pack(side='top', fill='x', expand=False)  # Pack the submit button below the entry

root.bind('<Escape>', lambda e: skip_video())  # Bind the escape key to quit the application

### End of Initial UI Setup


# Some Global Variables
correspondence_dict = defaultdict(list)
filename = None
class_dict = defaultdict(list)
class_dict_kyle = defaultdict(list)

#frames_folder = Path('/home/raphael/Documents/skywalker_6/raphael/meva/frames')
frames_folder = Path(args.hoi_path) / 'frames'
dataset = build(image_set='train', args=args)
frame_files = []




# Get started
process_next_video()
root.mainloop()

