from PIL import Image
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import random

class LogReg():
    def __init__(self):
        self.class_map = []
        self.DATA = pd.DataFrame()
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.classifier = None

    def generate_dataframe(self):
        self.DATA = self.gen_dataframe()

    def read_img(self, img_path):
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = img.resize((32, 32), Image.NEAREST)
        img_array = np.array(img).flatten()
        return img_array
    
    def overlay_images(self, background, foreground, position=(0,0)):
        bg = background.convert('RGBA')
        fg = foreground.convert('RGBA')

        combined = Image.new('RGBA', bg.size)
        combined.paste(bg, (0,0))
        combined.paste(fg, position, fg)
        return combined
    
    def overlay_triangle_on_image(self, background, tri):
        bg = background.convert('RGBA')

        combined = Image.new('RGBA', bg.size)
        combined.paste(bg, (0,0))
        combined.paste(tri, (0, 0), tri)
        return combined
    
    def combine_entity_on_floor(self, entity_img_path, floor_img_path, position=(0,0)):
        background = Image.open(floor_img_path).resize((32, 32), Image.NEAREST)
        foreground = Image.open(entity_img_path).resize((32, 32), Image.NEAREST)
        combined = self.overlay_images(background, foreground, position)
        combined = combined.convert('RGB')
        combined = combined.resize((32, 32), Image.NEAREST)
        arr = np.array(combined).flatten()
        return arr
    
    def combine_triangle_on_image(self, entity_img_path, floor_img_path, triangle_img_path):
        background = Image.open(floor_img_path).resize((32, 32), Image.NEAREST)
        foreground = Image.open(entity_img_path).resize((32, 32), Image.NEAREST)
        triangle = Image.open(triangle_img_path)
        combined2 = self.overlay_images(background, foreground, (0,0))
        combined2 = combined2.convert('RGBA')
        combined2 = combined2.resize((32, 32), Image.NEAREST)
        combined = self.overlay_triangle_on_image(combined2, triangle)
        combined = combined.convert('RGB')
        arr = np.array(combined).flatten()
        return arr

    def gen_dataframe(self):
        class_num = 0
        data = []
        labels = []
        base_path = "data/assets/imagen1/"
        floor_path = os.path.join(base_path, "floor")
        arrow_path = os.path.join(base_path, "arrows")
        classes = os.listdir(base_path)
        for cls in classes:
            cls_path = os.path.join(base_path, cls)
            if os.path.isdir(cls_path):
                if cls == "floor" or cls == "wall" or cls == "lava":
                    self.class_map.append(cls)
                    images = os.listdir(cls_path)
                    for img_name in images:
                        img_path = os.path.join(cls_path, img_name)
                        img_array = self.read_img(img_path)
                        data.append(img_array)
                        labels.append(class_num)
                    class_num += 1
                elif cls == "dragon" or cls == "wolf" or cls == "opened" or cls == 'arrows':
                    continue
                elif cls == "metalbox":
                    self.class_map.append(cls)
                    images = os.listdir(cls_path)
                    floors = os.listdir(floor_path)
                    directions = os.listdir(arrow_path)
                    for dir in directions:
                        directions_path_full = os.path.join(arrow_path, dir)
                        arrow_images = os.listdir(directions_path_full)
                        for arrow_name in arrow_images:
                            arrow_img_path = os.path.join(directions_path_full, arrow_name)
                            for img_name in images:
                                for floor_name in floors:
                                    floor_img_path =  os.path.join(floor_path, floor_name)
                                    entity_img_path = os.path.join(cls_path, img_name)
                                    img_array = self.combine_triangle_on_image(entity_img_path, floor_img_path, arrow_img_path)
                                    data.append(img_array)
                                    labels.append(class_num)
                        class_num += 1
                elif cls == "robot":
                    self.class_map.append(cls)
                    images = os.listdir(cls_path)
                    floors = os.listdir(floor_path)
                    directions = os.listdir(arrow_path)
                    for dir in directions:
                        directions_path_full = os.path.join(arrow_path, dir)
                        arrow_images = os.listdir(directions_path_full)
                        for arrow_name in arrow_images:
                            arrow_img_path = os.path.join(directions_path_full, arrow_name)
                            for img_name in images:
                                for floor_name in floors:
                                    floor_img_path =  os.path.join(floor_path, floor_name)
                                    entity_img_path = os.path.join(cls_path, img_name)
                                    img_array = self.combine_triangle_on_image(entity_img_path, floor_img_path, arrow_img_path)
                                    data.append(img_array)
                                    labels.append(class_num)
                        class_num += 1
                else:
                    self.class_map.append(cls)
                    images = os.listdir(cls_path)
                    floors = os.listdir(floor_path)
                    for img_name in images:
                        for floor_name in floors:
                            floor_img_path =  os.path.join(floor_path, floor_name)
                            entity_img_path = os.path.join(cls_path, img_name)
                            img_array = self.combine_entity_on_floor(entity_img_path, floor_img_path)
                            data.append(img_array)
                            labels.append(class_num)
                    class_num += 1

        df = pd.DataFrame(data, dtype=np.uint8)
        df['label'] = labels
        return df
    
    def split_train_test(self, test_size=0.2):
        if self.DATA.empty:
            return
        if test_size == 0.0:
            self.X_train = self.DATA.drop(columns=['label'])
            self.y_train = self.DATA['label']
            self.X_test = self.X_train.copy()
            self.y_test = self.y_train.copy()
            return
        train_data, test_data = train_test_split(self.DATA, test_size=test_size, random_state=42)
        self.X_train = train_data.drop(columns=['label'])
        self.y_train = train_data['label']
        self.X_test = test_data.drop(columns=['label'])
        self.y_test = test_data['label']


    def train_model(self):
        model = LogisticRegression(max_iter=10000)
        self.classifier = OneVsRestClassifier(model)
        self.classifier.fit(self.X_train, self.y_train)
        return self.classifier


logreg = LogReg()
logreg.generate_dataframe()
logreg.split_train_test(test_size=0.0)
classifier = logreg.train_model()

from utils import generate_sklearn_loader_snippet

binary_classifiers = classifier.estimators_
n_classes = len(binary_classifiers)

for i, clf in enumerate(binary_classifiers):
    sk_snippet = generate_sklearn_loader_snippet(clf, compression="zlib")
    print(f"# Binary classifier for class {i}")
    print(sk_snippet)


# Generate box images with arrows of different sizes
# Manually extracted arrows from box using pixel art editor
def build_box_level():
    level = Level(width=5, height=5, move_fn=default_move_fn, objective_fn=exit_objective_fn, seed=123)
    hero = create_agent(health=5)

    for y in range(level.height):
        for x in range(level.width):
            level.add((x, y), create_floor(cost_amount=3))

    level.add((0, 0), create_box(pushable=False, moving_axis=MovingAxis.HORIZONTAL, moving_direction=1))
    level.add((1, 0), hero)   
    return level

def build_box_state(seed=None):
    level = build_box_level()
    return to_state(level)

env = create_env(build_box_state, observation_type='image')
state, _ = env.reset()
image = Image.fromarray(state['image'])
img_array = state['image'][0:128, 0:128, : ]
img = Image.fromarray(img_array)
img = img.resize((32, 32), Image.NEAREST)

display(img)
def build_box_level_2():
    level = Level(width=6, height=6, move_fn=default_move_fn, objective_fn=exit_objective_fn, seed=123)
    hero = create_agent(health=5)

    for y in range(level.height):
        for x in range(level.width):
            level.add((x, y), create_floor(cost_amount=3))

    level.add((0, 0), create_box(pushable=False, moving_axis=MovingAxis.HORIZONTAL, moving_direction=1))
    level.add((1, 0), hero)   
    return level

def build_box_state_2(seed=None):
    level = build_box_level_2()
    return to_state(level)

env = create_env(build_box_state_2, observation_type='image')
state, _ = env.reset()
image = Image.fromarray(state['image'])
img_array = state['image'][0:106, 0:106, : ]
img = Image.fromarray(img_array)
img = img.resize((32, 32), Image.NEAREST)

display(img)

def build_box_level_3():
    level = Level(width=7, height=7, move_fn=default_move_fn, objective_fn=exit_objective_fn, seed=123)
    hero = create_agent(health=5)

    for y in range(level.height):
        for x in range(level.width):
            level.add((x, y), create_floor(cost_amount=3))

    level.add((0, 0), create_box(pushable=False, moving_axis=MovingAxis.HORIZONTAL, moving_direction=1))
    level.add((1, 0), hero)   
    return level

def build_box_state_3(seed=None):
    level = build_box_level_3()
    return to_state(level)

env = create_env(build_box_state_3, observation_type='image')
state, _ = env.reset()
image = Image.fromarray(state['image'])
img_array = state['image'][0:91, 0:91, : ]
img = Image.fromarray(img_array)
img = img.resize((32, 32), Image.NEAREST)

display(img)

def build_box_level_4():
    level = Level(width=8, height=8, move_fn=default_move_fn, objective_fn=exit_objective_fn, seed=3)
    hero = create_agent(health=5)

    for y in range(level.height):
        for x in range(level.width):
            level.add((x, y), create_floor(cost_amount=3))

    level.add((0, 0), create_box(pushable=False, moving_axis=MovingAxis.HORIZONTAL, moving_direction=1))
    level.add((1, 0), hero)   
    return level

def build_box_state_4(seed=None):
    level = build_box_level_4()
    return to_state(level)

env = create_env(build_box_state_4, observation_type='image')
state, _ = env.reset()
image = Image.fromarray(state['image'])
img_array = state['image'][0:80, 0:80, : ]
img = Image.fromarray(img_array)
img = img.resize((32, 32), Image.NEAREST)

display(img)

def build_box_level_7():
    level = Level(width=9, height=9, move_fn=default_move_fn, objective_fn=exit_objective_fn, seed=123)
    hero = create_agent(health=5)

    for y in range(level.height):
        for x in range(level.width):
            level.add((x, y), create_floor(cost_amount=3))

    level.add((0, 0), create_box(pushable=False, moving_axis=MovingAxis.HORIZONTAL, moving_direction=1))
    level.add((1, 0), hero)   
    return level

def build_box_state_7(seed=None):
    level = build_box_level_7()
    return to_state(level)

env = create_env(build_box_state_7, observation_type='image')
state, _ = env.reset()
image = Image.fromarray(state['image'])
img_array = state['image'][0:71, 0:71, : ]
img = Image.fromarray(img_array)
img = img.resize((32, 32), Image.NEAREST)

display(img)


def build_box_level_8():
    level = Level(width=10, height=10, move_fn=default_move_fn, objective_fn=exit_objective_fn, seed=123)
    hero = create_agent(health=5)

    for y in range(level.height):
        for x in range(level.width):
            level.add((x, y), create_floor(cost_amount=3))

    level.add((0, 0), create_box(pushable=False, moving_axis=MovingAxis.HORIZONTAL, moving_direction=1))
    level.add((1, 0), hero)   
    return level

def build_box_state_8(seed=None):
    level = build_box_level_8()
    return to_state(level)

env = create_env(build_box_state_8, observation_type='image')
state, _ = env.reset()
image = Image.fromarray(state['image'])
img_array = state['image'][0:64, 0:64, : ]
img = Image.fromarray(img_array)
img = img.resize((32, 32), Image.NEAREST)

display(img)

# def build_box_level_5():
#     level = Level(width=11, height=11, move_fn=default_move_fn, objective_fn=exit_objective_fn, seed=123)
#     hero = create_agent(health=5)

#     for y in range(level.height):
#         for x in range(level.width):
#             level.add((x, y), create_floor(cost_amount=3))

#     level.add((0, 0), create_box(pushable=False, moving_axis=MovingAxis.HORIZONTAL, moving_direction=1))
#     level.add((1, 0), hero)   
#     return level

# def build_box_state_5(seed=None):
#     level = build_box_level_5()
#     return to_state(level)

# env = create_env(build_box_state_5, observation_type='image')
# state, _ = env.reset()
# image = Image.fromarray(state['image'])
# img_array = state['image'][0:57, 0:57, : ]
# img = Image.fromarray(img_array)
# img = img.resize((32, 32), Image.NEAREST)

# display(img)

def build_box_level_6():
    level = Level(width=13, height=13, move_fn=default_move_fn, objective_fn=exit_objective_fn, seed=123)
    hero = create_agent(health=5)

    for y in range(level.height):
        for x in range(level.width):
            level.add((x, y), create_floor(cost_amount=3))

    level.add((0, 0), create_box(pushable=False, moving_axis=MovingAxis.HORIZONTAL, moving_direction=1))
    level.add((1, 0), hero)   
    return level

def build_box_state_6(seed=None):
    level = build_box_level_6()
    return to_state(level)

env = create_env(build_box_state_6, observation_type='image')
state, _ = env.reset()
image = Image.fromarray(state['image'])
img_array = state['image'][0:49, 0:49, : ]
img = Image.fromarray(img_array)
img = img.resize((32, 32), Image.NEAREST)

display(img)

