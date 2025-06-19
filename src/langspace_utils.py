# Code from the Taskography API project
# https://taskography.github.io/

from typing import Tuple, Dict, DefaultDict

import numpy as np
import networkx as nx
from collections import defaultdict

from typing import Dict, Tuple

import json
import os.path as osp

#from pddlgym.parser import PDDLDomainParser



# general constants
DOMAIN_ALIAS = {
    "taskographyv1": "flat_rearrangement",
    "taskographyv2": "rearrangement",
    "taskographyv3": "courier",
    "taskographyv4": "lifted_rearrangement",
    "taskographyv5": "lifted_courier",
}


DOMAIN_BAGSLOTS = {
    "flat_rearrangement": False,
    "rearrangement": False,
    "courier": True,
    "lifted_rearrangement": False,
    "lifted_courier": True,
}


OFFICIAL_SPLITS = {"tiny": "tiny/verified_graph", "medium": "medium/automated_graph"}


SPLIT_SCENES = {
    "tiny": 35,
    "tiny/verified_graph": 35,
    "medium": 105,
    "medium/automated_graph": 105,
}


# scene entities
ROOMS = {
    "bathroom",
    "bedroom",
    "childs_room",
    "closet",
    "corridor",
    "dining_room",
    "empty_room",
    "exercise_room",
    "garage",
    "home_office",
    "kitchen",
    "living_room",
    "lobby",
    "pantry_room",
    "playroom",
    "staircase",
    "storage_room",
    "television_room",
    "utility_room",
}


OBJECTS = {
    "apple",
    "backpack",
    "banana",
    "baseball bat",
    "baseball glove",
    "book",
    "bottle",
    "bowl",
    "cake",
    "cell phone",
    "clock",
    "cup",
    "frisbee",
    "handbag",
    "keyboard",
    "kite",
    "knife",
    "laptop",
    "mouse",
    "orange",
    "potted plant",
    "remote",
    "spoon",
    "sports ball",
    "suitcase",
    "teddy bear",
    "tie",
    "toothbrush",
    "umbrella",
    "vase",
    "wine glass",
    "bicycle",
    "motorcycle",
    "surfboard",
    "tv",
}

SMALL_OBJECTS = {
    "apple",
    "banana",
    "baseball glove",
    "book",
    "bottle",
    "bowl",
    "cell phone",
    "cup",
    "knife",
    "mouse",
    "orange",
    "remote",
    "spoon",
    "tie",
    "toothbrush",
    "wine glass",
}

MEDIUM_OBJECTS = {"cake", "clock", "frisbee", "laptop", "teddy bear", "vase"}

LARGE_OBJECTS = {
    "backpack",
    "baseball bat",
    "handbag",
    "keyboard",
    "kite",
    "potted plant",
    "sports ball",
    "suitcase",
    "umbrella",
    "bicycle",
    "motorcycle",
    "surfboard",
    "tv",
}

assert len(SMALL_OBJECTS.union(MEDIUM_OBJECTS.union(LARGE_OBJECTS))) == len(OBJECTS)


# objects that can be placed in HEATING receptacle type
HEATABLE_OBJECTS = {"apple", "banana", "bottle", "bowl", "cake", "cup", "orange"}

# objects that can be placed in COOLING receptacle type
COOLABLE_OBJECTS = {
    "apple",
    "banana",
    "bottle",
    "bowl",
    "cake",
    "cup",
    "orange",
    "wine glass",
}

# objects that can be placed in CLEANING receptacle type
CLEANABLE_OBJECTS = {
    "apple",
    "banana",
    "bottle",
    "bowl",
    "cup",
    "frisbee",
    "knife",
    "orange",
    "spoon",
    "sports ball",
    "toothbrush",
    "vase",
    "wine glass",
}

# receptacle objects can store one or more non-receptacle object
RECEPTACLE_OBJECTS = {
    "backpack",
    "baseball glove",
    "bottle",
    "bowl",
    "handbag",
    "suitcase",
    "vase",
    "wine glass",
}

# non-receptacle objects cannot store other objects
NON_RECEPTACLE_OBJECTS = OBJECTS - RECEPTACLE_OBJECTS


RECEPTACLES = {
    "bed",
    "bench",
    "boat",
    "chair",
    "couch",
    "dining table",
    "microwave",
    "oven",
    "refrigerator",
    "sink",
    "toaster",
    "toilet",
}

# receptacle types (for generating receptacle receptacle_type facts)
OPENING_RECEPTACLES = {"microwave", "oven", "refrigerator"}

HEATING_RECEPTACLES = {"microwave", "oven", "toaster"}

COOLING_RECEPTACLES = {"refrigerator"}

CLEANING_RECEPTACLES = {"sink"}

# TODO: leave out toggle for now, macro-actions assume toggle on the receptacles
# TOGGLEABLE_RECEPTACLES = {
#     'microwave',
#     'oven',
#     'sink',
#     'toaster',
#     'toilet'
# }


MATERIALS = {
    "ceramic",
    "fabric",
    "foliage",
    "food",
    "glass",
    "leather",
    "metal",
    "mirror",
    "other",
    "oven",
    "paper",
    "plastic",
    "polished stone",
    "stone",
    "wood",
}

TEXTURES = {
    "visual": {
        "blotchy",
        "chequered",
        "crosshatched",
        "dotted",
        "grid",
        "interlaced",
        "lined",
        "marbled",
        "paisley",
        "polka-dotted",
        "smeared",
        "stained",
        "striped",
        "swirly",
        "zigzagged",
    },
    "tactile": {
        "braided",
        "bumpy",
        "crystalline",
        "fibrous",
        "frilly",
        "gauzy",
        "grooved",
        "knitted",
        "matted",
        "meshed",
        "perforated",
        "pleated",
        "potholed",
        "scaly",
        "spiralled",
        "stratified",
        "studded",
        "veined",
        "waffled",
        "woven",
        "wrinkled",
    },
}


# PDDLGym Taskography benchmark domain names
BENCHMARK_DOMAINS = [
    ## Full Domains
    # Rearrangment(k)
    "taskographyv2tiny1",
    "taskographyv2medium1",
    "taskographyv2tiny2",
    "taskographyv2medium2",
    "taskographyv2tiny5",
    "taskographyv2medium5",
    "taskographyv2tiny10",
    "taskographyv2medium10",
    # Courier(n, k)
    "taskographyv3tiny5bagslots5",
    "taskographyv3medium5bagslots5",
    "taskographyv3tiny10bagslots3",
    "taskographyv3medium10bagslots3",
    "taskographyv3tiny10bagslots5",
    "taskographyv3medium10bagslots5",
    "taskographyv3tiny10bagslots7",
    "taskographyv3medium10bagslots7",
    "taskographyv3tiny10bagslots10",
    "taskographyv3medium10bagslots10",
    # Lifted Rearrangement(k)
    "taskographyv4tiny5",
    "taskographyv4medium5",
    # Lifted Courier(n, k)
    "taskographyv5tiny5bagslots5",
    "taskographyv5medium5bagslots5",
    ## Scrubbed Domains
    # Rearrangement(k)
    "taskographyv2tiny1scrub",
    "taskographyv2medium1scrub",
    "taskographyv2tiny2scrub",
    "taskographyv2medium2scrub",
    "taskographyv2tiny10scrub",
    "taskographyv2medium10scrub",
    # Courier(n, k)
    "taskographyv3tiny10bagslots10scrub",
    "taskographyv3medium10bagslots10scrub",
    "taskographyv3tiny10bagslots3scrub",
    "taskographyv3medium10bagslots3scrub",
    "taskographyv3tiny10bagslots5scrub",
    "taskographyv3medium10bagslots5scrub",
    "taskographyv3tiny10bagslots7scrub",
    "taskographyv3medium10bagslots7scrub",
    # Lifted Rearrangement(k)
    "taskographyv4tiny5scrub",
    "taskographyv4medium5scrub",
    # Lifted Courier(n, k)
    "taskographyv5tiny5bagslots5scrub",
    "taskographyv5medium5bagslots5scrub",
]




class SceneGraphNode(object):
    def __init__(self):
        pass

    def set_attribute(self, attr, value):
        if attr not in self.__dict__.keys():
            raise ValueError(f"Unknown attribute: {attr}")
        self.__dict__[attr] = value

    def get_attribute(self, attr):
        if attr not in self.__dict__.keys():
            raise ValueError(f"Unknown attribute: {attr}")
        return self.__dict__[attr]


class Building(SceneGraphNode):
    def __init__(self):
        # 2D floor area (sq. meters)
        self.floor_area = None
        # Functionality of the building
        self.function = None
        # Gibson split (tiny, medium, large)
        self.gibson_split = None
        # Unique building id
        self.id = None
        # Name of the Gibson model
        self.name = None
        # Number of panoramic cameras in the model
        self.num_cameras = None
        # Number of floors in the building
        self.num_floors = None
        # Number of objects in the building
        self.num_objects = None
        # Number of rooms in the building
        self.num_rooms = None
        # Building reference point
        self.reference_point = None
        # 3D size of building
        self.size = None
        # 3D volume of building (in cubic meters, computed from the 3D convex hull)
        self.volume = None
        # Size of each voxel
        self.voxel_size = None
        # 3D coordinates of voxel centers (N x 3)
        self.voxel_centers = None
        # Number of voxels per axis (k x l x m)
        self.voxel_resolution = None

        # Instantiate other layers in the graph
        self.room = {}
        self.camera = {}
        self.object = {}

    def print_attributes(self):
        print(f"--- Building ID: {self.id} ---")
        for key in self.__dict__.keys():
            if key not in ["room", "camera", "object", "voxel_centers"]:
                print(f"Key: {key} | Value: {self.get_attribute(key)}")


class Room(SceneGraphNode):
    def __init__(self):
        # 2D floor area (in square meters)
        self.floor_area = None
        # Index of the floor that contains this room
        self.floor_number = None
        # Unique space id per building
        self.id = None
        # 3D coordinates of room center
        self.location = None
        # Building face indices that correspond to this room
        self.inst_segmentation = None
        # Functionality of the room
        self.scene_category = None
        # 3D size of the room
        self.size = None
        # Building's voxel indices tha correspond to this space
        self.voxel_occupancy = None
        # 3D volume of the room (in cubic meters, computed from the 3D convex hull)
        self.volume = None
        # Parent building that contains this room
        self.parent_building = None
        # Connected Rooms
        self.connected_rooms = set()

    def print_attributes(self):
        print(f"--- Room ID: {self.id} ---")
        for key in self.__dict__.keys():
            print(f"Key: {key} | Value: {self.get_attribute(key)}")


class SceneObject(SceneGraphNode):
    def __init__(self):
        # List of possible actions
        self.action_affordance = None
        # 2D floor area (in square meters)
        self.floor_area = None
        # Total surface coverage (in square meters)
        self.surface_coverage = None
        # Object label
        self.class_ = None
        # Unique object id per building
        self.id = None
        # 3D coordinates of object center
        self.location = None
        # List of main object materials
        self.material = None
        # 3D object size
        self.size = None
        # Building face indices that correspond to this object
        self.inst_segmentation = None
        # Main tactile texture (may be None)
        self.tactile_texture = None
        # Main visible texture (may be None)
        self.visual_texture = None
        # 3D volume of object (in cubic meters, computed from the 3D convex hull)
        self.volume = None
        # Building voxel indices corresponding to this object
        self.voxel_occupancy = None
        # Parent room that contains this object
        self.parent_room = None

    def print_attributes(self):
        print(f"--- Object ID: {self.id} ---")
        for key in self.__dict__.keys():
            print(f"Key: {key} | Value: {self.get_attribute(key)}")


class Camera(SceneGraphNode):
    def __init__(self):
        # Name of the camera
        self.name = None
        # Unique camera id
        self.id = None
        # Camera field of view
        self.FOV = None
        # 3D location of camera in the model
        self.location = None
        # 3D orientation of camera (quaternion)
        self.rotation = None
        # Camera modality (e.g., RGB, grayscale, depth, etc.)
        self.modality = None
        # Camera resolution
        self.resolution = None
        # Parent room that contains this camera
        self.parent_room = None

def loader(path: str) -> Building:
    """Load a 3D scene graph.

    args:
        path: path to an iGibson scene graph pickle file.

    returns:
        building: 3D scene graph building
    """
    data = np.load(path, allow_pickle=True)["output"].item()
    building = Building()

    # Set building attributes
    for key in data["building"].keys():
        if key in [
            "object_inst_segmentation",
            "room_inst_segmentation",
            "object_voxel_occupancy",
            "room_voxel_occupancy",
        ]:
            continue
        building.set_attribute(key, data["building"][key])
    res = building.voxel_resolution
    voxel_centers = np.reshape(building.voxel_centers, (res[0], res[1], res[2], 3))
    building.set_attribute("voxel_centers", voxel_centers)

    # Set room attributes
    unique_rooms = np.unique(data["building"]["room_inst_segmentation"])
    for room_id in unique_rooms:
        if room_id == 0:
            continue
        building.room[room_id] = Room()
        room_faces = np.where(data["building"]["room_inst_segmentation"] == room_id)[0]
        building.room[room_id].set_attribute("inst_segmentation", room_faces)
        room_voxels = np.where(data["building"]["room_voxel_occupancy"] == room_id)[0]
        building.room[room_id].set_attribute("voxel_occupancy", room_voxels)
        for key in data["room"][room_id].keys():
            building.room[room_id].set_attribute(key, data["room"][room_id][key])

    # Set object attributes
    unique_objects = np.unique(data["building"]["object_inst_segmentation"])
    for object_id in unique_objects:
        if object_id == 0:
            continue
        building.object[object_id] = SceneObject()
        object_faces = np.where(data["building"]["object_inst_segmentation"] == object_id)[0]
        building.object[object_id].set_attribute("inst_segmentation", object_faces)
        object_voxels = np.where(data["building"]["object_voxel_occupancy"] == object_id)[0]
        building.object[object_id].set_attribute("voxel_occupancy", object_voxels)
        for key in data["object"][object_id].keys():
            building.object[object_id].set_attribute(key, data["object"][object_id][key])

    # Set camera attributes
    for cam_id in data["camera"].keys():
        if cam_id == 0:
            continue
        building.camera[cam_id] = Camera()
        for key in data["camera"][cam_id].keys():
            building.camera[cam_id].set_attribute(key, data["camera"][cam_id][key])

    scenegraph_mst(building)
    return building


def scenegraph_mst(building: Building) -> None:
    """Apply Kruskal's algorithm to find the minimum spanning tree of room connectivities.
    Edge weights are determined by the distance between rooms' centroids. Heuristics are
    used to determine floor adjacency such that only a single connection exists between floors.

    args:
        building: a loaded 3D scene graph building
    """
    room_ids, room_loc, floor_rooms = index_building(building)

    # sanity check on scene graph pickle data
    if building.num_rooms is None:
        building.num_rooms = len(building.room)
    assert len(building.room) == building.num_rooms
    num_floors_with_rooms = len(floor_rooms)

    # room-room distance matrix
    room_loc_np = np.array(list(room_loc.values()))  # n x 3
    room_loc_np_exp = np.expand_dims(room_loc_np.copy(), axis=2)  # n x 3 x 1
    room_dist_mat = np.linalg.norm(
        (room_loc_np_exp.transpose(1, 0, 2) - room_loc_np_exp.transpose(1, 2, 0)),
        axis=0,
    )

    # compute minumal spanning tree of rooms
    room_graph = nx.Graph()
    if num_floors_with_rooms > 1:

        # compute minimal spanning tree of floors
        floor_graph = nx.Graph()
        floor_adj_data = dict()
        for floor_a, floor_a_rooms in floor_rooms.items():
            for floor_b, floor_b_rooms in floor_rooms.items():
                if (
                    floor_a == floor_b
                    or (floor_a, floor_b) in floor_adj_data
                    or (floor_b, floor_a) in floor_adj_data
                ):
                    continue
                floor_a_rooms = list(floor_a_rooms)
                floor_b_rooms = list(floor_b_rooms)

                # floor-floor heuristic: mean of min connection between rooms in both floors
                n, m = len(floor_a_rooms), len(floor_b_rooms)
                floor_a_rooms_repeat = np.repeat(np.array(floor_a_rooms, dtype=int), m)
                floor_b_rooms_tile = np.tile(np.array(floor_b_rooms, dtype=int), n)
                room_a_to_b_dist = room_dist_mat[floor_a_rooms_repeat, floor_b_rooms_tile].reshape(
                    n, m
                )
                floor_dist_heuristic = np.amin(room_a_to_b_dist, axis=0).mean()
                floor_graph.add_edge(floor_a, floor_b, weight=floor_dist_heuristic)

                # store minimum connection between floors
                room_a_tidx, room_b_tidx = np.unravel_index(
                    np.argmin(room_a_to_b_dist), shape=room_a_to_b_dist.shape
                )
                data = {
                    "min_rooms": [
                        floor_a_rooms[room_a_tidx],
                        floor_b_rooms[room_b_tidx],
                    ],
                    "min_dist": np.amin(room_a_to_b_dist),
                }
                floor_adj_data[(floor_a, floor_b)] = data
                floor_adj_data[(floor_b, floor_a)] = data

        floor_mst = nx.minimum_spanning_tree(floor_graph)
        assert floor_mst.order() == num_floors_with_rooms

        # add edge between closest rooms connecting floors
        for floor_a, floor_b in floor_mst.edges():
            data = floor_adj_data[(floor_a, floor_b)]
            room_graph.add_edge(*data["min_rooms"], weight=data["min_dist"])

    # connect all rooms in each floor
    for _, rooms in floor_rooms.items():
        room_idx_repeat = np.repeat(np.array(list(rooms), dtype=int), len(rooms))
        room_idx_tile = np.tile(np.array(list(rooms), dtype=int), len(rooms))
        room_dist = room_dist_mat[room_idx_repeat, room_idx_tile]
        room_graph.add_weighted_edges_from(list(zip(room_idx_repeat, room_idx_tile, room_dist)))

    # add room adjacency list to scene graph
    room_mst = nx.minimum_spanning_tree(room_graph)
    assert nx.number_connected_components(room_mst) == 1, "Minimum spanning tree is not complete"
    assert building.num_rooms == room_mst.order(), "Missing rooms in computed minimum spanning tree"
    assert (
        building.num_rooms - 1 == room_mst.size()
    ), "Missing edges in the computed minimum spanning tree"
    for room_a_idx, room_b_idx in room_mst.edges():
        building.room[room_ids[room_a_idx]].connected_rooms.add(room_ids[room_b_idx])
        building.room[room_ids[room_b_idx]].connected_rooms.add(room_ids[room_a_idx])


def index_building(building: Building) -> Tuple[Dict, Dict, DefaultDict]:
    """Index rooms and floors in the building."""
    room_ids = dict()  # dict(key=room_idx, value=room_id)
    room_loc = dict()  # dict(key=room_idx, value=room_location)
    room_floor = dict()  # dict(key=room_idx, value=floor_idx)
    floor_idx = dict()  # dict(key=floor_id, value=floor_idx)
    floor_ids = dict()  # dict(key=floor_idx, value=floor_id)
    floor_rooms = defaultdict(set)  # dict(key=floor_idx, value=set(room_idx))

    count = 0
    for idx, id in enumerate(building.room):
        room_ids[idx] = id
        room_loc[idx] = building.room[id].location
        f_id = building.room[id].floor_number
        if f_id not in floor_idx:
            floor_idx[f_id] = count
            floor_ids[count] = f_id
            count += 1
        room_floor[idx] = floor_idx[f_id]

    for r_idx, f_idx in room_floor.items():
        floor_rooms[f_idx].add(r_idx)

    return room_ids, room_loc, floor_rooms


REQUIRED_BASE_KEYS = [
    "domain_type",
    "split",
    "bagslots",
    "complexity",
    "train_scenes",
    "samples_per_train_scene",
    "samples_per_test_scene",
    "seed",
]


_KEY_MAP = {
    "domain_type": "",
    "split": "split",
    "bagslots": "n",
    "complexity": "k",
    "train_scenes": "trsc",
    "samples_per_train_scene": "trsa",
    "samples_per_test_scene": "tesa",
    "seed": "seed",
}


_KEY_MAP_INV = {v: k for k, v in _KEY_MAP.items()}


def room_to_str_name(room_inst: Room) -> str:
    """Construct room name."""
    return f"room{int(room_inst.id)}_{room_inst.scene_category.replace(' ', '_')}"


def place_to_str_name(
    place_id: int, inst: SceneGraphNode, is_object: bool = False, is_room: bool = False
) -> str:
    """Construct place name."""
    assert not (is_object and is_room)
    if is_room:
        return (
            f"place{int(place_id)}_door_room{int(inst.id)}_{inst.scene_category.replace(' ', '_')}"
        )
    elif is_object:
        return f"place{int(place_id)}_item{int(inst.id)}_{inst.class_.replace(' ', '_')}"
    return f"place{int(place_id)}_receptacle{int(inst.id)}_{inst.class_.replace(' ', '_')}"


def receptacle_to_str_name(rec_inst: SceneObject) -> str:
    """Construct receptacle name."""
    return f"receptacle{int(rec_inst.id)}_{rec_inst.class_.replace(' ', '_')}"


def object_to_str_name(obj_inst: SceneObject, size: int) -> str:
    """Construct object name."""
    return f"item{int(obj_inst.id)}_{obj_inst.class_.replace(' ', '_')}_{size}"


def location_to_str_name(room_data: Tuple, place_id: int) -> str:
    """Construct location name."""
    (cx, cy), room_id, floor_num = room_data
    cx = f"neg{-cx}" if cx < 0 else f"pos{cx}"
    cy = f"neg{-cy}" if cy < 0 else f"pos{cy}"
    return f"location_X{cx}_Y{cy}_place{place_id}_room{int(room_id)}_floor{floor_num}"


#def write_domain_file(
#    pddlgym_domain: PDDLDomainParser, domain_filepath: str, domain_name: str = None
#) -> None:
#    """Write out PDDL domain file while scanning for and removing the
#    untyped equality (= ?v0 ?v1) written by PDDLGymDomainParser.

#    args:
#        pddlgym_domain: PDDLGymDomainParser object
#        domain_filepath: path to write PDDL domain file
#        domain_name: relabel the domain with this name (default: None)
#    """
#    if domain_name is not None:
#        pddlgym_domain.domain_name = domain_name
#    pddlgym_domain.write(domain_filepath)

#    with open(domain_filepath, "rt") as fh:
#        lines = fh.readlines()
#        lines = [l for l in lines if l.strip("\n").strip() != "(= ?v0 ?v1)"]
#    with open(domain_filepath, "wt") as fh:
#        fh.writelines(lines)


def register_pddlgym_domain(problem_dir: str, domain_name: str) -> None:
    """Add new domain to the list of environments to be registered by PDDLGym.

    args:
        problem_dir: path to pddlgym/pddlgym/pddl directory
        domain_name: name of the domain
    """
    register_filepath = osp.realpath(osp.join(osp.dirname(problem_dir), "__init__.py"))
    with open(register_filepath, "rt") as fh:
        lines = fh.readlines()

    idx = -1
    for i, line in enumerate(lines):
        if line.strip("\n").strip() == "]:":
            idx = i
            break
    assert idx != -1, "Could not find appropriate location to insert domain declaration"
    decl_str = "\t\t(\n"
    decl_str += f'\t\t\t"{domain_name}",\n'
    decl_str += "\t\t\t{\n"
    decl_str += '\t\t\t\t"operators_as_actions": True,\n'
    decl_str += '\t\t\t\t"dynamic_action_space": True\n'
    decl_str += "\t\t\t}\n"
    decl_str += "\t\t)\n"
    lines.insert(idx, decl_str)

    # extend list
    prev = lines[idx - 1].strip("\n")
    assert prev[-1] in [",", ")"]
    if prev[-1] != ",":
        prev += ","
    prev += "\n"
    lines[idx - 1] = prev

    with open(register_filepath, "wt") as fh:
        fh.writelines(lines)


def config_to_domain_name(**kwargs) -> str:
    """Return conventional domain name from REQUIRED_BASE_KEYS in the config."""
    # Ensure base keys provided
    for k in REQUIRED_BASE_KEYS:
        assert k in kwargs, f"Missing keyword argument {k} required to name the domain."

    # Construct name
    domain_name = kwargs["domain_type"]
    for k in REQUIRED_BASE_KEYS[1:]:
        k, v = _KEY_MAP[k], kwargs[k]
        if k == "n" and v == None:
            v = 0
        domain_name += f"_{k}_{v}"

    return domain_name.lower()


def domain_to_pddlgym_name(domain_name: str, test: bool = False) -> str:
    """Return domain name as registered by PDDLGym."""
    pddlgym_name = domain_name.capitalize()
    if test:
        pddlgym_name += "Test"
    return f"PDDLEnv{pddlgym_name}-v0"


def domain_name_to_config(domain_name: str) -> Dict:
    """Construct config with REQUIRED_BASE_KEYS from a domain name."""
    params = domain_name.lower().split("_")
    split_idx = params.index("split")

    # Domain type and split
    config = dict()
    config["domain_type"] = "_".join([p for p in params[:split_idx]])
    config["split"] = params[split_idx + 1]

    # Remaining keys
    prev_key = None
    for p in params[split_idx + 2 :]:
        if p in _KEY_MAP_INV:
            prev_key = _KEY_MAP_INV[p]
        else:
            config[prev_key] = int(p)

    assert all(k in config for k in REQUIRED_BASE_KEYS)
    return config


def scene_graph_name(scene_graph_filepath: str) -> str:
    """Retrieve scene graph model."""
    return scene_graph_filepath.split(".")[0].split("_")[-1].lower()


def sampler_name(scene_graph_filepath: str, complexity: int, bagslots: int = None):
    """Construct sampler name."""
    sampler_name = scene_graph_name(scene_graph_filepath)
    bagslots = 0 if bagslots is None else bagslots
    sampler_name += f"_n{bagslots}_k{complexity}"
    return sampler_name


def load_json(filepath: str) -> Dict:
    data = None
    with open(filepath, "r") as fp:
        data = json.load(fp)
    return data


def save_json(filepath: str, data: Dict) -> None:
    with open(filepath, "w") as fp:
        json.dump(data, fp, indent=4, sort_keys=True)
