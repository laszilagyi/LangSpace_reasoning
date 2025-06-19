import json
import random

from langspace_utils import *

class LSSceneGraph:
    def __init__(self, scene_graph: Building = None):
        """
        Initializes LSSceneGraph either from a Building instance (default)
        or by using LSSceneGraph.from_json(json_data).
        """
        self.nodes = {"room": [], "agent": [], "object": [], "asset": []}
        self.links = []
        self.scene_graph_subset = {"nodes": {"room": []}, "links": []}

        self.agent_id = "agent"
        self.color_seed = 42
        random.seed(self.color_seed)

        self.asset_classes = {"microwave", "oven", "refrigerator", "sink", "toaster", "toilet", "bed", "chair", "couch", "dining table"}
        self.opening_assets = {"microwave", "oven", "refrigerator"}
        self.switching_assets = {"sink", "toilet"}
        self.passive_assets = {"bed", "chair", "couch", "dining table"}

        if scene_graph:
            assert isinstance(scene_graph, Building)
            self.sg = scene_graph
            self.parse_rooms()
            self.parse_objects_and_assets()
            self.parse_agent()
            self.parse_room_connections()
            self.initialize_scene_graph_subset()

    @classmethod
    def from_json(cls, json_data):

        instance = cls()  # Create an empty instance
        instance.nodes = json_data["nodes"]
        instance.links = json_data["links"]

        # Reconstruct the scene_graph_subset
        instance.scene_graph_subset = {
            "nodes": {
                "room": instance.nodes["room"],
                "agent": instance.nodes["agent"],
                "object": [],
                "asset": []
            },
            "links": [link for link in instance.links if any(node["id"] == instance.agent_id for node in instance.nodes["agent"])]
        }

        return instance

    def parse_rooms(self):
        for room_id, room in self.sg.room.items():
            room_node = {
                "id": f"{room.scene_category}_{int(room_id)}",
                "class": room.scene_category,
            }
            self.nodes["room"].append(room_node)
            self.scene_graph_subset["nodes"]["room"].append(room_node)

    def parse_objects_and_assets(self):
        asset_containment = {}

        for obj_id, obj in self.sg.object.items():
            #print(f"Processing object: {obj.class_}, ID: {obj_id}, Parent Room: {obj.parent_room}")  # 🔍 Debugging
            
            parent_room = obj.parent_room
            obj_name = f"{obj.class_}_{int(obj_id)}"

            if parent_room:
                room_key = f"{self.sg.room[parent_room].scene_category}_{int(parent_room)}"
            else:
                room_key = None

            node_data = {
                "id": obj_name,
                "class": obj.class_,
                "room": room_key,
                "affordances": []
            }

            if obj.class_ in self.asset_classes:
                #print(f"Adding asset: {obj_name} to room {room_key}")  # 🔍 Debugging
                node_data["contains"] = []

                if obj.class_ in self.opening_assets:
                    node_data["state"] = "closed"
                    node_data["affordances"] = ["open", "close", "release"]
                elif obj.class_ in self.switching_assets:
                    node_data["state"] = "off"
                    node_data["affordances"] = ["turn_on", "turn_off", "release"]
                elif obj.class_ in self.passive_assets:
                    node_data["state"] = "free"
                    node_data["affordances"] = ["release"]
                
                asset_containment[obj_name] = node_data
                self.nodes["asset"].append(node_data)

            else:
                #print(f"Adding object: {obj_name} to room {room_key}")  # 🔍 Debugging
                node_data["affordances"] = ["pickup", "release"]
                if hasattr(obj, 'parent_receptacle') and obj.parent_receptacle:
                    container_id = f"{self.sg.object[obj.parent_receptacle].class_}_{int(obj.parent_receptacle)}"
                    node_data["state"] = f"inside_of({container_id})"
                    if container_id in asset_containment:
                        asset_containment[container_id]["contains"].append(node_data["id"])
                else:
                    node_data["state"] = "accessible"
                
                self.nodes["object"].append(node_data)

            if parent_room:
                self.links.append(f"{room_key}-{node_data['id']}")

    def parse_agent(self):
        start_room = next(iter(self.sg.room.keys()))
        agent_location = f"{self.sg.room[start_room].scene_category}_{int(start_room)}"
        self.nodes["agent"].append({
            "id": self.agent_id,
            "location": agent_location,
            "accessing": agent_location,
            "carries": [],
            "class": "agent"
        })
        self.links.append(f"{agent_location}-{self.agent_id}")

    def parse_room_connections(self):
        for room_id, room in self.sg.room.items():
            for connected_room in room.connected_rooms:
                room1 = f"{self.sg.room[room_id].scene_category}_{int(room_id)}"
                room2 = f"{self.sg.room[connected_room].scene_category}_{int(connected_room)}"
                self.links.append(f"{room1}-{room2}")
                self.scene_graph_subset["links"].append(f"{room1}-{room2}")

    def goto(self, room_id):

        current_location = self.nodes["agent"][0]["location"]

        if current_location == room_id:
            return "Valid step."

        if f"{current_location}-{room_id}" in self.links or f"{room_id}-{current_location}" in self.links:
            self.nodes["agent"][0]["location"] = room_id
            self.nodes["agent"][0]["accessing"] = room_id
            self.links.remove(f"{current_location}-{self.agent_id}")
            self.links.append(f"{room_id}-{self.agent_id}")
            return "Valid step."
        else:
            return f"Invalid step: {current_location} is not adjacent to {room_id}."

    def access(self, asset_id):

        current_room = self.nodes["agent"][0]["location"]

        # Check if the asset exists in the current room
        asset_properties = next((asset for asset in self.nodes["asset"] if asset["id"] == asset_id and asset["room"] == current_room), None)

        if asset_properties:
            self.nodes["agent"][0]["accessing"] = asset_id
            return "Valid step."
        else:
            return f"Invalid step: {asset_id} is not in {current_room}."

    def abandon(self):
        self.nodes["agent"][0]["accessing"] = self.nodes["agent"][0]["location"]
        return 'Valid action.'

    def pickup(self, object_id):

        agent = self.nodes["agent"][0]
        location = agent["accessing"]

        # Find the object in the current room or accessed asset
        found_object = None
        for obj in self.nodes["object"]:
            if obj["id"] == object_id:
                # Check if the object is accessible
                if "room" in obj and obj["room"] == location:
                    found_object = obj
                    break
                elif "state" in obj and obj["state"] == f"inside_of({location})":
                    found_object = obj
                    break

        if found_object:
            # Move object to agent's carries list
            agent["carries"].append(object_id)
            self.links.append(f"{self.agent_id}-{object_id}")

            # Remove object from previous location
            if "room" in found_object:
                self.links.remove(f"{found_object['room']}-{object_id}")
                del found_object["room"]
            if "state" in found_object and "inside_of" in found_object["state"]:
                container_id = found_object["state"].split("(")[1][:-1]
                for asset in self.nodes["asset"]:
                    if asset["id"] == container_id and object_id in asset["contains"]:
                        asset["contains"].remove(object_id)
                found_object["state"] = "inside_of(agent)"
            else:
                found_object["state"] = "inside_of(agent)"

            return "Valid step."
        
        return f"Invalid step: {object_id} is not available at {location}."

    def release(self, object_id):

        agent = self.nodes["agent"][0]
        location = agent["accessing"]

        if object_id not in agent["carries"]:
            return f"Invalid step: Agent is not carrying {object_id}."

        # Check if the object is being released into an asset
        for asset in self.nodes["asset"]:
            if asset["id"] == location:
                if asset["class"] in self.opening_assets and asset["state"] == "closed":
                    return f"Invalid step: {object_id} cannot be released as {location} is closed."

                # Remove the object from the agent's carries list
                agent["carries"].remove(object_id)

                # Update the object's state and containment
                for obj in self.nodes["object"]:
                    if obj["id"] == object_id:
                        obj["state"] = f"inside_of({location})"
                        break

                # Add the object to the asset's contains field
                asset["contains"].append(object_id)

                # Update links
                self.links.remove(f"{self.agent_id}-{object_id}")
                self.links.append(f"{location}-{object_id}")

                return "Valid step."

        # Otherwise, release the object into the room
        agent["carries"].remove(object_id)

        for obj in self.nodes["object"]:
            if obj["id"] == object_id:
                obj["state"] = "accessible"
                obj["room"] = location
                break

        self.links.remove(f"{self.agent_id}-{object_id}")
        self.links.append(f"{location}-{object_id}")

        return "Valid step."

    def turn_on(self, asset_id):

        agent = self.nodes["agent"][0]
        location = agent["accessing"]

        if asset_id != location:
            return f"Invalid step: {asset_id} is not being accessed. Access the asset first."

        for asset in self.nodes["asset"]:
            if asset["id"] == asset_id:
                if asset["class"] in self.switching_assets:  # Check class, not ID
                    asset["state"] = "on"
                    return "Valid step."
                else:
                    return f"Invalid step: {asset_id} is not a switching asset."

        return f"Invalid step: {asset_id} not found."

    def turn_off(self, asset_id):

        agent = self.nodes["agent"][0]
        location = agent["accessing"]

        if asset_id != location:
            return f"Invalid step: {asset_id} is not being accessed. Access the asset first."

        for asset in self.nodes["asset"]:
            if asset["id"] == asset_id:
                if asset["class"] in self.switching_assets:  # Check class, not ID
                    asset["state"] = "off"
                    return "Valid step."
                else:
                    return f"Invalid step: {asset_id} is not a switching asset."

        return f"Invalid step: {asset_id} not found."

    def open(self, asset_id):

        agent = self.nodes["agent"][0]
        location = agent["accessing"]

        if asset_id != location:
            return f"Invalid step: {asset_id} is not being accessed. Access the asset first."

        for asset in self.nodes["asset"]:
            if asset["id"] == asset_id:
                if asset["class"] in self.opening_assets:
                    asset["state"] = "open"
                    return "Valid step."
                else:
                    return f"Invalid step: {asset_id} is not an opening asset."

        return f"Invalid step: {asset_id} not found."

    def close(self, asset_id):

        agent = self.nodes["agent"][0]
        location = agent["accessing"]

        if asset_id != location:
            return f"Invalid step: {asset_id} is not being accessed. Access the asset first."

        for asset in self.nodes["asset"]:
            if asset["id"] == asset_id:
                if asset["class"] in self.opening_assets:
                    asset["state"] = "closed"
                    return "Valid step."
                else:
                    return f"Invalid step: {asset_id} is not an opening asset."

        return f"Invalid step: {asset_id} not found."

    def execute_plan(self, plan):

        try:
            for action in plan:
                # Extract method name and parameter
                if "(" in action and ")" in action:
                    method_name, param = action.split("(")
                    param = param.rstrip(")")

                    # Check if method exists
                    if hasattr(self, method_name):
                        method = getattr(self, method_name)

                        # Execute method with parameter
                        result = method(param) if param else method()

                        # Stop execution immediately if an invalid step is encountered
                        if not result.startswith("Valid step."):
                            return f"Invalid plan. {result}"

                    else:
                        return f"Invalid plan. {action}"
                else:
                    return f"Invalid plan. Malformed action: {action}"

            return "Valid plan."

        except Exception as e:
            return f"Invalid plan. Error executing plan: {e}"

    def check_object_at(self, object_id, location_id):

        # Find the object in the scene graph
        for obj in self.nodes["object"]:
            if obj["id"] == object_id:
                # Check if the object is directly in the specified location
                if obj["state"] == f"inside_of({location_id})" or obj.get("room") == location_id:
                    return "Valid plan."
                else:
                    return f"Invalid plan. {object_id} is not at {location_id}."

        return f"Invalid plan. {object_id} not found."

    def check_class_at(self, object_class, location_id):

        # Iterate through all objects in the scene
        for obj in self.nodes["object"]:
            if obj["class"] == object_class:
                # Check if the object is directly in the specified location
                if obj["state"] == f"inside_of({location_id})" or obj.get("room") == location_id:
                    return "Valid plan."

        return f"Invalid plan. No {object_class} found at {location_id}."

    def check_objects_classes(self, checks):

        invalid_messages = []  # Collect all failure messages

        for check in checks:
            check = check.strip()  # Ensure no leading/trailing spaces

            if check.startswith("check_object_at"):
                try:
                    obj, location = check[16:-1].split(", ")
                    obj, location = obj.strip(), location.strip()
                    result = self.check_object_at(obj, location)
                    if result != "Valid plan.":
                        invalid_messages.append(result)
                except ValueError:
                    invalid_messages.append(f"Invalid format: {check}")

            elif check.startswith("check_class_at"):
                try:
                    obj_class, location = check[15:-1].split(", ")
                    obj_class, location = obj_class.strip(), location.strip()
                    result = self.check_class_at(obj_class, location)
                    if result != "Valid plan.":
                        invalid_messages.append(result)
                except ValueError:
                    invalid_messages.append(f"Invalid format: {check}")

            else:
                invalid_messages.append(f"Invalid check type: {check}")

        # Return "Valid plan." or concatenated error messages
        return "Valid plan." if not invalid_messages else " ".join(invalid_messages)

    def get_room_graph(self):

        agent_location = self.nodes["agent"][0]["location"]
        
        # Filter only room adjacency links
        room_links = [link for link in self.links if all(node in [room["id"] for room in self.nodes["room"]] for node in link.split("-"))]
        
        # Add the agent's current location link
        agent_link = next((link for link in self.links if link.endswith(f"-{self.agent_id}") or link.startswith(f"{self.agent_id}-")), None)
        
        if agent_link:
            room_links.append(agent_link)

        return {"links": room_links}

    def get_scene_graph_subset(self):
        return self.scene_graph_subset

    def get_scene_graph(self):
        return {"nodes": self.nodes, "links": self.links}
    
    def save_to_json(self, filename="scene_graph.json"):
        with open(filename, "w") as f:
            json.dump(self.get_scene_graph(), f, indent=4)

    def initialize_scene_graph_subset(self):
        """Initialize the scene graph subset with only rooms, room connections, and the agent's location."""
        self.scene_graph_subset = {
            "nodes": {
                "room": self.nodes["room"],
                "agent": [self.nodes["agent"][0]],  # Add agent node
                "object": [],
                "asset": []
            },
            "links": self.scene_graph_subset["links"]  # Preserve room links
        }

        # Add the agent's link to its location with the new format
        agent_location = self.nodes["agent"][0]["location"]
        self.scene_graph_subset["links"].append(f"{agent_location}-{self.agent_id}")

    def subset_expand_node(self, room_id):
        """Expand the scene graph subset to include assets and objects in the given room."""
        
        # Ensure we are using the correct room key from `self.nodes`
        room_key = next((room["id"] for room in self.nodes["room"] if room["id"] == room_id), None)

        if room_key is None:
            #print(f" Room '{room_id}' not found in self.nodes['room']. Available rooms: {[r['id'] for r in self.nodes['room']]}")
            return  # Exit function if room is not found

        #print(f"Expanding objects and assets for room: {room_key}")

        # Iterate through objects and assets and add matching ones
        for node in self.nodes["object"] + self.nodes["asset"]:
            if node["room"] == room_key and node not in self.scene_graph_subset["nodes"].get("object", []) and node not in self.scene_graph_subset["nodes"].get("asset", []):
                
                # Correct classification for asset vs object
                if node["class"] in self.asset_classes:
                    self.scene_graph_subset["nodes"].setdefault("asset", []).append(node)
                else:
                    self.scene_graph_subset["nodes"].setdefault("object", []).append(node)

                # Add link between the room and the object/asset
                if f"{room_key}-{node['id']}" not in self.scene_graph_subset["links"]:
                    self.scene_graph_subset["links"].append(f"{room_key}-{node['id']}")


    def subset_contract_node(self, room_id):
        """Remove assets and objects associated with the given room from the scene graph subset."""
        room_key = room_id
        self.scene_graph_subset["nodes"].setdefault("object", [])[:] = [n for n in self.scene_graph_subset["nodes"].get("object", []) if n["room"] != room_key]
        self.scene_graph_subset["nodes"].setdefault("asset", [])[:] = [n for n in self.scene_graph_subset["nodes"].get("asset", []) if n["room"] != room_key]
        self.scene_graph_subset["links"][:] = [l for l in self.scene_graph_subset["links"] if not l.startswith(f"{room_key}-")]

