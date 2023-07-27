import pickle as pkl
from pathlib import Path
import numpy as np

DATA_PATH = Path("/home/siddhant/dataset/expert_demos/robotgym") # Path - /path/to/dir/expert_demos/robotgym
SAVE_PATH = Path("./data")

tasks = {
    "RobotBoxOpen": "open the box", 
    "RobotButtonPress": "press the button",
    "RobotCupStacking": "stack the cups",
    "RobotDoorClose": "close the door",
    "RobotEraseBoard": "erase the board",
    "RobotHangBag": "hang the bag on the hook",
    "RobotHangHanger": "hang the hanger on the rod",
    "RobotHangMug": "hang the mug on the hook",
    "RobotInsertPeg": "insert the peg in the cup",
    "RobotPour": "pour the almonds into the cup",
    "RobotReach": "reach the blue mark on the table",
    "RobotTurnKnob": "turn the knob",
}

# 0 - open, 1 - closed
gripper = {
    "RobotBoxOpen": 0, 
    "RobotButtonPress": 1,
    "RobotCupStacking": 1,
    "RobotDoorClose": 1,
    "RobotEraseBoard": 1,
    "RobotHangBag": 1,
    "RobotHangHanger": 1,
    "RobotHangMug": 1,
    "RobotInsertPeg": 1,
    "RobotPour": 1,
    "RobotReach": 1,
    "RobotTurnKnob": 1,
}

# Default = [180, 0, 180]  Roll pitch yaw
RPY = {
    "RobotBoxOpen": [90, 0 , 180], 
    "RobotButtonPress": [180, 0, 180],
    "RobotCupStacking": [180, 0, 180],
    "RobotDoorClose": [180, 0, 180],
    "RobotEraseBoard": [180, 0, 180],
    "RobotHangBag": [180, -90, 180],
    "RobotHangHanger": [180, -90, 180],
    "RobotHangMug": [180, -90, 180],
    "RobotInsertPeg": [180, 0, 180],
    "RobotPour": [-90, 90, 180],
    "RobotReach": [180, 0, 180],
    "RobotTurnKnob": [180, 0, 180],
}


# Create data dir
SAVE_PATH.mkdir(parents=True, exist_ok=True)
SAVE_PATH.joinpath("train").mkdir(parents=True, exist_ok=True)

# List folders in the data path
file_count = 0
folders = [f for f in DATA_PATH.iterdir() if f.is_dir()]
for folder in folders:
    if folder.name.split("-")[0] not in tasks.keys():
        continue

    files = [f for f in folder.iterdir() if f.is_file()]

    # Save data
    for file in files:
        data = pkl.load(open(file, "rb"))
        observations = data[0][0]
        states = data[1][0]
        actions = data[2][0]
        rewards = data[3][0]
        language_instruction = tasks[folder.name.split("-")[0]]
        
        # Clip states and actions
        states = np.clip(states, -1, 1)
        actions = np.clip(actions, -1, 1)

        episode = []
        for step in range(len(observations)):
            rpy = np.array(RPY[folder.name.split("-")[0]]) / 180 * 2 * np.pi
            if 'Pour' not in folder.name:
                state = np.array([
                    states[step][0], states[step][1], states[step][2],
                    rpy[0], rpy[1], rpy[2], gripper[folder.name.split("-")[0]]
                ])
                action = np.array([
                    actions[step][0], actions[step][1], actions[step][2],
                    0, 0, 0, 0
                ])
            else:
                state = np.array([
                    states[step][0], states[step][1], states[step][2],
                    rpy[0], states[step][3], rpy[2], gripper[folder.name.split("-")[0]]
                ])
                action = np.array([
                    actions[step][0], actions[step][1], actions[step][2],
                    0, actions[step][3], 0, 0
                ])

            # Store episode
            episode.append({
                'image': np.transpose(np.array(observations[step], dtype=np.uint8), (1,2,0)),
                'state': np.array(state, dtype=np.float32),
                'action': np.array(action, dtype=np.float32),
                'reward': rewards[step],
                'language_instruction': language_instruction,
            })
            print(actions[step].shape, actions[step], folder.name)
        np.save(SAVE_PATH.joinpath("train", f"episode_{file_count}.npy"), episode)
        file_count += 1