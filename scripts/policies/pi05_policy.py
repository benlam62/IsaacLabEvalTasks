# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from gr00t.experiment.data_config import DATA_CONFIG_MAP
#from gr00t.model.policy import Gr00tPolicy
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.schema import DatasetMetadata
from io_utils import load_gr1_joints_config
from policies.image_conversion import resize_frames_with_padding
from policies.joints_conversion import remap_policy_joints_to_sim_joints, remap_sim_joints_to_policy_joints
from policies.policy_base import PolicyBase
from robot_joints import JointsAbsPosition

from isaaclab.sensors import Camera
from config.args import Gr00tN1ClosedLoopArguments
from typing import Any, Dict, Optional, Union
from openpi_client import websocket_client_policy
from openpi_client import image_tools
import torch
from pathlib import Path
import json
import numpy as np

class Pi05Policy(PolicyBase):
    def __init__(self, args: Gr00tN1ClosedLoopArguments):
        self.args = args
        self.policy = self._load_policy()
        self._load_policy_joints_config()
        self._load_sim_joints_config()
        data_config = DATA_CONFIG_MAP[self.args.data_config]
        self._modality_transform=data_config.transform()
        self._modality_transform.eval()  # set this to eval mode

        embodiment_tag = self.args.embodiment_tag
        if isinstance(embodiment_tag, str):
            self.embodiment_tag = EmbodimentTag(embodiment_tag)
        else:
            self.embodiment_tag = embodiment_tag
        self._load_metadata()

    def _load_policy_joints_config(self):
        """Load the policy joint config from the data config."""
        self.gr00t_joints_config = load_gr1_joints_config(self.args.gr00t_joints_config_path)

    def _load_sim_joints_config(self):
        """Load the simulation joint config from the data config."""
        self.gr1_state_joints_config = load_gr1_joints_config(self.args.state_joints_config_path)
        self.gr1_action_joints_config = load_gr1_joints_config(self.args.action_joints_config_path)

    def _load_policy(self):
        #"""Load the policy from the model path."""
        #assert os.path.exists(self.args.model_path), f"Model path {self.args.model_path} does not exist"

        # Use the same data preprocessor as the loaded fine-tuned ckpts

        # load the policy
        #return Gr00tPolicy(
        #    model_path=self.args.model_path,
        #    modality_config=modality_config,
        #    modality_transform=modality_transform,
        #    embodiment_tag=self.args.embodiment_tag,
        #    denoising_steps=self.args.denoising_steps,
        #    device=self.args.policy_device,
        #)
        return websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

    def _load_metadata(self):
        """Load the transforms for the model."""
        # Load metadata for normalization stats
        #metadata_path = "/home/benlam/IsaacLabEvalTasks/scripts/metadata.json"
        metadata_path = "/home/benlam/IsaacLabEvalTasks/scripts/taskconfig/" + self.args.task_name + "/metadata.json"

        with open(metadata_path, "r") as f:
            metadatas = json.load(f)

        # Get metadata for the specific embodiment
        metadata_dict = metadatas.get(self.embodiment_tag.value)
        if metadata_dict is None:
            raise ValueError(
                f"No metadata found for embodiment tag: {self.embodiment_tag.value}",
                f"make sure the metadata.json file is present at {metadata_path}",
            )

        metadata = DatasetMetadata.model_validate(metadata_dict)
        self._modality_transform.set_metadata(metadata)
        self.metadata = metadata

    def step(self, current_state: JointsAbsPosition, camera: Camera) -> JointsAbsPosition:
        """Call every simulation step to update policy's internal state."""
        pass

    def apply_transforms(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply transforms to the observation.

        Args:
            obs (Dict[str, Any]): The observation to transform.

        Returns:
            Dict[str, Any]: The transformed observation.
        """
        # Ensure correct dimensions before applying transforms
        return self._modality_transform(obs)

    def unapply_transforms(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unapply transforms to the action.

        Args:
            action (Dict[str, Any]): The action to unapply transforms to.

        Returns:
            Dict[str, Any]: The untransformed action.
        """
        return self._modality_transform.unapply(action)
      
    #def _get_action_from_normalized_input(self, normalized_input: Dict[str, Any]) -> torch.Tensor:
        # Set up autocast context if needed
        #with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
        #    model_pred = self.model.get_action(normalized_input)
        #normalized_action = model_pred["action_pred"].float()
        #return normalized_action
    def _get_action_from_normalized_input(self, normalized_input: Dict[str, Any]) -> Dict[str, Any]:        
        normalized_action = self.policy.infer(normalized_input)
        return normalized_action

    #def _get_unnormalized_action(self, normalized_action: torch.Tensor) -> Dict[str, Any]:
    #    return self.unapply_transforms({"action": normalized_action.cpu()})
    def _get_unnormalized_action(self, normalized_action: Dict[str, Any]) -> Dict[str, Any]:
        return self.unapply_transforms({"action": normalized_action})
    
    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction with the model.
        Args:
            obs (Dict[str, Any]): The observation to make a prediction for.

        e.g. obs = {
            "video.<>": np.ndarray,  # (T, H, W, C)
            "state.<>": np.ndarray, # (T, D)
        }

        or with batched input:
        e.g. obs = {
            "video.<>": np.ndarray,, # (B, T, H, W, C)
            "state.<>": np.ndarray, # (B, T, D)
        }

        Returns:
            Dict[str, Any]: The predicted action.
        """
        # let the get_action handles both batch and single input
        is_batch =_check_state_is_batched(observations)
        if not is_batch:
            observations = unsqueeze_dict_values(observations)

        normalized_input = unsqueeze_dict_values
        # Apply transforms
        normalized_input = self.apply_transforms(observations)

        normalized_action = self._get_action_from_normalized_input(normalized_input)
        unnormalized_action = self._get_unnormalized_action(normalized_action)

        if not is_batch:
            unnormalized_action = squeeze_dict_values(unnormalized_action)
        return unnormalized_action
    
    def get_openpi_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction with the model.
        Args:
            obs (Dict[str, Any]): The observation to make a prediction for.

        e.g. obs = {
            "video.<>": np.ndarray,  # (T, H, W, C)
            "state.<>": np.ndarray, # (T, D)
        }

        or with batched input:
        e.g. obs = {
            "video.<>": np.ndarray,, # (B, T, H, W, C)
            "state.<>": np.ndarray, # (B, T, D)
        }

        Returns:
            Dict[str, Any]: The predicted action.
        """

        data_config = DATA_CONFIG_MAP[self.args.data_config]

        
        for key in data_config.video_keys:
            # Remove the batch and time dimension of the image (b t h w c) > (h w c), resize the image and convert to uint 8) 
            observations[key] = image_tools.convert_to_uint8(image_tools.resize_with_pad(observations[key][0,0,:,:,:], 224, 224))

        observations["observation/image"] = observations.pop(data_config.video_keys[0])
        #from PIL import Image
        #Image.fromarray(observations["observation/image"]).save("debug_camera_image.png")

        #Remove the batch and time dimension of the state
        observations["observation/state"] = np.concatenate([observations.pop(key)[0,0,...] for key in data_config.state_keys], axis=-1)
        action = self.policy.infer(observations)
        return action
        
    def get_new_goal(
        self, current_state: JointsAbsPosition, ego_camera: Camera, language_instruction: str
    ) -> JointsAbsPosition:
        """
        Run policy prediction on the given observations. Produce a new action goal for the robot.

        Args:
            current_state: robot proprioceptive state observation
            ego_camera: camera sensor observation
            language_instruction: language instruction for the task

        Returns:
            A dictionary containing the inferred action for robot joints.
        """
        rgb = ego_camera.data.output["rgb"]
        # Apply preprocessing to rgb
        rgb = resize_frames_with_padding(
            rgb, target_image_size=self.args.target_image_size, bgr_conversion=False, pad_img=True
        )
        # Retrieve joint positions as proprioceptive states and remap to policy joint orders
        robot_state_policy = remap_sim_joints_to_policy_joints(current_state, self.gr00t_joints_config)

        # Pack inputs to dictionary and run the inference
        observations = {
            #"annotation.human.action.task_description": [language_instruction] # list of strings,
            "prompt": language_instruction,
            "video.ego_view": rgb.reshape(-1, 1, 256, 256, 3),  # numpy array of shape (N, 1, 256, 256, 3)
            "state.left_arm": robot_state_policy["left_arm"].reshape(-1, 1, 7),  # numpy array of shape (N, 1, 7)
            "state.right_arm": robot_state_policy["right_arm"].reshape(-1, 1, 7),  # numpy array of shape (N, 1, 7)
            "state.left_hand": robot_state_policy["left_hand"].reshape(-1, 1, 6),  # numpy array of shape (N, 1, 6)
            "state.right_hand": robot_state_policy["right_hand"].reshape(-1, 1, 6),  # numpy array of shape (N, 1, 6)
        }
        
        #robot_action_policy = self.policy.get_action(observations)
        action_data = self.get_openpi_action(observations)

        #populate the raw actions by robot modalities
        start_dim = 0
        robot_action_policy = {}
        action_tensor = action_data.pop("actions")
        for key in self.metadata.modalities.action:
            end_dim = start_dim + self.metadata.modalities.action[key].shape[0]
            robot_action_policy[f"action.{key}"] = action_tensor[..., start_dim:end_dim]
            start_dim = end_dim

        #if the action data is not in batched, converts it to be batched of size 1.
        for k, v in robot_action_policy.items():
            if len(v.shape) <3:
               robot_action_policy[k] = np.expand_dims(v, axis=0)

        robot_action_sim = remap_policy_joints_to_sim_joints(
            robot_action_policy, self.gr00t_joints_config, self.gr1_action_joints_config, self.args.simulation_device
        )

        return robot_action_sim

    def reset(self):
        """Resets the policy's internal state."""
        # As GN1 is a single-shot policy, we don't need to reset its internal state
        pass

#######################################################################################################


# Helper functions

def _check_state_is_batched(obs: Dict[str, Any]) -> bool:
    for k, v in obs.items():
        if "state" in k and len(v.shape) < 3:  # (B, Time, Dim)
            return False
    return True

def unsqueeze_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unsqueeze the values of a dictionary.
    This converts the data to be batched of size 1.
    """
    unsqueezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            unsqueezed_data[k] = np.expand_dims(v, axis=0)
        elif isinstance(v, torch.Tensor):
            unsqueezed_data[k] = v.unsqueeze(0)
        else:
            unsqueezed_data[k] = v
    return unsqueezed_data


def squeeze_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Squeeze the values of a dictionary. This removes the batch dimension.
    """
    squeezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            squeezed_data[k] = np.squeeze(v)
        elif isinstance(v, torch.Tensor):
            squeezed_data[k] = v.squeeze()
        else:
            squeezed_data[k] = v
    return squeezed_data
