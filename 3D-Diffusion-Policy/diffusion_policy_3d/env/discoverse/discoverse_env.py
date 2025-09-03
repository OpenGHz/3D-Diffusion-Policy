import gym
import numpy as np
from termcolor import cprint
from gym import spaces
import torch
import pytorch3d.ops as torch3d_ops
from diffusion_policy_3d.env.discoverse.airbot_play_mujoco_env import MujocoEnv


def point_cloud_sampling(point_cloud: np.ndarray, num_points: int, method: str = "fps"):
    """
    support different point cloud sampling methods
    point_cloud: (N, 6), xyz+rgb or (N, 3), xyz
    """
    if num_points == "all":  # use all points
        return point_cloud

    if point_cloud.shape[0] <= num_points:
        # cprint(f"warning: point cloud has {point_cloud.shape[0]} points, but we want to sample {num_points} points", 'yellow')
        # pad with zeros
        point_cloud_dim = point_cloud.shape[-1]
        point_cloud = np.concatenate(
            [
                point_cloud,
                np.zeros((num_points - point_cloud.shape[0], point_cloud_dim)),
            ],
            axis=0,
        )
        return point_cloud

    if method == "uniform":
        # uniform sampling
        sampled_indices = np.random.choice(
            point_cloud.shape[0], num_points, replace=False
        )
        point_cloud = point_cloud[sampled_indices]
    elif method == "fps":
        # fast point cloud sampling using torch3d
        point_cloud = torch.from_numpy(point_cloud).unsqueeze(0).cuda()
        num_points = torch.tensor([num_points]).cuda()
        # remember to only use coord to sample
        _, sampled_indices = torch3d_ops.sample_farthest_points(
            points=point_cloud[..., :3], K=num_points
        )
        point_cloud = point_cloud.squeeze(0).cpu().numpy()
        point_cloud = point_cloud[sampled_indices.squeeze(0).cpu().numpy()]
    else:
        raise NotImplementedError(
            f"point cloud sampling method {method} not implemented"
        )

    return point_cloud


TASK_BOUNDS = {
    "default": [-0.5, -1.5, -0.795, 1, -0.4, 100],
}


class DiscoverseEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(
        self,
        task_name,
        device="cuda:0",
        use_point_crop=True,
        num_points=1024,
    ):
        super(DiscoverseEnv, self).__init__()
        self._env = MujocoEnv(f"discoverse/examples/tasks_airbot_play/{task_name}.py")
        self.device_id = int(device.split(":")[-1])

        self.image_size = 128

        self.use_point_crop = use_point_crop
        cprint("[MetaWorldEnv] use_point_crop: {}".format(self.use_point_crop), "cyan")
        self.num_points = num_points  # 512

        # x_angle = 61.4
        # y_angle = -7
        # self.pc_transform = np.array([
        #     [1, 0, 0],
        #     [0, np.cos(np.deg2rad(x_angle)), np.sin(np.deg2rad(x_angle))],
        #     [0, -np.sin(np.deg2rad(x_angle)), np.cos(np.deg2rad(x_angle))]
        # ]) @ np.array([
        #     [np.cos(np.deg2rad(y_angle)), 0, np.sin(np.deg2rad(y_angle))],
        #     [0, 1, 0],
        #     [-np.sin(np.deg2rad(y_angle)), 0, np.cos(np.deg2rad(y_angle))]
        # ])
        self.pc_transform = None
        # self.pc_scale = np.array([1, 1, 1])
        # self.pc_offset = np.array([0, 0, 0])
        self.pc_scale = None
        self.pc_offset = None
        if task_name in TASK_BOUNDS:
            x_min, y_min, z_min, x_max, y_max, z_max = TASK_BOUNDS[task_name]
        else:
            x_min, y_min, z_min, x_max, y_max, z_max = TASK_BOUNDS["default"]
        self.min_bound = [x_min, y_min, z_min]
        self.max_bound = [x_max, y_max, z_max]

        self.episode_length = self._max_episode_steps = 200
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        # self.obs_sensor_dim = self.get_robot_state().shape[0]

        self.observation_space = spaces.Dict(
            {
                "agent_pos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
                ),
                "point_cloud": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_points, 3),
                    dtype=np.float32,
                ),
            }
        )

    def _process_pcd(self, point_cloud, use_rgb=True):
        if not use_rgb:
            point_cloud = point_cloud[..., :3]

        if self.pc_transform is not None:
            point_cloud[:, :3] = point_cloud[:, :3] @ self.pc_transform.T
        if self.pc_scale is not None:
            point_cloud[:, :3] = point_cloud[:, :3] * self.pc_scale

        if self.pc_offset is not None:
            point_cloud[:, :3] = point_cloud[:, :3] + self.pc_offset

        if self.use_point_crop:
            if self.min_bound is not None:
                mask = np.all(point_cloud[:, :3] > self.min_bound, axis=1)
                point_cloud = point_cloud[mask]
            if self.max_bound is not None:
                mask = np.all(point_cloud[:, :3] < self.max_bound, axis=1)
                point_cloud = point_cloud[mask]

        point_cloud = point_cloud_sampling(point_cloud, self.num_points, "fps")

        return point_cloud

    def _process_obs(self, obs: dict) -> dict:
        point_cloud = self._process_pcd(obs["point_cloud"], use_rgb=False)
        obs_dict = {
            # "image": obs_pixels,
            # "depth": depth,
            "agent_pos": obs["agent_pos"],
            "point_cloud": point_cloud,
            # "full_state": raw_state,
        }
        self._image = obs["cam_1"]
        return obs_dict

    def step(self, action: np.ndarray):
        obs = self._env.step(action)
        self.cur_step += 1

        done = False
        reward = 0.0
        env_info = {"goal_achieved": False}
        done = done or self.cur_step >= self.episode_length

        return self._process_obs(obs), reward, done, env_info

    def reset(self):
        self.cur_step = 0
        return self._process_obs(self._env.reset())

    def seed(self, seed=None):
        pass

    def set_seed(self, seed=None):
        pass

    def render(self, mode="rgb_array"):
        img = self._image
        return img

    def close(self):
        pass


if __name__ == "__main__":
    env = DiscoverseEnv(task_name="lift_block", device="cuda:0")
    obs = env.reset()
    print(obs.keys())
    for key, value in obs.items():
        print(key)
        print(value.shape)
    obs, reward, done, info = env.step(env.action_space.sample())
    print(obs.keys())
    for key, value in obs.items():
        print(key)
        print(value.shape)
