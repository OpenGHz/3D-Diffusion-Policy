import importlib
from discoverse.robots_env.airbot_play_base import AirbotPlayCfg
from discoverse.task_base import AirbotPlayTaskBase
import numpy as np


class MujocoEnv(object):
    """
    Mujoco environment for airbot_play
    path: path to the script containing the SimNode class and config (mainly for data collection)
    """

    def __init__(self, path: str):
        module = importlib.import_module(path.replace("/", ".").replace(".py", ""))
        node_cls = getattr(module, "SimNode")
        cfg: AirbotPlayCfg = getattr(module, "cfg")
        cfg.headless = False
        self.exec_node: AirbotPlayTaskBase = node_cls(cfg)
        self.reset_position = None
        print("MujocoEnv initialized")

    def _process_raw_obs(self, raw_obs: dict) -> dict:
        obs = {}
        obs["agent_pos"] = np.array(raw_obs["jq"])
        for id in self.exec_node.config.obs_rgb_cam_id:
            obs[f"cam_{id}"] = raw_obs["img"][id][:, :, ::-1]
        # choose camera 1 xyz data
        obs["point_cloud"] = np.hstack(raw_obs["point_cloud"][1])
        return obs

    def reset(self) -> dict:
        self.exec_node.domain_randomization()
        raw_obs = self.exec_node.reset()
        return self._process_raw_obs(raw_obs)

    def step(
        self,
        action,
    ) -> dict:
        raw_obs, pri_obs, rew, ter, info = self.exec_node.step(action)
        return self._process_raw_obs(raw_obs)


if __name__ == "__main__":
    env = MujocoEnv("discoverse/examples/tasks_airbot_play/lift_block.py")
    obs = env.reset()
    print(obs.keys())
    for key, value in obs.items():
        print(key)
        print(value.shape)
    # pprint(obs)
