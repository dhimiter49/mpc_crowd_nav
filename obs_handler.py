import numpy as np


class ObsHandler:
    def __init__(self, env_type: str, n_crowd: int = 0):
        self.env_type = env_type
        self.n_inter_crowd = (n_crowd - 1) * 2
        self.n_crowd = n_crowd * 2  # two dimensions x and y


    def get_obs(self, obs):
        return self.__call__(obs)


    def __call__(self, obs):
        if isinstance(obs, tuple):
            obs = obs[0]
        if self.env_type == "CrowdNavigationStatic":
            # goal, crowd, agent velocity, walls, radii
            return (
                obs[:2],
                obs[2:self.n_crowd + 2],
                obs[self.n_crowd + 2:self.n_crowd + 4],
                None,
                obs[self.n_crowd + 4:],
                None
            )
        elif self.env_type == "CrowdNavigation" or\
            self.env_type == "CrowdNavigationConst":
            # goal, crowd, agent velocity, crowd velocities, walls, radii
            return (
                obs[: 2],
                obs[2:2 + self.n_crowd],
                obs[2 + self.n_crowd:self.n_crowd + 4],
                obs[self.n_crowd + 4:2 * self.n_crowd + 4],
                obs[2 * self.n_crowd + 4:],
                None
            )
        elif self.env_type == "CrowdNavigationInter":
            list_of_obs = list(np.array(obs).reshape(self.n_crowd // 2, -1))
            new_list_of_obs = []
            for obs_ in list_of_obs:
                new_list_of_obs.append((
                    obs_[: 2],
                    obs_[2:2 + self.n_inter_crowd],
                    obs_[2 + self.n_inter_crowd:self.n_inter_crowd + 4],
                    obs_[self.n_inter_crowd + 4:2 * self.n_inter_crowd + 4],
                    obs_[2 * self.n_inter_crowd + 4:2 * self.n_inter_crowd + 8],
                    obs_[2 * self.n_inter_crowd + 8:]
                ))
            return new_list_of_obs
        else:
            # goal, agent velocity, walls
            return obs[:2], None, obs[2:4], None, obs[-4:]
