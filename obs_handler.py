class ObsHandler:
    def __init__(self, env_type: str, n_crowd: int = 0):
        self.env_type = env_type
        self.n_crowd = n_crowd * 2  # two dimensions x and y


    def get_obs(self, obs):
        return self.__call__(obs)


    def __call__(self, obs):
        if isinstance(obs, tuple):
            obs = obs[0]
        if self.env_type == "CrowdNavigationStatic":
            # goal, crowd, agent velocity, walls
            return (
                obs[0][: 2],
                obs[0][2:self.n_crowd + 2],
                obs[0][self.n_crowd + 2:self.n_crowd + 4],
                obs[0][self.n_crowd + 4:]
            )
        elif self.env_type == "CrowdNavigation":
            # goal, crowd, agent velocity, crowd velocities, walls
            return (
                obs[: 2],
                obs[2:2 + self.n_crowd],
                obs[2 + self.n_crowd:self.n_crowd + 4],
                obs[self.n_crowd + 4:2 * self.n_crowd + 4],
                obs[2 * self.n_crowd + 4:]
            )
        else:
            # goal, agent velocity, walls
            return obs[:2], obs[2:4], obs[-4:]
