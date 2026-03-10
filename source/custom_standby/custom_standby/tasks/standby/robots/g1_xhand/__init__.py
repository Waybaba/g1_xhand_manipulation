import gymnasium as gym

from custom_standby.tasks.standby import agents

gym.register(
    id="Custom-Standby-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.standby_env_cfg:G1StandbyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1StandbyPPORunnerCfg",
    },
)

gym.register(
    id="Custom-Standby-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.standby_env_cfg:G1StandbyEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1StandbyPPORunnerCfg",
    },
)
