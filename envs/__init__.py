from gymnasium.envs.registration import register

register(
    id="Autoscaling-v0",
    entry_point="envs.autoscaling_env:AutoscalingEnv",
)

register(
    id="AutoscalingContinuous-v0",
    entry_point="envs.autoscaling_env:AutoscalingEnv",
    kwargs={"continuous": True},
)

register(
    id="AutoscalingContinuousDelta-v0",
    entry_point="envs.autoscaling_env:AutoscalingEnv",
    kwargs={"continuous": True, "continuous_action_mode": "delta"},
)
