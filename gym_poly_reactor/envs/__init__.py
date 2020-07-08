from gym.envs.registration import register

register(
    id='poly-reactor-v0',
    entry_point='gym_poly_reactor.envs:PolyReactor',
)


