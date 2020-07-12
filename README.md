# gym-poly-reactor
a discrete version of industrial polymerization reactor environment. this package is a gym-style warapping of `do-mpc` industrial polyerization reactor model.
the original formulation and documentation about the model can be found from [here](https://www.do-mpc.com/en/latest/example_gallery/industrial_poly.html).

## installation 
we support `pip` installation.

```
pip install gym-poly-reactor
```

## Dependencies
`gym-poly-reactor` is build upon followings:
1. `numpy`
2. `gym`
3. `do-mpc`

If you install the package through `pip`, then the pip installation precedure would install the dependencies also. 

## Quickstart
You can easily simulate one episode of industrial polyerization reactor environment by using the following code snippet

```python
from gym_poly_reactor.envs.poly_reactor import PolyReactor

env = PolyReactor() # instantiate industrial polyerization reactor environment
s0 = env.reset() # setup the model
while True:
  action = env.env.action_space.sample()
  next_state, reward, done, _ = env.step(action)
```
