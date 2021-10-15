# puzzlegen

<p align="center">
<img src="https://i.imgur.com/gPbEli3.gif" alt="DigitJump" width="15%"/> <img src="https://imgur.com/JdnGwqH.gif" alt="DigitJump" width="15%"/> <img src="https://imgur.com/dkuYE0c.gif" alt="DigitJump" width="15%"/> <img src="https://imgur.com/iEnX3ka.gif" alt="IceSlider" width="15%"/> <img src="https://imgur.com/ovNLLkm.gif alt="IceSlider" width="15%"/> <img src="https://imgur.com/pcYIgZ9.gif" alt="IceSlider" width="15%"/>   
</p>

## Implementation of two procedurally simulated environments with gym interfaces.

- *IceSlider*: the agent needs to reach and stop on the pink square. Each action propels him to the closest obstacle in that direction.
- *DigitJump*: the agent needs to reach the bottom right corner. Each action moves the agent by a number of steps equal to the number at the current position.

These environments are released as supplementary material for the anonymous NeurIPS submission "Planning from Pixels in Environments with Combinatorially Hard Search Spaces".

### Interactive Demo

The environments can be tested interactively in Google COLAB [here](https://colab.research.google.com/drive/1G5l18NXY3O2XVQAOpzy8gDfjyVM6-Pl-?usp=sharing).

### Installation

Simply download and unzip the package, then

```
pip install -e https://github.com/martius-lab/puzzlegen
```

### Example

```
from puzzlegen import DigitJump, IceSlider

env = DigitJump(seed=42)
env.reset()

for _ in range(20):
  obs, rew, done, info = env.step(env.action_space.sample())
  print(f'Reward: {rew}')
```