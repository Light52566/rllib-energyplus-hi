
from rleplus.examples.amphitheater.env import AmphitheaterEnv

env = AmphitheaterEnv({"output": "/tmp/tests_output"})

obs, _ = env.reset()
print(obs)
obs, rew, done, _, _ = env.step(0)
print(obs, rew, done)
env.close()
