import glfw
import gym
from types import MethodType


# override env.close()
def env_wrapper(env):
    def close(self):
        if self.viewer is not None:
            glfw.destroy_window(self.viewer.window)
            self.viewer = None

    env.unwrapped.close = MethodType(close, env.unwrapped)
    return env


def test():
    env = env_wrapper(gym.make("Reacher-v2"))
    env.reset()
    for _ in range(50):
        env.render()
        env.step([0, 0])
    env.close()


if __name__ == '__main__':
    test()
