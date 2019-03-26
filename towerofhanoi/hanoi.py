import gym
import numpy as np
import gym_hanoi
import random


def state_to_q_state(_state):
    """
    Convert from environment state (#disks-tuple) to an integer between 0-3^#disks
    :param _state:
    :return:
    """
    _q_state = 0
    exp = len(_state) - 1
    for disk in _state:
        _q_state += disk * 3 ** exp
        exp -= 1
    return _q_state


def hanoi(num_disks, env_noise=0, save=False, verbose=False, config={}):
    env = gym.make("Hanoi-v0")
    env.set_env_parameters(num_disks=num_disks, env_noise=env_noise, verbose=True)

    action_size = env.action_space.n
    if verbose: print("Action size ", action_size)

    state_size = 3 ** env.num_disks
    if verbose: print("State size ", state_size)

    qtable = np.zeros((state_size, action_size))

    total_episodes = config.get('total_episodes', 10000)  # Total episodes
    total_test_episodes = config.get('total_test_episodes', 100)  # Total test episodes
    max_steps = config.get('max_steps', 99)  # Max steps per episode

    learning_rate = config.get('learning_rate', 0.7)  # Learning rate
    gamma = config.get('gamma', 0.618)  # Discounting rate

    # Exploration parameters
    epsilon = config.get('epsilon', 1.0)  # Exploration rate
    max_epsilon = config.get('max_epsilon', 1.0)  # Exploration probability at start
    min_epsilon = config.get('min_epsilon', 0.01)  # Minimum exploration probability
    decay_rate = config.get('decay_rate', 0.01)  # Exponential decay rate for exploration prob

    exploit, explore = 0, 0

    for episode in range(total_episodes):
        if verbose and episode > 0 and episode % 1000 == 0:
            print("### EPISODE %s ###" % episode)

        # reset environment each episode
        state = env.reset()

        for step in range(max_steps):
            q_state = state_to_q_state(state)

            # decide if explore or exploit
            tradeoff = random.uniform(0, 1)

            if tradeoff > epsilon:
                action = np.argmax(qtable[q_state, :])
                exploit += 1
            else:
                action = env.action_space.sample()
                explore += 1

            new_state, reward, done, info = env.step(action)

            q_new_state = state_to_q_state(new_state)

            if verbose and episode % 1000 == 0 and (step == max_steps - 1 or done):
                print(qtable)
                print(epsilon, exploit, explore)
                print(np.sum(qtable))

            qtable[q_state, action] = qtable[q_state, action] + learning_rate * (
                        reward + gamma * np.max(qtable[q_new_state, :]) - qtable[q_state, action])

            state = new_state

            if done:
                break

        # update exploration-exploitation tradeoff limit
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

    rewards = []

    for episode in range(total_test_episodes):
        state = env.reset()
        total_rewards = 0

        for step in range(max_steps):
            q_state = state_to_q_state(state)

            action = np.argmax(qtable[q_state, :])

            new_state, reward, done, info = env.step(action)

            total_rewards += reward

            if done:
                rewards.append(total_rewards)
                break

            state = new_state

    env.close()
    print("Score over time: %s" % (sum(rewards) / total_test_episodes))

    if verbose:
        print("## FINAL Q-TABLE FOR %s DISKS ##" % num_disks)
        print(qtable)

    if save:
        with open('towerofhanoi/qtable_%s.txt' % num_disks, 'w') as the_file:
            for row in qtable:
                the_file.write(",".join(map(str, row)) + "\n")


if __name__ == "__main__":
    hanoi(2, env_noise=0, save=True, verbose=False)
    hanoi(3, env_noise=0, save=True, verbose=False)
    hanoi(4, env_noise=0, save=True, verbose=False)
    hanoi(5, env_noise=0, save=True, verbose=False)
    hanoi(6, env_noise=0, save=True, verbose=False)
