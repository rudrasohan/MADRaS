import ray
import gym
import argparse
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from ray.tune.logger import pretty_print
from ray.rllib.env.multi_agent_env import MultiAgentEnv

parser = argparse.ArgumentParser()
parser.add_argument("--num-iters", type=int, default=20)

if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()

    # Simple environment with 4 independent cartpole entities
    #register_env("multi_cartpole", lambda _: MultiCartpole(4))
    env = gym.make("Madras-v0")
    obs_space_1 = env.agents["MadrasAgent_0"].observation_space
    act_space_1 = env.agents["MadrasAgent_0"].action_space
    obs_space_2 = env.agents["MadrasAgent_1"].observation_space
    act_space_2 = env.agents["MadrasAgent_1"].action_space
    # You can also have multiple policies per trainer, but here we just
    # show one each for PPO and DQN.
    policies = {
        "ppo_policy_1": (PPOTFPolicy, obs_space_1, act_space_1, {}),
        "ppo_policy_2": (PPOTFPolicy, obs_space_2, act_space_2, {}),
    }

    def policy_mapping_fn(agent_id):
        if agent_id == "MadrasAgent_0":
            return "ppo_policy_1"
        else:
            return "ppo_policy_2"

    ppo_trainer = PPOTrainer(
        env="Madras-v0",
        config={
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
                #"policies_to_train": ["ppo_policy_1","ppo_policy_2"],
            },
            # disable filters, otherwise we would need to synchronize those
            # as well to the DQN agent
           
        })


    # disable DQN exploration when used by the PPO trainer
    # ppo_trainer.workers.foreach_worker(
    #     lambda ev: ev.for_policy(
    #         lambda pi: pi.set_epsilon(0.0), policy_id="dqn_policy"))

    # You should see both the printed X and Y approach 200 as this trains:
    # info:
    #   policy_reward_mean:
    #     dqn_policy: X
    #     ppo_policy: Y
    for i in range(args.num_iters):
        print("== Iteration", i, "==")

        # improve the DQN policy
        #print("-- DQN --")
        #print(pretty_print(dqn_trainer.train()))

        # improve the PPO policy
        print("-- PPO --")
        print(pretty_print(ppo_trainer.train()))

        # swap weights to synchronize
        #dqn_trainer.set_weights(ppo_trainer.get_weights(["ppo_policy"]))
        #ppo_trainer.set_weights(dqn_trainer.get_weights(["dqn_policy"]))
