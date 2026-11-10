"""
Training script for the Wordle RL agent.

Usage:
    python train.py                  # train with defaults
    python train.py --episodes 50    # fewer episodes per iteration (faster)
    python train.py --eval           # run evaluation only (loads saved model)
"""

import argparse
import time
import numpy as np
import torch

from word_list import get_word_list
from wordle_env import WordleEnv
from agent import WordleNetwork, PPOTrainer, RolloutBuffer


def collect_rollouts(
    env: WordleEnv, network: WordleNetwork, buffer: RolloutBuffer, num_episodes: int
) -> dict:
    """Play num_episodes games, storing transitions in the buffer."""
    ep_rewards = []
    ep_solves = []
    ep_turns = []

    for _ in range(num_episodes):
        state = env.reset()
        ep_reward = 0.0

        while True:
            action_mask = env.get_valid_action_mask()
            state_t = torch.FloatTensor(state).unsqueeze(0)
            mask_t = torch.FloatTensor(action_mask).unsqueeze(0)

            action, log_prob, value = network.get_action(state_t, mask_t)
            next_state, reward, done, info = env.step(action)

            buffer.store(state, action, reward, log_prob, value, done, action_mask)
            ep_reward += reward
            state = next_state

            if done:
                ep_rewards.append(ep_reward)
                ep_solves.append(info["solved"])
                ep_turns.append(info["turn"])
                break

    return {
        "mean_reward": np.mean(ep_rewards),
        "solve_rate": np.mean(ep_solves),
        "mean_turns": np.mean([t for t, s in zip(ep_turns, ep_solves) if s]) if any(ep_solves) else 6.0,
    }


def evaluate(env: WordleEnv, network: WordleNetwork, num_games: int = 100, verbose: bool = False) -> dict:
    """Evaluate the agent deterministically (greedy action selection)."""
    solves = 0
    total_turns = []

    for i in range(num_games):
        state = env.reset()
        guesses = []

        while True:
            action_mask = env.get_valid_action_mask()
            state_t = torch.FloatTensor(state).unsqueeze(0)
            mask_t = torch.FloatTensor(action_mask).unsqueeze(0)

            with torch.no_grad():
                logits, _ = network(state_t, mask_t)
                action = logits.argmax(dim=-1).item()  # greedy

            state, _, done, info = env.step(action)
            guesses.append((info["guess"], info["feedback"]))

            if done:
                if info["solved"]:
                    solves += 1
                    total_turns.append(info["turn"])
                if verbose and i < 10:
                    target = info["target"]
                    status = "SOLVED" if info["solved"] else "FAILED"
                    print(f"\n  Game {i+1} [{status}] target={target}")
                    for g, fb in guesses:
                        print(f"    {env.render(g, fb)}  ({g})")
                break

    return {
        "solve_rate": solves / num_games,
        "mean_turns": np.mean(total_turns) if total_turns else 6.0,
        "solves": solves,
        "total": num_games,
    }


def train(args):
    words = get_word_list()
    print(f"Word list: {len(words)} words")

    env = WordleEnv(words)
    network = WordleNetwork(env.state_dim, env.action_dim, hidden=args.hidden)
    trainer = PPOTrainer(
        network,
        lr=args.lr,
        k_epochs=args.k_epochs,
        batch_size=args.batch_size,
        entropy_coeff=args.entropy_coeff,
    )
    buffer = RolloutBuffer()

    print(f"\nState dim:  {env.state_dim}")
    print(f"Action dim: {env.action_dim}")
    params = sum(p.numel() for p in network.parameters())
    print(f"Network:    {params:,} parameters")
    print(f"\nTraining for {args.iterations} iterations, {args.episodes} episodes each\n")
    print(f"{'Iter':>5} | {'Reward':>8} | {'Solve%':>7} | {'Turns':>5} | {'π loss':>8} | {'V loss':>8} | {'Entropy':>8} | {'Time':>6}")
    print("-" * 80)

    best_solve_rate = 0.0
    for iteration in range(1, args.iterations + 1):
        t0 = time.time()
        buffer.clear()

        stats = collect_rollouts(env, network, buffer, args.episodes)
        loss_info = trainer.update(buffer)

        elapsed = time.time() - t0

        if iteration % args.log_every == 0 or iteration == 1:
            print(
                f"{iteration:5d} | "
                f"{stats['mean_reward']:8.2f} | "
                f"{stats['solve_rate']:6.1%} | "
                f"{stats['mean_turns']:5.2f} | "
                f"{loss_info['policy_loss']:8.4f} | "
                f"{loss_info['value_loss']:8.4f} | "
                f"{loss_info['entropy']:8.4f} | "
                f"{elapsed:5.1f}s"
            )

        if stats["solve_rate"] > best_solve_rate:
            best_solve_rate = stats["solve_rate"]
            torch.save(network.state_dict(), "wordle_best.pt")

    print(f"\nBest training solve rate: {best_solve_rate:.1%}")
    print("Saved best model to wordle_best.pt")

    print("\n=== Final Evaluation (100 games, greedy) ===")
    eval_stats = evaluate(env, network, num_games=100, verbose=True)
    print(f"\nSolve rate: {eval_stats['solve_rate']:.1%} ({eval_stats['solves']}/{eval_stats['total']})")
    print(f"Mean turns (when solved): {eval_stats['mean_turns']:.2f}")


def run_eval(args):
    words = get_word_list()
    env = WordleEnv(words)
    network = WordleNetwork(env.state_dim, env.action_dim, hidden=args.hidden)

    print("Loading model from wordle_best.pt ...")
    network.load_state_dict(torch.load("wordle_best.pt", weights_only=True))
    network.eval()

    eval_stats = evaluate(env, network, num_games=200, verbose=True)
    print(f"\nSolve rate: {eval_stats['solve_rate']:.1%} ({eval_stats['solves']}/{eval_stats['total']})")
    print(f"Mean turns (when solved): {eval_stats['mean_turns']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Wordle RL agent")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--episodes", type=int, default=64, help="episodes per iteration")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--k-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--entropy-coeff", type=float, default=0.02)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--eval", action="store_true", help="evaluate saved model")
    args = parser.parse_args()

    if args.eval:
        run_eval(args)
    else:
        train(args)
