# Wordle Reinforcement Learning Agent (Action-Masked PPO)

A Proximal Policy Optimization (**PPO-Clip + GAE**) agent trained to play Wordle in a custom Gymnasium-style PyTorch environment using a compact belief-state representation and dynamic invalid-action masking.

## Key Results

- **Algorithm:** PPO-Clip ($\epsilon = 0.2$) with Generalized Advantage Estimation ($\gamma = 0.99, \lambda = 0.95$)
- **Evaluation (100 games, greedy policy):** **100% solve rate**, **3.90 mean turns** to solve (677-word vocabulary)
- **Pretrained weights:** Included in `wordle_best.pt`

## Architecture & MDP Design

1. **Sufficient-Statistic Belief State (`209` dimensions):**
   Instead of feeding raw guess histories ($6 \times 5 \times 29$) and forcing the network to re-derive constraints on every forward pass, `WordleEnv` maintains a compact belief state:
   - `letter_position_mask` ($5 \times 26 = 130$): binary mask indicating whether letter $c$ remains valid at position $p$ (handles duplicate-letter gray/yellow edge cases via a two-pass feedback rule).
   - `letter_status` ($26 \times 3 = 78$): one-hot encoding per letter (`[unknown, confirmed_in_word, eliminated]`).
   - `turn_number` ($1$): normalized turn index in $[0, 1]$.
2. **Dynamic Action Masking:**
   At each step, `get_valid_action_mask()` filters the discrete action space ($|V| = 677$) to words consistent with current positional constraints, adding $\log(\text{mask})$ to policy logits before the `Categorical` softmax so zero probability mass is wasted on eliminated words.
3. **Actor-Critic Network (`WordleNetwork`):**
   Shared 3-layer MLP trunk (`209 → 512 → 512 → 256`, ReLU) feeding a masked policy head (`256 → |V|`) and scalar state-value head (`256 → 1`).

## Quick Start

```bash
pip install -r requirements.txt
python train.py --eval           # evaluate saved checkpoint (wordle_best.pt)
python train.py                  # train from scratch (500 iterations × 64 episodes)
```

## Repository Structure

- `wordle_env.py` — Custom RL environment, belief-state transitions, duplicate-letter feedback logic, and valid-action mask generation
- `agent.py` — `WordleNetwork` (masked Actor-Critic), `RolloutBuffer` (GAE computation), and `PPOTrainer`
- `train.py` — Rollout collection, training loop, and greedy evaluation harness
- `word_list.py` — 677-word 5-letter vocabulary
- `wordle_best.pt` — Trained model checkpoint

