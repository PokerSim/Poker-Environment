# --- PokerGym.py (Modified + PokerSim.py update combined) ---
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
from collections import Counter
import matplotlib.pyplot as plt
from itertools import combinations
import StatsBot

from AllInBot import take_action as allin_take_action
from StatsBot import take_action as stats_take_action

class PokerEnv(gym.Env):
    def __init__(self, players, agent_index=0):
        super(PokerEnv, self).__init__()
        self.players = players
        self.agent_index = agent_index
        self.pot = 0
        self.big_blind = 20
        self.small_blind = 10
        self.dealer = 0
        self.deck = self.initialize_deck()
        self.community_cards = []
        self.burn_cards = []
        self.round_stage = "pre-flop"
        self.current_bet = 0
        self.action_pointer = 0
        self.raise_counter = 0
        self.re_raise_occurred = False
        self.all_in_occurred = False
        self.pending_all_in = False
        self.betting_locked = False

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(107,), dtype=np.float32)

    def initialize_deck(self):
        suits = ["s", "h", "d", "c"]
        ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
        return [rank + suit for suit in suits for rank in ranks]

    def shuffle_deck(self):
        random.shuffle(self.deck)

    def deal_hands(self):
        self.shuffle_deck()
        for player in self.players:
            player.hand = []
        for _ in range(2):
            for player in self.players:
                if player.active:
                    player.receive_card(self.deck.pop())

    def post_blinds(self):
        small_blind_index = (self.dealer + 1) % len(self.players)
        big_blind_index = (self.dealer + 2) % len(self.players)
        sb_player = self.players[small_blind_index]
        bb_player = self.players[big_blind_index]

        # Automatically fold players with 0 chips and skip blind posting
        for player in self.players:
            if player.chips <= 0:
                player.active = False

        # Post small blind
        if sb_player.active:
            sb_blind = min(self.small_blind, sb_player.chips)
            sb_player.chips -= sb_blind
            sb_player.current_bet = sb_blind
            self.pot += sb_blind
        else:
            sb_blind = 0

        # Post big blind
        if bb_player.active:
            bb_blind = min(self.big_blind, bb_player.chips)
            bb_player.chips -= bb_blind
            bb_player.current_bet = bb_blind
            self.pot += bb_blind
        else:
            bb_blind = 0

        self.current_bet = max(sb_blind, bb_blind)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pot = 0
        self.deck = self.initialize_deck()
        self.community_cards = []
        self.burn_cards = []
        self.round_stage = "pre-flop"
        self.current_bet = 0
        self.raise_counter = 0
        self.re_raise_occurred = False
        self.all_in_occurred = False
        self.pending_all_in = False
        self.betting_locked = False
        for player in self.players:
            player.active = True
            player.current_bet = 0
            player.hand = []
        self.dealer = (self.dealer + 1) % len(self.players)
        self.deal_hands()
        self.post_blinds()
        self.action_pointer = (self.dealer + 3) % len(self.players)
        return self._get_obs(), {}

    def step(self, action):
        player = self.players[self.action_pointer]

        if not player.active:
            self._advance_pointer()
            return self._get_obs(), 0, False, False, {}

        action_name, bet_amount = player.take_action(self.current_bet, self.community_cards)

        if action_name == "fold":
            player.active = False
            print(f"{player.name} folds")
        elif action_name == "call":
            call_amount = self.current_bet - player.current_bet
            actual_call = min(call_amount, player.chips)
            player.chips -= actual_call
            player.current_bet += actual_call
            self.pot += actual_call
            print(f"{player.name} calls {actual_call}")
            if player.chips == 0:
                self.all_in_occurred = True
                self.pending_all_in = True
        elif action_name == "raise":
            if self.betting_locked or self.re_raise_occurred:
                # Cannot raise anymore, only call
                call_amount = self.current_bet - player.current_bet
                actual_call = min(call_amount, player.chips)
                player.chips -= actual_call
                player.current_bet += actual_call
                self.pot += actual_call
                print(f"{player.name} (forced) calls {actual_call}")
                if player.chips == 0:
                    self.all_in_occurred = True
                    self.pending_all_in = True
            else:
                if isinstance(player, AllInPlayer):
                    # Special handling: AllInPlayer *always* all-ins
                    print(f"{player.name} goes all-in with {player.chips} chips!")
                    self.handle_all_in_raise(player)
                    self.all_in_occurred = True
                    self.pending_all_in = True
                else:
                    raise_amount = bet_amount - player.current_bet
                    if raise_amount >= player.chips:
                        # Player can't fully raise — goes all-in
                        print(f"{player.name} goes all-in with {player.chips} chips!")
                        self.handle_all_in_raise(player)
                        self.all_in_occurred = True
                        self.pending_all_in = True
                    else:
                        actual_raise = raise_amount
                        player.chips -= actual_raise
                        player.current_bet += actual_raise
                        self.pot += actual_raise
                        self.current_bet = player.current_bet
                        print(f"{player.name} raises to {self.current_bet}")
                        if self.raise_counter == 0:
                            self.raise_counter += 1
                        else:
                            self.re_raise_occurred = True

        self._advance_pointer()

        active_players = [p for p in self.players if p.active]
        if len(active_players) == 1:
            self.distribute_pot(active_players)
            return self._get_obs(), 0, True, False, {}

        if self.pending_all_in and self.action_pointer == self.dealer:
            print("All-in occurred and players finished actions. Dealing rest of board...")
            self.betting_locked = True
            self.deal_remaining_board()
            self.determine_winner()
            return self._get_obs(), 0, True, False, {}

        return self._get_obs(), 0, False, False, {}

    def handle_all_in_raise(self, player):
        all_in_amount = player.chips
        player.current_bet += all_in_amount
        self.pot += all_in_amount
        player.chips = 0
        self.current_bet = max(self.current_bet, player.current_bet)
    def deal_remaining_board(self):
        stages = ["flop", "turn", "river"]
        if len(self.community_cards) < 3:
            self.burn_cards.append(self.deck.pop())
            self.community_cards.extend([self.deck.pop() for _ in range(3)])
            print(f"Dealing flop: {self.community_cards}")
        while len(self.community_cards) < 5:
            self.burn_cards.append(self.deck.pop())
            self.community_cards.append(self.deck.pop())
            print(f"Board now: {self.community_cards}")

    #def determine_showdown(self):
     #   active_players = [p for p in self.players if p.active]
      #  self.distribute_pot(active_players)

    def _advance_pointer(self):
        self.action_pointer = (self.action_pointer + 1) % len(self.players)

    def _get_obs(self):
        return np.zeros(self.observation_space.shape)

    def distribute_pot(self, winners):
        if len(winners) == 1:
            winners[0].chips += self.pot
            print(f"{winners[0].name} wins {self.pot} chips!")
        else:
            split_amount = self.pot // len(winners)
            for winner in winners:
                winner.chips += split_amount
            print(f"Pot split among {[w.name for w in winners]} for {split_amount} chips each!")
        self.pot = 0

    def print_game_state(self):
        print("\n--- Current Game State ---")
        for p in self.players:
            status = "Active" if p.active else "Folded"
            print(f"{p.name}: Hand: {p.hand}, Chips: {p.chips}, Status: {status}")
        print(f"Board: {self.community_cards}")
        print(f"Pot: {self.pot}\n")

    def evaluate_hand(self, cards):
        values = {str(k): k for k in range(2, 11)}
        values.update({"J": 11, "Q": 12, "K": 13, "A": 14})
        suits = [c[-1] for c in cards]
        ranks = [c[:-1] for c in cards]
        rank_values = sorted([values[r] for r in ranks], reverse=True)

        is_flush = any(suits.count(suit) >= 5 for suit in set(suits))
        rank_counter = Counter(rank_values)
        ordered = sorted(rank_counter.items(), key=lambda x: (-x[1], -x[0]))

        unique_ranks = sorted(set(rank_values), reverse=True)
        straight_high = 0
        if set([14, 2, 3, 4, 5]).issubset(set(rank_values)):
            straight_high = 5
        else:
            for i in range(len(unique_ranks) - 4):
                if unique_ranks[i] - unique_ranks[i + 4] == 4:
                    straight_high = unique_ranks[i]
                    break

        if is_flush:
            flush_cards = [c for c in cards if suits.count(c[-1]) >= 5]
            flush_ranks = sorted([values[c[:-1]] for c in flush_cards], reverse=True)
            if set([14, 2, 3, 4, 5]).issubset(set(flush_ranks)):
                return 8
            for i in range(len(flush_ranks) - 4):
                if flush_ranks[i] - flush_ranks[i + 4] == 4:
                    return 8

        if ordered[0][1] == 4:
            return 7

        if ordered[0][1] == 3 and ordered[1][1] >= 2:
            return 6

        if is_flush:
            return 5

        if straight_high:
            return 4

        if ordered[0][1] == 3:
            return 3

        if ordered[0][1] == 2 and ordered[1][1] == 2:
            return 2

        if ordered[0][1] == 2:
            return 1

        return 0

    def determine_winner(self):
        active_players = [p for p in self.players if p.active]
        hand_ranks = []
        for p in active_players:
            combined = p.hand + self.community_cards
            rank = self.evaluate_hand(combined)
            p.hand_rank = rank
            hand_ranks.append((rank, p))

        hand_ranks.sort(key=lambda x: x[0], reverse=True)
        best_rank = hand_ranks[0][0]
        winners = [p for (rank, p) in hand_ranks if rank == best_rank]
        self.distribute_pot(winners)


class RandomPlayer:
    def __init__(self, name="Random", chips=1000):
        self.name = name
        self.chips = chips
        self.hand = []
        self.active = True
        self.current_bet = 0

    def receive_card(self, card):
        self.hand.append(card)

    def take_action(self, current_bet, community_cards):
        action = random.choice(["fold", "fold","call", "raise"])
        if action == "fold":
            return ("fold", self.current_bet)
        elif action == "call":
            return ("call", current_bet)
        else:
            return ("raise", current_bet + 20)


class AllInPlayer:
    def __init__(self, name="AllInBot", chips=1000):
        self.name = name
        self.chips = chips
        self.hand = []
        self.active = True
        self.current_bet = 0

    def receive_card(self, card):
        self.hand.append(card)

    def take_action(self, current_bet, community_cards):
        return ("raise", self.chips)

class PPOPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPOPolicy, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


class PPOPlayer:
    def __init__(self, name="PPOBot", chips=1000):
        self.name = name
        self.chips = chips
        self.hand = []
        self.active = True
        self.current_bet = 0
        self.all_in_status = 0
        self.policy = PPOPolicy(input_dim=15, output_dim=3)

    def receive_card(self, card):
        self.hand.append(card)

    def take_action(self, current_bet, community_cards):
        obs = torch.randn(15)
        logits = self.policy(obs)
        action = torch.argmax(logits).item()
        if self.chips == 0:
            self.active = False
            self.all_in_status = 2
            return ("fold", 0)
        if action == 0:
            self.all_in_status = 2
            return ("fold", self.current_bet)
        elif action == 1:
            return ("call", current_bet)
        else:
            raise_amt = current_bet*2
            if self.chips <= raise_amt:
                self.all_in_status = 1
                raise_amt = self.chips
            return ("raise", raise_amt)


class StatsPlayer:
    def __init__(self, name="StatsBot", chips=1000):
        self.name = name
        self.chips = chips
        self.hand = []
        self.active = True
        self.current_bet = 0
        self.all_in_status = 0

    def receive_card(self, card):
        self.hand.append(card)

    def take_action(self, current_bet, community_cards):
        StatsBot.hand = [(c[:-1], c[-1]) for c in self.hand]
        StatsBot.chip_count = self.chips

        if self.chips == 0:
            self.active = False
            self.all_in_status = 2
            return ("fold", 0)

        if len(community_cards) == 0:
            move = StatsBot.preflop_action(
                numPlayers=4,
                playerIndex=0,
                been_raised=lambda: False,
                are_limpers=lambda: False
            )
        else:
            move = StatsBot.postflop_action(
                numPlayers=4,
                playerIndex=0,
                current_bet=current_bet,
                community_cards=[(c[:-1], c[-1]) for c in community_cards],
                pot_size=100
            )

        if move == "R":
            if self.chips <= current_bet + 20:
                self.all_in_status = 1
                return ("raise", self.chips)
            return ("raise", self.current_bet + 20)
        elif move == "C":
            return ("call", current_bet)
        else:
            d = random.random()
            if d < 0.3:
                self.all_in_status = 2
                return ("fold", self.current_bet)
            else:
                return ("call", current_bet)


def main():
    players = [
        StatsPlayer("StatsBot"),
        RandomPlayer("RandomBot"),
        PPOPlayer("PPOBot"),
       # AllInPlayer("AllInBot")
    ]
    env = PokerEnv(players)

    num_hands = 50
    chip_history = {player.name: [] for player in players}
    action_counts = {player.name: {"fold": 0, "call": 0, "raise": 0} for player in players}
    prev_chips = {player.name: player.chips for player in players}
    acted_this_hand = set()

    for hand in range(num_hands):
        print(f"\n===== Hand {hand + 1} =====")
        # Record chip counts at the start of the hand
        for player in players:
            chip_history[player.name].append(player.chips)
        obs, _ = env.reset()
        done = False
        acted_this_hand.clear()

        while not done:
            statuses = [getattr(p, 'all_in_status', 0) for p in players if p.active]
            if all(s != 0 for s in statuses):
                env.deal_remaining_board()
                env.determine_winner()
                break

            current_player = env.players[env.action_pointer]
            if current_player.active:
                action, amount = current_player.take_action(env.current_bet, env.community_cards)
                acted_this_hand.add(current_player.name)
                if action == "fold":
                    action_counts[current_player.name]["fold"] += 1
                elif action == "call":
                    action_counts[current_player.name]["call"] += 1
                elif action == "raise":
                    action_counts[current_player.name]["raise"] += 1
            obs, reward, done, _, _ = env.step(None)

        active_names = [p.name for p in env.players if p.active]
        for name in active_names:
            if name not in acted_this_hand:
                action_counts[name]["call"] += 1

        for player in players:
            if player.chips <= 0 and prev_chips[player.name] > 0:
                action_counts[player.name]["fold"] += 1
            prev_chips[player.name] = player.chips
            if hasattr(player, 'all_in_status'):
                player.all_in_status = 0

    plt.figure(figsize=(10, 6))
    for name, chips in chip_history.items():
        plt.plot(range(1, num_hands + 1), chips, label=name)

    plt.xlabel("Hand Number")
    plt.ylabel("Chip Count")
    plt.title("Chip Counts Over Hands")
    plt.legend()
    plt.grid()
    plt.show()

    print("\n=== Player Action Summary ===")
    for player, actions in action_counts.items():
        print(f"{player}: Folds: {actions['fold']}, Calls: {actions['call']}, Raises: {actions['raise']}")


if __name__ == "__main__":
    main()

