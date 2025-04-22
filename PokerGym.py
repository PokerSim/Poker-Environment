import numpy as np
import random
import gym
from gym import spaces

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

        self.action_space = spaces.Discrete(3)  # Fold, Call, Raise
        self.observation_space = spaces.Box(low=0, high=1, shape=(107,), dtype=np.float32)

    def initialize_deck(self):
        suits = ["s", "h", "d", "c"]
        ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
        return [rank + suit for suit in suits for rank in ranks]

    def shuffle_deck(self):
        random.shuffle(self.deck)

    def deal_hands(self):
        self.shuffle_deck()
        for _ in range(2):
            for player in self.players:
                if player.active:
                    player.receive_card(self.deck.pop())

    def post_blinds(self):
        small_blind_player = self.get_next_active_player(self.dealer + 1)
        big_blind_player = self.get_next_active_player(self.dealer + 2)
        small_blind_player.chips -= self.small_blind
        big_blind_player.chips -= self.big_blind
        small_blind_player.current_bet = self.small_blind
        big_blind_player.current_bet = self.big_blind
        self.pot += self.small_blind + self.big_blind
        self.current_bet = self.big_blind

    def deal_community_cards(self):
        if self.round_stage == "flop":
            self.burn_cards.append(self.deck.pop())
            self.community_cards.extend([self.deck.pop() for _ in range(3)])
        elif self.round_stage in ["turn", "river"]:
            self.burn_cards.append(self.deck.pop())
            self.community_cards.append(self.deck.pop())

    def step(self, action):
        player = self.players[self.action_pointer]

        if not player.active:
            self._advance_pointer()
            return self._get_obs(), 0, False, {}

        if self.action_pointer == self.agent_index:
            if action == 0:
                player.active = False
            elif action == 1:
                player.take_action(self.current_bet)
            elif action == 2:
                player.take_action(self.current_bet + 20)
        else:
            action_name, bet_amount = player.take_action(self.current_bet)
            if action_name == "fold":
                player.active = False
            else:
                added_amount = bet_amount - player.current_bet
                if added_amount > 0:
                    player.chips -= added_amount
                    self.pot += added_amount
                    self.current_bet = max(self.current_bet, bet_amount)
                player.current_bet = bet_amount

        self._advance_pointer()

        if self._betting_round_complete():
            self._advance_stage()

        active_players = [p for p in self.players if p.active]
        done = len(active_players) == 1 or self.round_stage == "showdown"

        reward = 0
        if done:
            winner = self.determine_winner()
            if winner == self.players[self.agent_index]:
                reward = self.pot
            else:
                reward = -self.players[self.agent_index].current_bet

        return self._get_obs(), reward, done, {}

    def _advance_pointer(self):
        start = self.action_pointer
        while True:
            self.action_pointer = (self.action_pointer + 1) % len(self.players)
            if self.players[self.action_pointer].active:
                break
            if self.action_pointer == start:
                break

    def _betting_round_complete(self):
        active = [p for p in self.players if p.active]
        return all(p.current_bet == self.current_bet for p in active)

    def _advance_stage(self):
        stages = ["pre-flop", "flop", "turn", "river", "showdown"]
        current_index = stages.index(self.round_stage)
        if current_index < len(stages) - 1:
            self.round_stage = stages[current_index + 1]
            if self.round_stage != "showdown":
                self.deal_community_cards()
                for p in self.players:
                    p.current_bet = 0
                self.current_bet = 0

    def reset(self):
        self.reset_hand()
        self.deal_hands()
        self.post_blinds()
        return self._get_obs()

    def reset_hand(self):
        self.pot = 0
        self.deck = self.initialize_deck()
        self.community_cards = []
        self.burn_cards = []
        self.current_bet = 0
        self.round_stage = "pre-flop"
        self.action_pointer = 0
        for player in self.players:
            player.hand = []
            player.active = True
            player.current_bet = 0
        self.calculate_dealer()

    def calculate_dealer(self):
        self.dealer = self.players.index(self.get_next_active_player(self.dealer + 1))

    def get_next_active_player(self, start_index):
        index = start_index % len(self.players)
        while not self.players[index].active:
            index = (index + 1) % len(self.players)
        return self.players[index]

    def determine_winner(self):
        best_score = -1
        winner = None
        for player in self.players:
            if player.active:
                score = sum(ord(c[0]) for c in player.hand)  # replace with real evaluation later
                if score > best_score:
                    best_score = score
                    winner = player
        return winner

    def _get_obs(self):
        agent = self.players[self.agent_index]
        hand_vec = self._encode_cards(agent.hand)
        board_vec = self._encode_cards(self.community_cards)
        scalar_vec = np.array([
            agent.chips / 1000,
            self.pot / 1000,
            self.current_bet / 100
        ])
        return np.concatenate([hand_vec, board_vec, scalar_vec])

    def _encode_cards(self, cards):
        suits = ["s", "h", "d", "c"]
        ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
        card_ids = [r + s for s in suits for r in ranks]
        card_map = {card: idx for idx, card in enumerate(card_ids)}
        vec = np.zeros(52)
        for card in cards:
            if card in card_map:
                vec[card_map[card]] = 1
        return vec
