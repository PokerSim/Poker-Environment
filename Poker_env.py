import random
import numpy as np
import pokerkit as pk

class Poker_env:
    def __init__(self, players):
        self.players = players
        self.pot = 0
        self.big_blind = 20
        self.small_blind = 10
        self.dealer = 0 # Possibly use a pointer
        self.deck = self.initialize_deck()
        self.community_cards = []
        self.burn_cards = []
        self.action_pointer = 0
        self.round_stage = "pre-flop"
        self.current_bet = 0

    def initialize_deck(self):
        suits = ["s", "h", "d", "c"]
        ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
        return [rank + suit for suit in suits for rank in ranks]

    def deal_hands(self):
        random.shuffle(self.deck)
        for i in range (2):
            for player in self.players:
                if player.active:
                    player.receive_card(self.deck.pop())

    def post_blinds(self):
        small_blind_player = self.get_next_active_player(self.dealer+1)
        big_blind_player = self.get_next_active_player(self.dealer+2)
        small_blind_player.chips -= self.small_blind
        big_blind_player.chips -= self.big_blind
        self.pot += self.small_blind + self.big_blind

    def betting_round(self):
        pass
    
    def deal_board(self):
        if self.round_stage == "flop":
            self.burn_cards.append(self.deck.pop())
            for i in range(3):
                self.community_cards.append(self.deck.pop())
            
        elif self.round_stage == "turn":
            self.burn_cards.append(self.deck.pop())
            self.community_cards.append(self.deck.pop())

        elif self.round_stage == "river":
            self.burn_cards.append(self.deck.pop())
            self.community_cards.append(self.deck.pop())
    def get_next_active_player(self, dealer):
        pass


