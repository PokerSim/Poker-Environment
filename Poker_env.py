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
        self.action_pointer = 0
        self.round_stage = "pre-flop"

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

    
