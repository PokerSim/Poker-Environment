import random
import pokerkit as pk

class PokerEnv:
    def __init__(self, players):
        self.players = players
        self.pot = 0
        self.big_blind = 20
        self.small_blind = 10
        self.dealer = 0  # Dealer position
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

        self.pot += self.small_blind + self.big_blind
        self.current_bet = self.big_blind

    def betting_round(self):
        """Handles the betting round. Players act in turn until bets are equal."""
        active_players = [p for p in self.players if p.active]

        while True:
            player = active_players[self.action_pointer]
            action = player.take_action(self.current_bet)

            if action == "fold":
                player.active = False
            elif action == "call":
                player.chips -= (self.current_bet - player.current_bet)
                self.pot += (self.current_bet - player.current_bet)
            elif action.startswith("raise"):
                _, amount = action.split()
                amount = int(amount)
                if amount > self.current_bet:
                    self.current_bet = amount
                    player.chips -= (amount - player.current_bet)
                    self.pot += (amount - player.current_bet)

            # Move to next active player
            self.action_pointer = (self.action_pointer + 1) % len(active_players)
            if all(p.current_bet == self.current_bet for p in active_players if p.active):
                break  # Betting round ends when all active players have equal bets

    def deal_board(self):
        """Deals community cards based on the game stage."""
        self.burn_cards.append(self.deck.pop())
        if self.round_stage == "flop":
            for _ in range(3):
                self.community_cards.append(self.deck.pop())
        else:
            self.community_cards.append(self.deck.pop())

    def get_next_active_player(self, start_index):
        """Finds the next active player from a given position."""
        index = start_index % len(self.players)
        while not self.players[index].active:
            index = (index + 1) % len(self.players)
        return self.players[index]

    def calculate_dealer(self):
        """Moves the dealer button to the next active player."""
        self.dealer = self.players.index(self.get_next_active_player(self.dealer + 1))

    def determine_winner(self):
        """Determines the winner using pokerkit hand evaluation."""
        best_hand = None
        winner = None
        for player in self.players:
            if player.active:
                hand = pk.Hand(player.hand + self.community_cards)
                if best_hand is None or hand > best_hand:
                    best_hand = hand
                    winner = player
        return winner

    def reset_hand(self):
        """Resets the table for a new hand."""
        self.pot = 0
        self.deck = self.initialize_deck()
        self.community_cards = []
        self.burn_cards = []
        self.current_bet = 0
        for player in self.players:
            player.hand = []
            player.active = True
        self.calculate_dealer()

    def play_hand(self):
        """Plays through a full hand of poker."""
        self.deal_hands()
        self.post_blinds()
        self.betting_round()

        for stage in ["flop", "turn", "river"]:
            self.round_stage = stage
            self.deal_board()
            self.betting_round()
            if sum(p.active for p in self.players) == 1:
                break  # End hand if only one player remains

        winner = self.determine_winner()
        if winner:
            winner.chips += self.pot

        self.reset_hand()
