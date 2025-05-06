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
