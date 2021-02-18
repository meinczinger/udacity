from sample_players import DataPlayer
import random
import pickle
from isolation import DebugState
from collections import defaultdict, Counter

MIN_DEPTH = 3
MAX_DEPTH = 3
MIN_NR_OF_GAMES = 3
_WIDTH = 11


class CustomPlayer(DataPlayer):
    """ Implement an agent using any combination of techniques discussed
    in lecture (or that you find online on your own) that can beat
    sample_players.GreedyPlayer in >80% of "fair" matches (see tournament.py
    or readme for definition of fair matches).

    Implementing get_action() is the only required method, but you can add any
    other methods you want to perform minimax/alpha-beta/monte-carlo tree search,
    etc.

    **********************************************************************
    NOTE: The test cases will NOT be run on a machine with GPU access, or
          be suitable for using any other machine learning techniques.
    **********************************************************************
    """

    def get_action(self, state, simulation=False, use_book=False):
        """ Choose an action available in the current state

        See RandomPlayer and GreedyPlayer for examples.

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller is responsible for
        cutting off the function after the search time limit has expired.

        **********************************************************************
        NOTE: since the caller is responsible for cutting off search, calling
              get_action() from your own code will create an infinite loop!
              See (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        if simulation:
            # Simulation - return action instead of putting it to queue
            return self.get_simulated_action(state)
        else:
            if not use_book:
                # Normal mode, play by using alpha-beta pruning
                if state.ply_count < 2:
                    self.queue.put(random.choice(state.actions()))
                else:
                    self.find_actions(state)
            else:
                # Play from the book
                if state.ply_count < 8 and state.board in self.data:
                    # Take the action with the highest win ratio
                    action = max(self.data[state.board], key=lambda k: self.get_ratio(self.data[state.board][k]))
                    # Shall we use this action?
                    if self.use_action(self.data[state.board][action]):
                        print("Using action, ply:", state.ply_count, "action:", action)
                        self.queue.put(action)
                    else:
                        if state.ply_count < 2:
                            if state.ply_count == 0:
                                self.random_action(state)
                            else:
                                self.random_action(state)

                        else:
                            self.find_actions(state)
                else:
                    # State not found in the book, play without the book
                    if state.ply_count < 2:
                        self.random_action(state)
                    else:
                        self.find_actions(state)

    """ Get the best (with highest win/loss ratio) first action from the book """
    def get_best_first_actions(self, state):
        d = DebugState()
        actions = [a for a in self.data[state.board]]
        actions.sort(reverse=True, key=lambda k: self.get_ratio(self.data[state.board][k]))
        for a in actions:
            print(a, self.data[state.board][a], self.data[state.board][a]['wins'] +
                  self.data[state.board][a]['losses'], self.get_ratio(self.data[state.board][a]),
                  d.from_state(state).ind2xy(a))

    """ Decide whether an action from the opening book should be used """
    @staticmethod
    def use_action(results):
        # if less than MIN_NR_OF_GAMES played with this position, return 0
        if (results['wins'] + results['losses']) < MIN_NR_OF_GAMES:
            return False
        else:
            # if less wins than losses, return 0
            if results['wins'] < results['losses']:
                return False
        return True

    """ For an action from the book, get the ration of wins/losses """
    @staticmethod
    def get_ratio(results):
        # if less than MIN_NR_OF_GAMES played with this position, return 0
        if (results['wins'] + results['losses']) < MIN_NR_OF_GAMES:
            return 0
        else:
            return results['wins'] / (results['wins'] + results['losses'])

    """ Put a random action to the queue """
    def random_action(self, state):
        self.queue.put(random.choice(state.actions()))

    """ Find action by running alpha-beta pruning with iterative deepening"""
    def find_actions(self, state):
        for i in range(MIN_DEPTH, MAX_DEPTH + 1):
            self.queue.put(self.alpha_beta(state, depth=i))

    """ Get an action if the agent is used in simulation mode """
    def get_simulated_action(self, state):
        if state.ply_count < 2:
            # create the playbook
            book = defaultdict()
            if state.ply_count == 0:
                action = random.choice(state.actions())
            else:
                action = random.choice(state.actions())
        else:
            # retrieve the book from the context
            book = self.context
            # get action
            action = self.alpha_beta(state, depth=MAX_DEPTH)

        # for the first 8 ply's:
        if state.ply_count < 8:
            book[state.board] = action
            self.context = book

        return action

    """ During simulation, state/action combination were stored
    After the end of the game, the input parameters 'won' tells whether the game was won or lost.
    Based on that the state/action combinations counters (wins/losses) is updated 
    """
    def store_book(self, won):
        book = self.context
        if book is not None:
            data = self.data
            if data is None:
                data = defaultdict()
            else:
                data = self.data
            # update book with result
            for k, v in book.items():
                if k not in data:
                    data[k] = {v: {'wins': 0, 'losses': 0}}
                else:
                    if v not in data[k]:
                        data[k][v] = {'wins': 0, 'losses': 0}
                if won:
                    data[k][v]['wins'] += 1
                else:
                    data[k][v]['losses'] += 1

            with open("data.pickle", 'wb') as f:
                pickle.dump(data, f)

    """ Alpha-beta pruning search """
    def alpha_beta(self, state, depth):
        alpha = -float("inf")
        beta = float("inf")
        return max(state.actions(), key=lambda x: self.beta(state.result(x), depth - 1, alpha, beta))

    """ Alpha search """
    def alpha(self, state, depth, alpha, beta):
        if state.terminal_test():
            return state.utility(self.player_id)
        if depth <= 0:
            return self.score(state)
        value = float("-inf")
        for action in state.actions():
            value = max(value, self.beta(state.result(action), depth - 1, alpha, beta))
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value

    """ Beta search """
    def beta(self, state, depth, alpha, beta):
        if state.terminal_test():
            return state.utility(self.player_id)
        if depth <= 0:
            return self.score(state)
        value = float("inf")
        for action in state.actions():
            value = min(value, self.alpha(state.result(action), depth - 1, alpha, beta))
            if value <= alpha:
                return value
            beta = min(beta, value)
        return value

    """ Get the score for a non-terminal state by applying heuristics"""
    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return self.score_liberties(own_liberties, opp_liberties) + \
               self.score_distance_from_center(state, own_loc, opp_loc)

    """ Heuristic based on the number of liberties """
    @staticmethod
    def score_liberties(own_liberties, opp_liberties):
        return len(own_liberties) - len(opp_liberties)

    """ Heuristic based on the distance of the agent's location from the center """
    def score_distance_from_center(self, state, own_loc, opp_loc):
        own_distance_from_center = self.distance_from_the_center(int(own_loc)) / 5.0
        opp_distance_from_center = self.distance_from_the_center(int(opp_loc)) / 5.0
        return -own_distance_from_center + opp_distance_from_center

    """ Convert location to x, y coordinates """
    @staticmethod
    def ind2xy(ind):
        """ Convert from board index value to xy coordinates

        The coordinate frame is 0 in the bottom right corner, with x increasing
        along the columns progressing towards the left, and y increasing along
        the rows progressing towards teh top.
        """
        return ind % (_WIDTH + 2), ind // (_WIDTH + 2)

    """ Compute the manhatten distance of the location from the center """
    def distance_from_the_center(self, loc):
        x, y = self.ind2xy(loc)
        manhatten_distance = abs(5 - x) + abs(4 - y)
        return manhatten_distance
