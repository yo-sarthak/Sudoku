#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
from typing import Any, Union, Tuple
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai
import copy
import math


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()

    def compute_best_move(self, game_state: GameState) -> None:

        class MiniGameState:
            """
            Class holding game_state info without Initial board
            """

            def __init__(
                    self, b: SudokuBoard,
                    tml: list[TabooMove],
                    sl: list[int]
            ):
                self.board = b
                self.taboo_moves = tml
                self.scores = sl

        def get_random_set_element(set_x: set) -> Any:
            """ Given a set, return a random element from that set without changing the set
            with the fastest Python implementation.

            :param set_x: Given set to return a random element of
            :return: Random element of set_x
            """
            elem = set_x.pop()
            set_x.add(elem)

            return elem

        def does_move_complete_column(
                state: GameState,
                i: int,
                j: int,
        ) -> Tuple[bool, set]:
            """ Checks whether a move to the given field could complete a column
            and updates the illegal numbers set on the column

            :param state: object of the current state of the game
            :param i: row index of the move
            :param j: column index of the move
            :return: tuple of bool value representing whether filling the cell
             gives the player a point for that column and the set of illegal
             numbers from that column
            """
            illegal_numbers_set = {0}
            is_col_filled = True
            # loop over the different rows of the column of the given cell,
            # and check if there are any other empty cells.
            for row_index in range(state.board.N):

                # Skip the current cell (We already know it's empty)
                if i == row_index:
                    continue

                # Add the value of the cells
                # in that column to the set of illegal numbers.
                cell = state.board.get(row_index, j)
                illegal_numbers_set.add(cell)

                # If a cell is empty, then filling another empty cell
                # will not result in filling the column.
                if cell == state.board.empty:
                    is_col_filled = False

            return is_col_filled, illegal_numbers_set

        def does_move_complete_row(
                state: GameState,
                i: int,
                j: int,
        ) -> Tuple[bool, set]:
            """ Checks whether a move to the given field could complete a row
            and also returns illegal numbers in that row

            :param state: object of the current state of the game
            :param i: row index of the move
            :param j: column index of the move
            :return: Tuple of the bool value representing
             whether filling the cell gives the player a point for that row.
            """
            illegal_numbers_set = {0}
            is_row_filled = True

            # loop over the different columns of the row of the given cell,
            # and check if there are any other empty cells.
            for col_index in range(state.board.N):

                # Skip the current cell (We already know it's empty)
                if j == col_index:
                    continue

                # Add the value of the cells
                # in that row to the set of illegal numbers.
                cell = state.board.get(i, col_index)
                illegal_numbers_set.add(cell)

                # If a cell is empty, then filling another empty cell
                # will not result in filling the row.
                if cell == state.board.empty:
                    is_row_filled = False

            return is_row_filled, illegal_numbers_set

        def completes_subgrid(
                state: GameState,
                i: int,
                j: int,
        ) -> Tuple[bool, set]:
            """ Checks whether a move to the given field could complete
             a subgrid and returns found illegal numbers in that subgrid

            :param state: object of the current state of the game
            :param i: row index of the move
            :param j: column index of the move
            :return: tuple of bool value representing whether filling
            the cell gives the player a point for that subgrid and the set of
            illegal (taken) numbers from that subgrid
            """
            # boolean that represents if filling the cell
            # gives the player a point for that subgrid.
            is_subgrid_filled = True
            illegal_numbers_set = {0}

            # Compute the starting and ending row
            # and column indices for the subgrid of the cell.
            row_subgrid_start = (i // state.board.m) * state.board.m
            row_subgrid_end = ((i // state.board.m) + 1) * state.board.m
            col_subgrid_start = (j // state.board.n) * state.board.n
            col_subgrid_end = ((j // state.board.n) + 1) * state.board.n

            # loop over the cells of the subgrid
            # and check if there are any other empty cells.
            for subgrid_row_index in range(row_subgrid_start, row_subgrid_end):

                for subgrid_col_index in range(col_subgrid_start,
                                               col_subgrid_end):

                    # Skip the current cell (We already know it's empty)
                    if i == subgrid_row_index and j == subgrid_col_index:
                        continue

                    # Add the numbers in the cell's subgrid
                    # to the illegal_numbers set
                    cell = state.board.get(
                        subgrid_row_index, subgrid_col_index)
                    illegal_numbers_set.add(cell)

                    # If a cell in the subgrid is empty,
                    # then filling the given cell
                    # will result in no points for that subgrid
                    if cell == state.board.empty:
                        is_subgrid_filled = False

            return is_subgrid_filled, illegal_numbers_set

        def find_best_candidate_move(dict_of_moves: dict[int: tuple]) -> tuple:
            """ Returns a tuple with coordinates of one of the best moves from the given move dictionary

            :param dict_of_moves: dictionary with scores as index and move tuples as values
            :return: tuple of one of the best scores in the given dictionary
            """
            # find the highest scoring move possible from the given dict
            for index in range(3, -1, -1):
                if dict_of_moves[f"score{index}"]:
                    return random.choice(dict_of_moves[f"score{index}"])

        def get_legal_move_list_from_dict(legal_move_dict: dict) -> list:
            """ Return a sorted-by-score list of moves from the given dict of legal moves with scores

            :param legal_move_dict: dictionary with scores as index and move tuples as values
            :return:
            """
            move_list = []
            # find the highest scoring move possible from the given dict
            # and add it to the list
            for index in range(3, -1, -1):
                if legal_move_dict[f"score{index}"]:
                    for move in legal_move_dict[f"score{index}"]:
                        move_list.append(move)

            return move_list

        def get_legal_moves(state: GameState, return_type: str) \
                -> Union[dict, list]:
            """ Returns list or dictionary of legal moves,
            depending on the requested return_type

            :param state: state of the game for which the moves are requested
            :param return_type: 'dict' or 'list' corresponding
             to the result type needed
            :return: Either a dictionary with scores as keys and corresponding
             moves as values if dict is requested,
             or simple list of legal moves otherwise
            """
            N = state.board.N

            # Initializing the dictionary that will contain the best moves.
            # Each of the scores shown below will contain
            # a list of tuples of the form (i ,j , value, score)
            # The 'i' and 'j' are the row and column indices of the cell.
            # Value is the number the cell is to be filled with.
            # Score refers to how good a move is right now -in a greedy manner.
            # score 0 means that playing the given move will result in 0 points
            # score 1 means that playing the given move will result in 1 point
            # score 2 means that playing the given move will result in 3 points
            # score 3 means that playing the given move will result in 7 points

            best_move_dict = {
                'score0': [],
                'score1': [],
                'score2': [],
                'score3': [],
            }

            # The set of all possible numbers given an empty board.
            possible_numbers = {
                number for number in range(state.board.N + 1)}

            # Loop over all cells in the board, find empty cells and
            # compute cell_info for each one.
            for i in range(N):
                for j in range(N):
                    cell = state.board.get(i, j)
                    if cell == state.board.empty:

                        # Set of numbers that we CAN'T use for this empty cell.
                        illegal_numbers = {0}
                        elem = None

                        # check whether filling this cell will
                        # complete a column, row, or subgrid
                        # note: We pass the illegal_numbers set by reference
                        # so we can compute the numbers already played
                        # in a column, row and subgrid,
                        # which saves an extra iteration later
                        will_fill_column, illegal_numbers_col = \
                            does_move_complete_column(state, i, j)
                        will_fill_row, illegal_numbers_row = \
                            does_move_complete_row(state, i, j)
                        will_fill_subgrid, illegal_numbers_subgrid = \
                            completes_subgrid(state, i, j)

                        illegal_numbers = illegal_numbers.union(
                            illegal_numbers_col,
                            illegal_numbers_row,
                            illegal_numbers_subgrid
                        )

                        score = will_fill_column + will_fill_row + \
                            will_fill_subgrid

                        # The set of legal numbers for a given cell, is
                        # the set of all possible moves - set of illegal moves
                        legal_numbers = possible_numbers.difference(illegal_numbers)

                        try:
                            elem = get_random_set_element(legal_numbers)
                        except KeyError:
                            continue

                        # add the move to the best move list
                        # (if it's not a TabooMove)
                        if not (TabooMove(i, j, elem) in state.taboo_moves):
                            best_move_dict[f"score{score}"].append(
                                (i, j, elem, score)
                            )

            if return_type == "dict":
                return best_move_dict
            elif return_type == "list":
                return get_legal_move_list_from_dict(best_move_dict)

        def compute_greediest_move(state: GameState) -> tuple:
            """ Computes best move according to the greedy player's strategy

            :param state: current state of the game
            :return: tuple of the legal moves (with scores as keys)
             and the best move according to the greedy strategy
            """
            best_move_dict = get_legal_moves(state, "dict")

            return best_move_dict, find_best_candidate_move(best_move_dict)

        def is_game_finished(b: SudokuBoard) -> bool:
            """ Checks whether game is finished on the given SudokuBoard

            :param b: SudokuBoard to be checked
            :return: False if there is at least one field empty, True otherwise
            """
            for i in range(game_state.board.N):
                for j in range(game_state.board.N):
                    if b.get(i, j) == SudokuBoard.empty:
                        return False
            return True

        def evaluate(t: MiniGameState, player_number):
            """ Evaluates the current game score for 'our' AI agent
             based on the scores of the 2 players

            :param t: MiniGameState object containing scores of the 2 players
            :param player_number: number of 'our' AI agent
            :return: Evaluation of the game score for 'our' AI agent
            """
            return t.scores[player_number - 1] - t.scores[2 - player_number]

        def get_child_states(
                t: Union[MiniGameState, GameState],
                our_agent: bool,
                our_player_number: int
        ) -> list[MiniGameState]:
            """ Returns list of possible child states of the given game state

            :param t: current game state as MiniGameState object
            :param our_agent: boolean value whether it is the move of 'our' AI
            :param our_player_number: player number of the our agent
            :return:
            """
            possible_moves_to_play = get_legal_moves(t, "list")
            possible_tree_nodes = []

            for move in possible_moves_to_play:
                # We're making a copy.deepcopy of game_state
                # to avoid editing the original state of the game
                new_tree = copy.deepcopy(t)

                # Update board state for new move
                new_tree.board.put(move[0], move[1], move[2])

                # Update score for new move
                if our_agent:
                    new_tree.scores[our_player_number - 1] += move_score(move)
                else:
                    new_tree.scores[2 - our_player_number] += move_score(move)

                possible_tree_nodes.append(new_tree)

            return possible_tree_nodes

        def minimax(
                t: MiniGameState,
                depth: int,
                is_our_agent: bool,
                alpha: Union[float, int],
                beta: Union[float, int],
                our_player_number: int
        ) -> int:
            """
            :param t: The current tree node
            :param depth: depth till which minimax recurse
            :param is_our_agent: bool value whether it is the move of 'our' AI
            :param our_player_number: integer (1 or 2)
             indicating our AI's player number
            :return: return the max or min score depending on the turn
            """
            # Check whether game is finished or we should finish evaluating
            if depth == 0 or is_game_finished(t.board):
                return evaluate(t, our_player_number)

            if is_our_agent:  # If it's the turn of the maximising player
                maxEval = -math.inf
                for child_t in get_child_states(t, True, our_player_number):
                    eval = minimax(child_t, depth - 1, False,
                                   alpha, beta, our_player_number)
                    maxEval = max(maxEval, eval)
                    alpha = max(alpha, maxEval)
                    if beta <= alpha:
                        break
                return maxEval
            else:  # If it's the turn of the minimising player
                minEval = math.inf
                for child_t in get_child_states(t, False, our_player_number):
                    eval = minimax(child_t, depth - 1, True,
                                   alpha, beta, our_player_number)
                    minEval = min(minEval, eval)
                    beta = min(beta, minEval)
                    if beta <= alpha:
                        break
                return minEval

        def create_move_from_tuple(move_tuple: tuple):
            """ Wrapper to create a Move object from the tuple representation

            :param move_tuple: tuple with i coordinate at position 0,
             j coordinate at position 1 and its value at index 2
            :return: Move object created from the tuple
            """
            return Move(move_tuple[0], move_tuple[1], move_tuple[2])

        def move_score(move_tuple: tuple) -> int:
            """Determine score gain based on the move

            :param move_tuple: tuple representations of the move
            :return: score of the given move
            """
            if move_tuple[3] == 0:
                return 0
            if move_tuple[3] == 1:
                return 1
            if move_tuple[3] == 2:
                return 3
            if move_tuple[3] == 3:
                return 7

        # We first propose the greediest move possible.
        legal_moves_dict, greediest_move_tuple = compute_greediest_move(game_state)

        proposed_move = create_move_from_tuple(greediest_move_tuple)
        self.propose_move(proposed_move)

        # Now we initialize the arguments for our minimax algorithm
        legal_moves_list = get_legal_move_list_from_dict(legal_moves_dict)
        cur_player = game_state.current_player()
        moves_after_AI_turn = []
        alpha = -math.inf
        beta = math.inf
        for legal_move in legal_moves_list:
            # The board after playing this move is derived from executing the move on the current board
            # We make a copy.deepcopy since otherwise we will also modify the original gamestate when we use .put!
            # See: https://www.programiz.com/python-programming/shallow-deep-copy for reference
            new_game_state = MiniGameState(copy.deepcopy(game_state.board), copy.deepcopy(
                game_state.taboo_moves), copy.deepcopy(game_state.scores))
            # Update board state for new move
            new_game_state.board.put(legal_move[0], legal_move[1],
                                     legal_move[2])
            # Update score for new move
            new_game_state.scores[cur_player -
                                  1] += move_score(legal_move)
            moves_after_AI_turn.append((new_game_state, legal_move))

        # start with an initial depth of 3.
        depth = 3
        while True:
            # Evaluate the nodes (new gamestates) we create by making one of the moves in legal_moves_list
            # Do this by using minimax with minimax_layers on the gamestate produced
            # by the moves our AI can play (Here we presented board to the opponent after playing our move) to
            # determine their max_val
            max_val = -math.inf
            selected_move = None
            for move in moves_after_AI_turn:
                value = minimax(move[0], depth, False, alpha, beta, cur_player)
                if value >= max_val:
                    max_val = value
                    selected_move = move[1]
            if selected_move is not None:
                self.propose_move(create_move_from_tuple(selected_move))
            depth += 1