#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import time
import copy
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()

    # N.B. This is a very naive implementation.
    def compute_best_move(self, game_state: GameState) -> None:
        N = game_state.board.N
        
        # Check whether a cell is empty, a value in that cell is not taboo, and that cell is allowed
        def possible(i, j, value, game_state_check):
            return game_state_check.board.get((i, j)) == SudokuBoard.empty \
                   and not TabooMove((i, j), value) in game_state_check.taboo_moves \
                       and (i, j) in game_state_check.player_squares()

        def generate_all_moves(game_state):
            return [Move((i, j), value) for i in range(N) for j in range(N)
                     for value in range(1, N+1) if possible(i, j, value, game_state)]
        
        # Calculate numerical evaluation of state, calculated from scores from GameState
        def evaluate(game_state: GameState) -> int:
            scores = game_state.scores
            return scores[0] - scores[1]  # Player 1's advantage
    
        # Function to apply a move to a game state and return the new state
        def apply_move(game_state: GameState, move):
            new_state = copy.deepcopy(game_state)
            print("Current player is ", new_state.current_player)
            new_state.board.put(move.square, move.value)
            new_state.moves.append(move)
            return new_state
        
         # Return list of valid child game states
        def get_children(game_state: GameState):
            moves = generate_all_moves(game_state)
            return [apply_move(game_state, move) for move in moves]
        

        children = get_children(game_state)
        #print(children[-1])
        all = generate_all_moves(children[-1])
        for i in all:
            print(i)
        '''
        children_of_children=[]
        for i in children:
            i.current_player = 3 - i.current_player
            children_of_children.append(get_children(i))
        count=0
        for i in children_of_children:
            for j in i:
                count+=1
        '''
        print(game_state.current_player)

        #print(children_of_children[-1][-1])



        '''
        # Implementing Minimax algorithm
        def minimax(game_state_minmax: GameState, depth: int, is_maximizing_player: bool):
            if depth == 0 or not generate_all_moves(game_state_minmax):
                return evaluate(game_state_minmax)
            
            children = get_children(game_state_minmax)

            if is_maximizing_player:
                max_eval = -float('inf')
                for child in children:
                    print(child)
                    max_eval = max(max_eval, minimax(child, depth-1, False))
                return max_eval
            else:
                min_eval = float('inf')
                for child in children:
                    print(child)
                    min_eval = min(min_eval, minimax(child, depth-1, True))
                return min_eval

        all_moves = generate_all_moves(game_state)
        best_move = None
        best_value = -float("inf")
        
        #print("Evaluating possible moves:")
        for move in all_moves:
            #print(f"Checking move: {move}")
            # Apply the move and evaluate using minimax
            new_state = apply_move(game_state, move)
            eval_score = minimax(new_state, depth=3, is_maximizing_player=False)
            #print(f"Move {move} has evaluation score: {eval_score}")
            # Update the best move if this move is better
            if eval_score >= best_value:
                best_value = eval_score
                best_move = move

        # Check if a valid move was found
        if not best_move:
            raise ValueError("No valid moves available!")

        # Propose the best move
        #print(f"Best move is {best_move} with score {best_value}")
        self.propose_move(best_move)
        '''