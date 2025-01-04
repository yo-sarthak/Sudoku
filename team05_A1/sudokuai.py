import random
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given Sudoku configuration.
    """

    def __init__(self):
        super().__init__()

    def compute_best_move(self, game_state: GameState) -> None:
        N = game_state.board.N
        board = game_state.board

        # Check if a move is valid
        def is_valid_move(board, i, j, value):
            return (
                board.get((i, j)) == SudokuBoard.empty
                and TabooMove((i, j), value) not in game_state.taboo_moves
                and (i, j) in game_state.player_squares()
            )

        # Get row, column, and block values for a given move
        def get_constraints(board, move: Move):
            row_values = [board.get((move.square[0], j)) for j in range(
                N) if board.get((move.square[0], j)) != SudokuBoard.empty]
            col_values = [board.get((i, move.square[1])) for i in range(
                N) if board.get((i, move.square[1])) != SudokuBoard.empty]

            block_i = (move.square[0] // board.m) * board.m
            block_j = (move.square[1] // board.n) * board.n
            block_values = [
                board.get((i, j))
                for i in range(block_i, block_i + board.m)
                for j in range(block_j, block_j + board.n)
                if board.get((i, j)) != SudokuBoard.empty
            ]
            return row_values, col_values, block_values

        # Evaluate a move based on its impact
        def evaluate_move(board, move):
            row_values, col_values, block_values = get_constraints(board, move)
            solves_row = len(row_values) == N - 1
            solves_col = len(col_values) == N - 1
            solves_block = len(block_values) == N - 1
            score = 0
            if solves_row:
                score += 1
            if solves_col:
                score += 1
            if solves_block:
                score += 3 if solves_row or solves_col else 7
            return score

        # Generate all legal moves
        def get_legal_moves(board):
            moves = []
            for i in range(N):
                for j in range(N):
                    for value in range(1, N + 1):
                        if is_valid_move(board, i, j, value):
                            row_values, col_values, block_values = get_constraints(
                                board, Move((i, j), value))
                            if value not in row_values + col_values + block_values:
                                moves.append(Move((i, j), value))
            return moves

        # Minimax with alpha-beta pruning
        def minimax(board, depth, current_score, alpha, beta, is_maximizing):
            legal_moves = get_legal_moves(board)
            if depth == 0 or not legal_moves:
                return None, current_score

            if is_maximizing:
                max_eval = float('-inf')
                best_move = None
                for move in legal_moves:
                    score = evaluate_move(board, move)
                    current_score += score
                    board.put(move.square, move.value)
                    _, eval = minimax(board, depth - 1,
                                      current_score, alpha, beta, False)
                    current_score -= score
                    board.put(move.square, SudokuBoard.empty)

                    if eval > max_eval:
                        max_eval = eval
                        best_move = move
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
                return best_move, max_eval
            else:
                min_eval = float('inf')
                best_move = None
                for move in legal_moves:
                    score = evaluate_move(board, move)
                    current_score -= score
                    board.put(move.square, move.value)
                    _, eval = minimax(board, depth - 1,
                                      current_score, alpha, beta, True)
                    current_score += score
                    board.put(move.square, SudokuBoard.empty)

                    if eval < min_eval:
                        min_eval = eval
                        best_move = move
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
                return best_move, min_eval

        # Determine if the current player is maximizing or minimizing
        is_maximizing = game_state.current_player == 2

        # Start with a greedy move as a fallback
        legal_moves = get_legal_moves(board)
        if legal_moves:
            self.propose_move(random.choice(legal_moves))

        # Perform iterative deepening with Minimax
        for depth in range(1, N * N + 1):
            best_move, _ = minimax(board, depth, 0, float(
                '-inf'), float('inf'), is_maximizing)
            if best_move:
                self.propose_move(best_move)