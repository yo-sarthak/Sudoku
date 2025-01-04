import random
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import copy
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
        player = game_state.current_player             

        '''
        Check if a move is valid, get constraints for it, and calculate
        score on placing move in board
        '''
        
        # Check if a move is valid
        def is_valid_move(game_state, i, j, value):
            return (
                game_state.board.get((i, j)) == SudokuBoard.empty
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
        def evaluate_score(board, move):
            row_values, col_values, block_values = get_constraints(board, move)
            solves_row = len(row_values) == N - 1
            solves_col = len(col_values) == N - 1
            solves_block = len(block_values) == N - 1
            sum = solves_row + solves_col + solves_block
            score = 0
            if sum == 0:
                score = 0
            elif sum == 1:
                score = 1
            elif sum == 2:
                score = 3
            elif sum == 3:
                score = 7
            return score

        '''
        Applying heuristics to remove naked twins from possible solution, 
        to limit number of overall possible moves and shorten potential minimax tree
        '''

        # Heuristic: Naked Pairs
        def naked_pairs(board, moves):
            possible_values = {}
            for move in moves:
                row_values, col_values, block_values = get_constraints(board, move)
                possibilities = set(range(1, N + 1)) - set(row_values + col_values + block_values)
                possible_values[move.square] = possibilities
 
            # Look for naked pairs in rows, columns, and blocks
            def eliminate_with_naked_pairs(group):
                naked_pairs = {}
                for square, possibilities in group.items():
                    if len(possibilities) == 2:
                        pair = tuple(possibilities)
                        naked_pairs[pair] = naked_pairs.get(pair, []) + [square]
                
                # Eliminate possibilities if a naked pair is found
                for pair, squares in naked_pairs.items():
                    if len(squares) == 2:
                        for square in group:
                            if square not in squares:
                                possible_values[square] -= set(pair)
            
            # Check rows and columns
            for i in range(N):
                row_group = {square: possible_values[square] for square in possible_values if square[0] == i}
                eliminate_with_naked_pairs(row_group)
                col_group = {square: possible_values[square] for square in possible_values if square[1] == i}
                eliminate_with_naked_pairs(col_group)
            
            # Check blocks
            for block_i in range(0, N, board.m):
                for block_j in range(0, N, board.n):
                    block_group = {
                        square: possible_values[square]
                        for square in possible_values
                        if (square[0] // board.m == block_i // board.m and square[1] // board.n == block_j // board.n)
                    }
                    eliminate_with_naked_pairs(block_group)
            filtered_moves = []
            for move in moves:
                if move.value in possible_values[move.square]:
                    filtered_moves.append(move)
            return filtered_moves
        
        # Apply heuristics to refine legal moves
        def apply_heuristics(moves):
            return naked_pairs(board, moves)
        
        '''
        Finding neighbors and new squares unlocked by move
        to identify moves that can block or conquer the board
        '''

        # Find all neighboring squares for a square
        def neighbors_square(square):
            neighbors = []
            for i in range(square[0] - 1, square[0] + 2):
                for j in range(square[1] - 1, square[1] + 2):
                    if (i<N) and (i>=0) and (j<N) and (j>=0):
                        neighbors.append((i,j))
            return neighbors
            
        
        # Find new squares unlocked if a value is placed in square
        def new_squares(game_state, square):
            all_squares = neighbors_square(square)
            squares = []
            for i in all_squares:

                # Check if move is in allowed squares of current player, or occupied by current player
                if i not in game_state.player_squares() and i not in game_state.occupied_squares():

                    #Check if move is in allowed squares of opposite player, or occupied by opposite player
                    game_state.current_player = 3 - game_state.current_player
                    if i not in game_state.player_squares() and i not in game_state.occupied_squares():
                        squares.append(i)        
                    game_state.current_player = 3 - game_state.current_player
            
            return squares

        def boundary(game_state):
            boundary = []
            for col in range(N):
                if game_state.current_player == 1:
                    square = (0,0)
                    for row in range(N):
                        if (row,col) in game_state.player_squares():
                            square = (row,col)
                else:
                    square = (0, N-1)
                    for row in range(N-1, -1, -1):
                        if(row,col) in game_state.player_squares():
                            square = (row,col)
                boundary.append(square)
            return boundary

        
        def get_boundary_moves(game_state):
            moves = []
            squares = boundary(game_state)
            for i in squares:
                for value in range(1, N + 1):
                    if is_valid_move(game_state, i[0], i[1], value):
                        row_values, col_values, block_values = get_constraints(
                            game_state.board, Move(i, value))
                        if value not in row_values + col_values + block_values:
                            moves.append(Move(i, value))
            return moves
            # return apply_heuristics(move)
        
        
        # Check whether placing a move in a square blocks opposite player
        def is_blocking(game_state, square):
            unlocked_squares = new_squares(game_state, square)

            # Check if neighbors of unlocked squares are part of allowed squares for opposing player, thus blocking them
            game_state.current_player = 3 - game_state.current_player
            for i in unlocked_squares:
                neighbors = neighbors_square(i)
                for j in neighbors:
                    if j in game_state.player_squares():
                        return 1
            game_state.current_player = 3 - game_state.current_player
            return 0

        '''
        Generating possible legal moves and constraints for use in minimax
        and evaluating score achieved by placing move
        '''

        # Check allowed moves to generate legal_moves
        def get_legal_moves(game_state):
            moves = []
            for i in game_state.player_squares():
                for value in range(1, N + 1):
                    if is_valid_move(game_state, i[0], i[1], value):
                        row_values, col_values, block_values = get_constraints(
                            game_state.board, Move(i, value))
                        if value not in row_values + col_values + block_values:
                            moves.append(Move(i, value))
            # return moves
            return apply_heuristics(moves)

        '''
        Functions for use in minimax
        '''
        # Add a move to current game_state, update and return a copy of game_state with new move attached 
        def add_to_game_state(game_state, move):
            new_state = copy.deepcopy(game_state)
            new_state.board.put(move.square, move.value)
            new_state.moves.append(move)
            new_state.occupied_squares().append(move.square)
            new_state.scores[new_state.current_player - 1] += evaluate_score(game_state.board, move)
            new_state.current_player = 3 - new_state.current_player    # Change player after adding move to game_state
            return new_state
        
        # Extract children from given game_state by adding all possible legal moves
        def get_children_game_states(game_state):
            children = []
            legal_moves = get_legal_moves(game_state)
            for i in legal_moves:
                child = add_to_game_state(game_state, i)
                children.append(child)
            return children

        # Minimax evaluator
        def evaluate(game_state):
            if player == 1:
                eval = game_state.scores[0] - game_state.scores[1]
            else:
                eval = game_state.scores[1] - game_state.scores[0]
            # print("Evaluate value being looked at is: ", eval)
            return eval
        
        '''
        Minimax evaluator with alpha beta pruning and using transpositio table
        '''

        
        # Minimax with alpha-beta pruning
        def minimax(state, depth, alpha, beta, is_maximizing):
            
            children = get_children_game_states(state)
            if depth == 0 or not children:
                eval = evaluate(state)
                return eval

            if is_maximizing:
                max_eval = float('-inf')
                for child in children:
                    eval = minimax(child, depth - 1, alpha, beta, False)
                    max_eval = max(eval, max_eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
                return max_eval

            else:
                min_eval = float('inf')
                for child in children:
                    eval = minimax(child, depth - 1, alpha, beta, True)
                    min_eval = min(eval, min_eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
                return min_eval


        '''
        Get allowed squares for heuristic strategy, try to conquer board + block opposite player
        '''

        legal_moves = get_legal_moves(game_state)

        # Random move fallback
        if legal_moves:
            move = random.choice(legal_moves)
            self.propose_move(random.choice(legal_moves))
            greedy_score = evaluate_score(game_state.board, move)
        
        # Greedy fallback
        for move in legal_moves:
            reward = evaluate_score(game_state.board, move)
            if greedy_score < reward:
                greedy_score = reward
                self.propose_move(move)
        
        
        boundary_moves = get_boundary_moves(game_state)
        
        
        # Propose conquering move
        max_degree = 0

        for move in boundary_moves:
            degree = len(new_squares(game_state, move.square))
            if degree > max_degree:
                max_degree = degree
                self.propose_move(move)
        
        
        '''
        Perform Minimax in possible legal moves of board
        '''


        # Perform iterative deepening with Minimax 
        for depth in range(1, N * N + 1):
            if greedy_score >= 0:
                best_eval = evaluate(game_state) + greedy_score
            else:
                best_eval = float('-inf')
            best_move = None
            for move in legal_moves:
                new_state = add_to_game_state(game_state, move)
                move_eval = minimax(new_state, depth, float('-inf'), float('inf'), False)
                if best_eval <  move_eval :
                    best_eval = move_eval
                    best_move = move
                    self.propose_move(best_move)
            if best_move:
                self.propose_move(best_move)

