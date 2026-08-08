use cozy_chess::{Board, Piece, Color};
use tract_onnx::tract_core::plan::eval;
use crate::engine::neural_network::ChessNN;


const BRANCHING_FACTOR: usize = 3;  // How wide should the tree be?
const MAX_DEPTH: usize = 1;  // How deep should the tree be?
const HALFMOVE_CLOCK_MAX: u8 = 100;  // Chess rule



#[derive(Debug, PartialEq, Eq)]
pub enum Draw {
    Stalemate,
    FiftyMoveRule,
    InsufficientMaterial
}


#[derive(Debug, PartialEq, Eq)]
pub enum GameResult {
    Ongoing,
    Checkmate { winner: Color },
    Draw { reason: Draw }
}


#[derive(Debug, Clone)]
pub struct Node<'a> {
    pub board_state: Board,
    pub evaluation: f32,
    pub line_eval: f32,
    pub depth: usize,
    pub children: Vec<Node<'a>>,
    model: &'a ChessNN,
}

impl<'a> Node<'a> {
    pub fn from(board: Board, model: &'a ChessNN, depth: usize) -> Self {
        Self {
            board_state: board,
            evaluation: 0.0,
            line_eval: 0.0,
            depth,
            children: Vec::new(),
            model,
        }
    }

    pub fn evaluate(&mut self, temperature: f32) {
        // 1. Generate legal moves (don't save to self)
        let mut legal_moves: Vec<Board> = Vec::with_capacity(35);
        self.board_state.generate_moves(|moves| {
            for mv in moves {
                let mut new_board = self.board_state.clone();
                new_board.play(mv);
                legal_moves.push(new_board);
            }
            false
        });

        // Handle terminal states immediately
        if legal_moves.is_empty() || self.depth >= MAX_DEPTH {
            self.evaluation = self.terminal_or_fallback(0.0); // Implement terminal logic here
            self.line_eval = self.evaluation;
            return;
        }

        // 2. Batch Inference! Evaluate all child states in ONE neural network call
        let child_evals = self.model.evaluate_boards(&legal_moves).unwrap_or_default();

        // Zip boards with their evaluations
        let mut children_with_evals: Vec<(Board, f32)> = legal_moves.into_iter()
            .zip(child_evals.into_iter())
            .collect();

        // 3. Sort Descending (Highest evaluation first)
        children_with_evals.sort_by(|(_, a), (_, b)| b.total_cmp(a));

        // 4. Take top BRANCHING_FACTOR
        let num_to_check = std::cmp::min(BRANCHING_FACTOR, children_with_evals.len());
        let mut children = Vec::with_capacity(num_to_check);
        let mut line_evals = Vec::with_capacity(num_to_check);

        for (board, eval) in children_with_evals.into_iter().take(num_to_check) {
            let mut child_node = Node::from(board, self.model, self.depth + 1);

            if child_node.depth < MAX_DEPTH {
                child_node.evaluate(temperature); // Recursive DFS
                line_evals.push(-child_node.line_eval);
            } else {
                child_node.evaluation = eval;
                child_node.line_eval = eval;
                line_evals.push(-eval);
            }
            children.push(child_node);
        }

        self.children = children;

        // 5. Apply Softmax (Fixed math: divide by temperature)
        let max_eval = line_evals.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exponentials: Vec<f32> = line_evals.iter()
            // Subtract max_eval for numerical stability to prevent Infinity
            .map(|x| ((x - max_eval) / temperature).exp())
            .collect();

        let sum: f32 = exponentials.iter().sum();
        let softmax: Vec<f32> = exponentials.iter().map(|x| x / sum).collect();

        self.line_eval = line_evals.iter().zip(softmax).map(|(val, weight)| val * weight).sum();
    }

    fn terminal_or_fallback(&self, fallback: f32) -> f32 {
        match self.evaluate_game_end() {
            GameResult::Ongoing => fallback,
            GameResult::Draw { .. } => 0.0,
            GameResult::Checkmate { winner } => if winner == self.board_state.side_to_move() { 1.0 } else { -1.0 }
        }
    }

    fn evaluate_(&self, evaluation: Option<f32>) -> f32 {
        match self.evaluate_game_end() {
            GameResult::Ongoing => evaluation.unwrap_or(self.evaluate_using_model()),
            GameResult::Draw { reason: _reason } => 0.0,
            GameResult::Checkmate { winner} => if winner == Color::White { 1.0 } else { -1.0 }
        }
    }

    fn evaluate_using_model(&self) -> f32 {
        self.model.evaluate_boards(&[self.board_state.clone()])
            .expect("An error occured while evaluating the board")[0]
    }

    fn legal_moves(board: &Board) -> Vec<Board> {
        // Generate all possible legal moves
        let mut legal_moves: Vec<Board> = Vec::new();
        board.generate_moves(|moves| {
            for mv in moves {
                // Copy the board
                let mut new_board: Board = board.clone();

                // Play the move
                new_board.play(mv);

                // Save the board
                legal_moves.push(new_board);
            }
            false
        });

        legal_moves
    }

    pub fn has_game_ended(&self) -> bool {
        match self.evaluate_game_end() {
            GameResult::Ongoing => false,
            _ => true
        }
    }

    pub fn evaluate_game_end(&self) -> GameResult {
        // 50 move rule
        if self.board_state.halfmove_clock() >= HALFMOVE_CLOCK_MAX {
            return GameResult::Draw { reason: Draw::FiftyMoveRule }
        }
        // Insufficient material
        else if Self::is_insufficient_material(&self.board_state) {
            return GameResult::Draw { reason: Draw::InsufficientMaterial }
        }

        if Self::legal_moves(&self.board_state).is_empty() {
            // Stalemate
            return if self.board_state.checkers().is_empty() {
                GameResult::Draw { reason: Draw::Stalemate }
            }
            // Checkmate
            else {
                GameResult::Checkmate { winner: !self.board_state.side_to_move() }
            }
        }

        GameResult::Ongoing
    }

    fn is_insufficient_material(board: &Board) -> bool {
        let total_pieces = board.occupied().len();

        // K vs K
        if total_pieces == 2 {
            return true;
        }

        // K+B vs K or K+N vs K
        if total_pieces == 3 {
            let knights = board.pieces(Piece::Knight).len();
            let bishops = board.pieces(Piece::Bishop).len();
            if knights == 1 || bishops == 1 {
                return true;
            }
        }

        false
    }

    fn softmax(evaluations: Vec<f32>, temperature: f32) -> Vec<f32> {
        let exponentials: Vec<f32> = evaluations
            .iter()
            .map(|x| x.exp().powf(temperature))
            .collect();
        let sum: f32 = exponentials.iter().sum();
        exponentials.iter().map(|x| x / sum).collect()
    }

    pub fn print_board(&self) {
        println!("\n  +-----------------+");

        // Loop ranks from 8 down to 1 (top to bottom)
        for rank in (0..8).rev() {
            print!("{} | ", rank + 1);

            for file in 0..8 {
                let square = cozy_chess::Square::index(rank * 8 + file);

                if let Some(piece) = self.board_state.piece_on(square) {
                    let color = self.board_state.color_on(square).unwrap();
                    let symbol = match (piece, color) {
                        (Piece::Pawn, Color::White) => "P",
                        (Piece::Knight, Color::White) => "N",
                        (Piece::Bishop, Color::White) => "B",
                        (Piece::Rook, Color::White) => "R",
                        (Piece::Queen, Color::White) => "Q",
                        (Piece::King, Color::White) => "K",
                        (Piece::Pawn, Color::Black) => "p",
                        (Piece::Knight, Color::Black) => "n",
                        (Piece::Bishop, Color::Black) => "b",
                        (Piece::Rook, Color::Black) => "r",
                        (Piece::Queen, Color::Black) => "q",
                        (Piece::King, Color::Black) => "k",
                    };
                    print!("{} ", symbol);
                } else {
                    print!(". ");
                }
            }
            println!("|");
        }

        println!("  +-----------------+");
        println!("    a b c d e f g h\n");
        println!("Side to move: {:?}", self.board_state.side_to_move());
    }
}