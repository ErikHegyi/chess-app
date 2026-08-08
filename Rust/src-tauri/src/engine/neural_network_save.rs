use ort::{
    inputs,
    value::TensorRef,
    session::{Session, builder::GraphOptimizationLevel},
    error::Result
};
use std::{
    sync::Mutex,
    path::Path
};
use cozy_chess::{Board, Color, Piece, Square};
use crate::engine::tensor::Tensor;


#[derive(Debug)]
pub struct ChessNN {
    model: Mutex<Session>
}


impl ChessNN {
    pub fn from_file<P>(file_path: P) -> Result<Self>
    where
        P: AsRef<Path>
    {
        // Import the model
        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(file_path)?;

        Ok(Self { model: Mutex::new(model) })
    }

    fn evaluate(&self, input_tensor: Tensor) -> Result<f32> {
        // Convert the input tensor into the appropriate data type
        let input_tensor_ref = TensorRef::from_array_view(&input_tensor)?;
        let input = inputs!["board_state" => input_tensor_ref];

        // Extract the model as mutable
        let mut model = self.model.lock().unwrap();

        // Inference
        let outputs = model.run(input)?;

        // Extract the scalar value and return
        outputs["evaluation"].try_extract_scalar::<f32>()

    }

    fn board_to_tensor(board: &Board) -> Tensor {
        // Create an empty board
        let mut tensor: Tensor = Tensor::zeros((1, 12, 8, 8));

        // Check who is moving
        let side_to_move: Color = board.side_to_move();

        for square in Square::ALL {
            if let Some(piece) = board.piece_on(square) {
                let piece_color = board.color_on(square).unwrap();

                // Friendly pieces go to channels 0..5
                // Opponent pieces go to channels 6..11
                let color_offset = if piece_color == side_to_move { 0 } else { 6 };

                // Get the dimension ID of the pieces
                let piece_idx = match piece {
                    Piece::Pawn => 0,
                    Piece::Knight => 1,
                    Piece::Bishop => 2,
                    Piece::Rook => 3,
                    Piece::Queen => 4,
                    Piece::King => 5,
                };

                // Get the total dimension ID
                let channel = color_offset + piece_idx;

                // Perspective Rotation:
                // If Black is to move, flip the square 180° so Black's pieces
                // advance upward from the bottom of the matrix (Rank 1 <-> 8, File A <-> H)
                let persp_square = if side_to_move == Color::White {
                    square
                } else {
                    square.flip_file().flip_rank()
                };

                // Rank 8 = Row 0, Rank 1 = Row 7
                let rank = 7 - persp_square.rank() as usize;

                // File A = Col 0, File H = Col 7
                let file = persp_square.file() as usize;

                tensor[[0, channel, rank, file]] = 1.0;
            }
        }

        tensor
    }

    pub fn evaluate_board(&self, board: &Board) -> Result<f32> {
        self.evaluate(Self::board_to_tensor(board))
    }
}