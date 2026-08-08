use std::path::Path;
use cozy_chess::{Board, Color, Piece};
use tract_onnx::prelude::*;
use tract_onnx::prelude::tract_ndarray::Array4;

type Tensor = Array4<f32>;
type ChessModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

#[derive(Debug)]
pub struct ChessNN {
    model: ChessModel,
}

impl ChessNN {
    pub fn from_file<P>(file_path: P) -> TractResult<Self> where P: AsRef<Path> {
        let model = onnx()
            .model_for_path(file_path)?
            .into_optimized()?
            .into_runnable()?;
        Ok(Self { model })
    }

    // Evaluate multiple boards in a single forward pass!
    pub fn evaluate_boards(&self, boards: &[Board]) -> TractResult<Vec<f32>> {
        if boards.is_empty() {
            return Ok(Vec::new());
        }

        let batch_size = boards.len();
        let mut tensor = Tensor::zeros((batch_size, 12, 8, 8));

        for (b_idx, board) in boards.iter().enumerate() {
            let side_to_move = board.side_to_move();

            // Iterate using bitboards (lightning fast)
            for color in [Color::White, Color::Black] {
                let color_offset = if color == side_to_move { 0 } else { 6 };

                for piece in Piece::ALL {
                    let piece_idx = piece as usize; // Pawn = 0, Knight = 1, etc.
                    let channel = color_offset + piece_idx;

                    // Get all squares containing this piece/color instantly
                    let bitboard = board.colored_pieces(color, piece);

                    for square in bitboard {
                        let persp_square = if side_to_move == Color::White {
                            square
                        } else {
                            square.flip_file().flip_rank()
                        };

                        let rank = 7 - persp_square.rank() as usize;
                        let file = persp_square.file() as usize;

                        tensor[[b_idx, channel, rank, file]] = 1.0;
                    }
                }
            }
        }

        let tract_tensor: tract_onnx::prelude::Tensor = tensor.into();
        let outputs = self.model.run(tvec![tract_tensor.into()])?;

        let eval_slice = outputs[0].as_slice::<f32>()?;
        Ok(eval_slice.to_vec())
    }
}