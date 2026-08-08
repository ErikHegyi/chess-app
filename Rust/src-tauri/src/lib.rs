mod engine;

use cozy_chess::{Board, Color};
use crate::engine::{ChessNN, Node, GameResult, Draw};

const MODEL_PATH: &str = r"model_path";


#[tauri::command]
fn get_move(fen: String, temperature: f32) -> String {
    println!("{fen}");
    // Create the board
    let board: Board = Board::from_fen(&fen.trim(), false).unwrap();

    // Import the model
    let model: ChessNN = ChessNN::from_file(MODEL_PATH).unwrap();

    // Create the node
    let mut node: Node = Node::from(board, &model, 0);

    // Evaluate the possible moves
    node.evaluate(temperature);

    // Check for game end
    if node.has_game_ended() {
        let game_end_reason: &str = match node.evaluate_game_end() {
            GameResult::Draw { reason } => match reason {
                Draw::Stalemate => "DRAW|STALEMATE",
                Draw::FiftyMoveRule => "DRAW|FIFTYMOVERULE",
                Draw::InsufficientMaterial => "DRAW|INSUFFICIENTMATERIAL"
            },
            GameResult::Checkmate { winner } => match winner {
                Color::White => "WIN|WHITE",
                Color::Black => "WIN|BLACK"
            },
            _ => unreachable!()
        };
        return format!("{game_end_reason}:{fen}");
    }

    // Select the best move
    let best_move: String = node
        .children
        .iter()
        .max_by(|x, y| x.line_eval.total_cmp(&y.line_eval))
        .unwrap()
        .board_state
        .clone()
        .to_string();

    format!("ONGOING|ONGOING:{best_move}")
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![get_move])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
