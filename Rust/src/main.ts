import {Piece} from "./logic/pieces.ts";
import {Color} from "./logic/colors.ts";
import {ChessBoard} from "./logic/chessBoard.ts";
import {ChessGame} from "./logic/chessGame.ts";


let pause = false;

// --- 2. Unicode Chess Piece Mapping ---
const UNICODE_PIECES = {
  [Piece.PAWN]: '♟',
  [Piece.KNIGHT]: '♞',
  [Piece.BISHOP]: '♝',
  [Piece.ROOK]: '♜',
  [Piece.QUEEN]: '♛',
  [Piece.KING]: '♚'
};

// --- 3. Board Renderer ---
function renderBoard(board: ChessBoard, containerId: string) {
  const container = document.getElementById(containerId);

  if (container === null) { return; }
  container.innerHTML = ''; // Clear previous board

  // Row 0 = Rank 8 (Top), Row 7 = Rank 1 (Bottom)
  for (let row = 7; row >= 0; --row) {
    for (let col = 0; col < 8; col++) {
      const square = document.createElement('div');

      // Alternating light/dark squares
      const isLight = (row + col) % 2 === 0;
      square.className = `square ${isLight ? 'light' : 'dark'}`;

      // Check for piece in matrix
      const coloredPiece = board.matrix[col][row];
      if (coloredPiece) {
        const pieceSpan = document.createElement('span');
        pieceSpan.textContent = UNICODE_PIECES[coloredPiece.piece];
        pieceSpan.className = coloredPiece.color === Color.WHITE ? 'piece-white' : 'piece-black';
        square.appendChild(pieceSpan);
      }

      container.appendChild(square);
    }
  }
}

let game = new ChessGame();
renderBoard(game.board, 'board');

window.addEventListener("keydown", event => {
  if (event.key != " " || pause) return;
  pause = true;

  // Option A: Using .then() chaining
  game.getNextState().then(() => {
    renderBoard(game.board, 'board');
    pause = false;
  }).catch(error => {
    console.error("Failed to fetch next state:", error);
    pause = false; // Ensure unpause even if Rust throws an error
  });
})
