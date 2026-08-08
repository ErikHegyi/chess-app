import chess
import numpy as np
import torch


PIECES = [
    (chess.PAWN, chess.WHITE), (chess.KNIGHT, chess.WHITE),
    (chess.BISHOP, chess.WHITE), (chess.ROOK, chess.WHITE),
    (chess.QUEEN, chess.WHITE), (chess.KING, chess.WHITE),
    (chess.PAWN, chess.BLACK), (chess.KNIGHT, chess.BLACK),
    (chess.BISHOP, chess.BLACK), (chess.ROOK, chess.BLACK),
    (chess.QUEEN, chess.BLACK), (chess.KING, chess.BLACK)
]

# Standard piece order for channels 0..5 (Friendly) and 6..11 (Opponent)
PIECE_TYPES = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    tensor = np.zeros((12, 8, 8), dtype=np.float32)

    # Determine who is to move
    stm = board.turn  # True for White, False for Black
    friendly_color = stm
    opponent_color = not stm

    # Map piece colors to channel offsets:
    # Channels 0..5  = Friendly pieces (Side To Move)
    # Channels 6..11 = Opponent pieces
    color_channel_offsets = {
        friendly_color: 0,
        opponent_color: 6,
    }

    for piece_type_idx, piece_type in enumerate(PIECE_TYPES):
        for color, channel_offset in color_channel_offsets.items():
            channel = channel_offset + piece_type_idx

            for sq in board.pieces(piece_type, color):
                # If Black is to move, flip the square 180° (a1 <-> h8)
                # python-chess has chess.square_mirror() built-in for this
                persp_sq = sq if stm == chess.WHITE else chess.square_mirror(sq)

                rank = 7 - chess.square_rank(persp_sq)  # Row 0 = Rank 8
                file = chess.square_file(persp_sq)  # Col 0 = File A

                tensor[channel, rank, file] = 1.0

    return torch.from_numpy(tensor)