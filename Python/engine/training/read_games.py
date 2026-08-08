from pathlib import Path
import io
import json
import zstandard as zstd
from .board_to_tensor import board_to_tensor
import torch
from math import tanh
import chess


def read_games(zst_file: Path, batch_size: int = 16, starting_index: int = 0):
    decompressor = zstd.ZstdDecompressor()

    with open(zst_file, mode='rb') as fh:
        with decompressor.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')

            batch = []

            n: int = 0
            for line in text_stream:
                n += 1
                if n < starting_index:
                    continue

                data = json.loads(line)

                board = chess.Board(data['fen'])
                evals = data['evals']

                if not evals:
                    continue

                # Select the deepest evaluation entry
                best_eval = max(evals, key=lambda item: item.get('depth', 0))
                pvs = best_eval.get('pvs', [])

                if not pvs:
                    continue

                top_pv = pvs[0]

                if top_pv.get('cp') is not None:
                    evaluation = float(top_pv['cp'])
                elif top_pv.get('mate') is not None:
                    evaluation = 10_000.0 if top_pv['mate'] > 0 else -10_000.0
                else:
                    continue

                evaluation = tanh(evaluation / 2500.0)

                # If Black is to move, negate the score so + means Black is winning
                if board.turn == chess.BLACK:
                    evaluation = -evaluation

                state_tensor = board_to_tensor(board)

                batch.append((state_tensor, evaluation))

                if len(batch) == batch_size:
                    x = torch.stack([b[0] for b in batch])
                    y = torch.tensor([b[1] for b in batch], dtype=torch.float32).unsqueeze(1)
                    yield x, y
                    batch = []

            if len(batch) != 0:
                x = torch.stack([b[0] for b in batch])
                y = torch.tensor([b[1] for b in batch], dtype=torch.float32).unsqueeze(1)
                yield x, y

