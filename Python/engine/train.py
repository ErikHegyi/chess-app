from os import getcwd
from pathlib import Path
import torch

from model import criterion, model, optimizer
from training import read_games, train_with_target

if __name__ == '__main__':
    # 1. Setup Paths & Ensure Directory Exists
    current_path = Path(getcwd()).resolve()
    main_folder = current_path.parent.parent
    data_path = main_folder / 'games' / 'lichess_db_eval.jsonl.zst'

    models_dir = main_folder / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / 'chessbot.pth'
    optimizer_path = models_dir / 'optimizer.pth'
    index_path = models_dir / 'index.txt'
    onnx_path = models_dir / 'chess_eval.onnx'

    # 2. Load Existing Weights & Optimizer State
    if model_path.exists():
        print(f"Loading model weights from {model_path}...")
        model.load_state_dict(torch.load(model_path, weights_only=True))

    if optimizer_path.exists():
        print(f"Loading optimizer state from {optimizer_path}...")
        optimizer.load_state_dict(torch.load(optimizer_path, weights_only=True))

    batch_size = 64

    # 3. Read Starting Index
    if index_path.exists():
        with open(index_path, 'r') as f:
            starting_index = int(f.read().strip())
    else:
        starting_index = 0

    print(f"Starting training at dataset sample index: {starting_index}")

    # 4. Hyperparameters & Logging
    log_frequency = 100
    save_onnx_frequency = 5000

    running_loss = 0.0
    total_batches = 0

    # 5. Main Training Loop
    for batch in read_games(data_path, batch_size, starting_index):
        model.train()

        # Train on single batch
        loss = train_with_target(model, batch, optimizer, criterion).item()

        running_loss += loss
        total_batches += 1

        current_dataset_index = starting_index + (total_batches * batch_size)

        # Periodic Logging & Checkpointing
        if total_batches % log_frequency == 0:
            avg_loss = running_loss / log_frequency
            print(f"Batch {total_batches} | Samples: {current_dataset_index} | Avg Loss: {avg_loss:.4f}")
            running_loss = 0.0

            # Save PyTorch Model & Optimizer States
            torch.save(model.state_dict(), model_path)
            torch.save(optimizer.state_dict(), optimizer_path)

            # Save dataset progress index
            with open(index_path, 'w') as f:
                f.write(str(current_dataset_index))

        # Export ONNX Model for Rust Engine
        if total_batches % save_onnx_frequency == 0 or total_batches == 100:
            print(f"--> Exporting ONNX model at batch {total_batches}...")

            model.eval()  # Set to evaluation mode for ONNX export
            dummy_input = torch.randn(1, 12, 8, 8)

            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                input_names=['board_state'],
                output_names=['evaluation'],
                dynamic_axes={
                    'board_state': {0: 'batch_size'},
                    'evaluation': {0: 'batch_size'}  # Critical for Rust batched inference!
                },
                verbose=False
            )