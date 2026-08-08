import {ChessBoard} from "./chessBoard.ts";
import {Color} from "./colors.ts";
import { invoke } from "@tauri-apps/api/core";

export class ChessGame {
    board: ChessBoard;
    toMove: Color;
    castlingRights: boolean[];
    enPassantTargetSquare: string;
    halfMoveClock: number;
    fullMoveCounter: number;

    constructor() {
        this.board = ChessBoard.defaultBoard();
        this.toMove = Color.WHITE;
        this.castlingRights = [false, false, false, false];
        this.enPassantTargetSquare = "-";
        this.halfMoveClock = 0;
        this.fullMoveCounter = 1;
    }

    static fromFEN(fen: String): ChessGame {
        console.log(fen);
        // Split the description into parts
        let parts: string[] = fen.split(" ");

        // Create a new object
        let game: ChessGame = new ChessGame();

        // Parse the first part - the board representation
        game.board = ChessBoard.fromFEN(parts[0]);

        // Parse the second part - who's turn it is
        if (parts[1] == "w") { game.toMove = Color.WHITE; }
        else { game.toMove = Color.BLACK; }

        // Parse the third part - which castling rights remain
        if (parts[2].includes("K")) { game.castlingRights[0] = true; }
        if (parts[2].includes("Q")) { game.castlingRights[1] = true; }
        if (parts[2].includes("k")) { game.castlingRights[2] = true; }
        if (parts[2].includes("q")) { game.castlingRights[3] = true; }

        // Parse the fourth part - the en passant target square
        game.enPassantTargetSquare = parts[3];

        // Parse the fifth part - the half move clock
        game.halfMoveClock = Number(parts[4]);

        // Parse the sixth part - the full move counter
        game.fullMoveCounter = Number(parts[5]);

        return game;
    }

    toFEN(): String {
        // Encode the board
        let fen = this.board.toFEN() + " ";

        // Encode who's turn it is
        let colorToMove;
        if (this.toMove == Color.WHITE) { colorToMove = "w"; }
        else { colorToMove = "b"; }

        fen += colorToMove + " ";

        // Encode castling rights
        let castlingRights = "";
        console.log(this.castlingRights);
        if (this.castlingRights[0]) { castlingRights += "K"; }
        if (this.castlingRights[1]) { castlingRights += "Q"; }
        if (this.castlingRights[2]) { castlingRights += "k"; }
        if (this.castlingRights[3]) { castlingRights += "q"; }
        console.log(castlingRights);
        // Edge case: no one can castle
        if (castlingRights == "") { castlingRights = "-"; }

        fen += castlingRights + " ";

        // Encode the rest of the information, which does not need processing
        fen += this.enPassantTargetSquare + " ";
        fen += this.halfMoveClock.toString() + " ";
        fen += this.fullMoveCounter.toString() + " ";

        return fen
    }

    async getNextState() {
        let newFEN: string = await invoke("get_move", { fen: this.toFEN(), temperature: 1.0 });
        console.log(newFEN);
        let split: string[] = newFEN.split(":");
        let state = split[0];
        let fen = split[1];
        if (state == "ONGOING|ONGOING") {
            let game = ChessGame.fromFEN(fen);
            this.board = game.board;
            this.toMove = game.toMove;
            this.castlingRights = game.castlingRights;
            this.enPassantTargetSquare = game.enPassantTargetSquare;
            this.halfMoveClock = game.halfMoveClock;
            this.fullMoveCounter = game.fullMoveCounter;
        }
    }
}