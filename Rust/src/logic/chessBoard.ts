import {Piece} from "./pieces.ts";
import {Color} from "./colors.ts";

const WIDTH: number = 8;
const HEIGHT: number = 8;


export class ColoredPiece {
    color: Color
    piece: Piece

    constructor(color: Color, piece: Piece) {
        this.color = color; this.piece = piece;
    }
}


export class ChessBoard {
    matrix: Array<Array<ColoredPiece | null>>;

    constructor() {
        this.matrix = new Array(WIDTH);
        for (let i = 0; i < HEIGHT; ++i) {
            this.matrix[i] = new Array(HEIGHT).fill(null);
        }
    }

    addPiece(piece: ColoredPiece, x: number, y: number): void {
        this.matrix[x][y] = piece;
    }

    removePiece(x: number, y: number): void {
        this.matrix[x][y] = null;
    }

    movePiece(fromX: number, fromY: number, toX: number, toY: number): void {
        this.matrix[toX][toY] = this.matrix[fromX][fromY];
        this.matrix[fromX][fromY] = null;
    }

    toFEN(): String {
        let fenString: String = "";

        // Go from the last row (black's back rank), as FEN dictates
        for (let j = HEIGHT - 1; j >= 0; --j) {
            // Keep track of the number of empty squares
            let emptySquares: number = 0;

            // Iterate through each tile in the row
            for (let i = 0; i < WIDTH; ++i) {
                let coloredPiece: ColoredPiece | null = this.matrix[i][j];

                // If the tile is empty, continue
                if (coloredPiece == null) {
                    ++emptySquares;
                } else {
                    // Convert the piece object to a string
                    let pieceString: String = "";
                    switch (coloredPiece.piece) {
                        case Piece.PAWN: pieceString = "P"; break;
                        case Piece.BISHOP: pieceString = "B"; break;
                        case Piece.KNIGHT: pieceString = "N"; break;
                        case Piece.ROOK: pieceString = "R"; break;
                        case Piece.QUEEN: pieceString = "Q"; break;
                        case Piece.KING: pieceString = "K"; break;
                    }

                    // Convert to lowercase if the piece belongs to black
                    if (coloredPiece.color == Color.BLACK) {
                        pieceString = pieceString.toLowerCase();
                    }

                    // If there were some empty squares, add them, and zero the tracker
                    if (emptySquares != 0) {
                        fenString += emptySquares.toString();
                        emptySquares = 0;
                    }

                    // Add the piece
                    fenString += pieceString.toString();
                }
            }
            // If there were some empty squares left, add them
            if (emptySquares != 0) {
                fenString += emptySquares.toString();
            }

            // Add a slash if this wasn't the last row
            if (j != 0) { fenString += "/"; }
        }

        return fenString;
    }

    static fromFEN(fen: String): ChessBoard {
        // Helper functions
        const isInteger = (str: string) => Number.isInteger(Number(str));
        const isUpperCase = (str: string) => str.toUpperCase() == str;

        // Create a new chessboard
        let board: ChessBoard = new ChessBoard();

        // Split the FEN representation into rows
        let rows: string[] = fen.split("/");

        // Iterate through each row from the back, as FEN dictates
        for (let j = rows.length - 1; j >= 0; --j) {
            let row: string = rows[j];

            // Keep track of the number of empty squares in the row
            let emptySquares = 0;

            // Iterate through each tile in the row
            for (let i = 0; i < row.length; ++i) {
                let tile: string = row[i];
                // If the tile contains an integer, it was an empty tile
                if (isInteger(tile)) {
                    emptySquares += Number(tile) - 1;
                } else {
                    // Get the color of the piece
                    let color: Color;

                    if (isUpperCase(tile)) { color = Color.WHITE; }
                    else { color = Color.BLACK; }

                    // Convert to lowercase
                    tile = tile.toLowerCase();

                    // Get the piece type
                    let piece: Piece;

                    if (tile == "k") { piece = Piece.KING; }
                    else if (tile == "q") { piece = Piece.QUEEN; }
                    else if (tile == "r") { piece = Piece.ROOK; }
                    else if (tile == "b") { piece = Piece.BISHOP; }
                    else if (tile == "n") { piece = Piece.KNIGHT; }
                    else { piece = Piece.PAWN; }

                    // Construct the colored piece object
                    let coloredPiece: ColoredPiece = new ColoredPiece(color, piece);

                    // Add the piece to the board
                    board.addPiece(coloredPiece, i + emptySquares, HEIGHT - j - 1);
                }
            }
        }

        return board
    }

    static defaultBoard(): ChessBoard {
        let board: ChessBoard = new ChessBoard();

        // Pawns
        for (let i = 0; i < WIDTH; ++i) {
            board.addPiece(new ColoredPiece(Color.WHITE, Piece.PAWN), i, 1);
            board.addPiece(new ColoredPiece(Color.BLACK, Piece.PAWN), i, 6);
        }

        // Rooks
        board.addPiece(new ColoredPiece(Color.WHITE, Piece.ROOK), 0, 0);
        board.addPiece(new ColoredPiece(Color.WHITE, Piece.ROOK), 7, 0);
        board.addPiece(new ColoredPiece(Color.BLACK, Piece.ROOK), 0, 7);
        board.addPiece(new ColoredPiece(Color.BLACK, Piece.ROOK), 7, 7);

        // Knights
        board.addPiece(new ColoredPiece(Color.WHITE, Piece.KNIGHT), 1, 0);
        board.addPiece(new ColoredPiece(Color.WHITE, Piece.KNIGHT), 6, 0);
        board.addPiece(new ColoredPiece(Color.BLACK, Piece.KNIGHT), 1, 7);
        board.addPiece(new ColoredPiece(Color.BLACK, Piece.KNIGHT), 6, 7);

        // Bishops
        board.addPiece(new ColoredPiece(Color.WHITE, Piece.BISHOP), 2, 0);
        board.addPiece(new ColoredPiece(Color.WHITE, Piece.BISHOP), 5, 0);
        board.addPiece(new ColoredPiece(Color.BLACK, Piece.BISHOP), 2, 7);
        board.addPiece(new ColoredPiece(Color.BLACK, Piece.BISHOP), 5, 7);

        // Queens
        board.addPiece(new ColoredPiece(Color.WHITE, Piece.QUEEN), 3, 0);
        board.addPiece(new ColoredPiece(Color.BLACK, Piece.QUEEN), 3, 7);

        // Kings
        board.addPiece(new ColoredPiece(Color.WHITE, Piece.KING), 4, 0);
        board.addPiece(new ColoredPiece(Color.BLACK, Piece.KING), 4, 7);

        return board;
    }
}