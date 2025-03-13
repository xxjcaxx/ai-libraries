#include "../chess-library-master/include/chess.hpp"
#include <map>    // Include this for std::map
#include <vector> // Needed for std::vector

extern "C"
{

  using namespace chess;

  using Matrix = std::array<std::array<int, 8>, 8>;

  Matrix board_to_matrix(Board &board, PieceType piece_type)
  {
    Matrix matrix = {}; // Initialize an 8x8 matrix with zeros
                        // std::cout <<  board.getFen();
    std::uint64_t bbw = board.pieces(piece_type, Color::WHITE).getBits();
    std::uint64_t bbb = board.pieces(piece_type, Color::BLACK).getBits();

    for (int i = 0; i < 64; ++i)
    {
      // Get the row and column based on the bit index
      int row = i / 8; // Row index (0 to 7)
      int col = i % 8; // Column index (0 to 7)

      // Extract the bit at the current position
      if (((bbw >> (63 - i)) & 1) == 1)
      {
        matrix[row][col] = 1;
      }
      if (((bbb >> (63 - i)) & 1) == 1)
      {
        matrix[row][col] = -1;
      }
    }

    return matrix;
  }

  using Matrix3D = std::vector<Matrix>;

  Matrix3D board_to_3D_matrix(Board &board)
  {
    Matrix3D matrix3D = {};
    std::vector<PieceType> pieces = {
        PieceType::PAWN, PieceType::KNIGHT, PieceType::BISHOP,
        PieceType::ROOK, PieceType::QUEEN, PieceType::KING};

    for (auto piece : pieces)
    {
      std::uint64_t bbw = board.pieces(piece, Color::WHITE).getBits();
      std::uint64_t bbb = board.pieces(piece, Color::BLACK).getBits();
      Matrix white_matrix = {}, black_matrix = {};

      for (int i = 0; i < 64; ++i)
      {
        // Get the row and column based on the bit index
        int row = i / 8; // Row index (0 to 7)
        int col = i % 8; // Column index (0 to 7)

        // Extract the bit at the current position
        if (((bbw >> (63 - i)) & 1) == 1)
        {
          white_matrix[row][7-col] = 1;
        }
        if (((bbb >> (63 - i)) & 1) == 1)
        {
          black_matrix[row][7-col] = 1;
        }
      }

      matrix3D.push_back(white_matrix);
      matrix3D.push_back(black_matrix);
    }

    Matrix color_matrix = {};
    if (board.sideToMove() == Color::WHITE)
    {
      for (auto &row : color_matrix)
      {
        std::fill(row.begin(), row.end(), 1);
      }
    }
    else
    {
      for (auto &row : color_matrix)
      {
        std::fill(row.begin(), row.end(), 0);
      }
    }
    matrix3D.push_back(color_matrix);

    return matrix3D;
  }

  std::map<std::pair<int, int>, int> codes; // Declare the map

  void initialize_codes()
  {
    int i = 0;
    // All 56 regular moves
    for (int nSquares = 1; nSquares < 8; ++nSquares)
    {
      for (const auto &direction : std::vector<std::pair<int, int>>{
               {0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}})
      {
        codes[{nSquares * direction.first, nSquares * direction.second}] = i++;
      }
    }

    // 8 Knight moves
    codes[{1, 2}] = i++;
    codes[{2, 1}] = i++;
    codes[{2, -1}] = i++;
    codes[{1, -2}] = i++;
    codes[{-1, -2}] = i++;
    codes[{-2, -1}] = i++;
    codes[{-2, 1}] = i++;
    codes[{-1, 2}] = i++;
  }

  using Matrix3DMoves = std::array<std::array<std::array<int, 8>, 8>, 64>;


  Matrix3DMoves legal_moves_to_64_8_8(const Board &board)
  {
    Matrix3DMoves array6488 = {}; // Initialize to all zeros
    initialize_codes();
    // Get legal moves
//std::cout << board.chess960() << std::endl;
    Movelist legal_moves;
    movegen::legalmoves(legal_moves, board);

    for (auto move : legal_moves)
    { // square.rank();
    if (move.typeOf() == Move::CASTLING) {
   //   std::cout << "enroque"  << move << std::endl;
        std::string move_str = uci::moveToUci(move);
        std::string from_str = move_str.substr(0, 2);
    std::string to_str = move_str.substr(2, 2);
        Square from_sq = Square(from_str);
    Square to_sq = Square(to_str);

        move = Move::make<Move::NORMAL>(from_sq, to_sq);
     //   std::cout << "enroque"  << move << std::endl;
    }
      // std::cout << move << std::endl;
      int from_rank = move.from().rank();
      int from_file = move.from().file();
      int to_rank = move.to().rank();
      int to_file = move.to().file();

     /* if (move.typeOf() == Move::CASTLING) {
        
        // Create UCI string representation of the move (e.g., "e8a8", "e8h8")
            std::string move_str = uci::moveToUci(move);
std::cout << "enroque" << to_rank << move_str << std::endl;
            if (move_str == "e8a8") {
                to_rank = 2;
            }
            else if (move_str == "e8h8") {
                to_rank = 6;
            }
            else if (move_str == "e1a1") {
               to_rank = 2;
            }
            else if (move_str == "e1h1") {
                to_rank = 6;
            }
            std::cout << "enroque" << to_rank << std::endl;
    }*/


      // Compute encoded move
      std::pair<int, int> move_vector = {to_file - from_file, to_rank - from_rank};
      // std::cout << "(" << move_vector.first << ", " << move_vector.second << ")" << std::endl;

      // Check if move exists in `codes`
      if (codes.find(move_vector) != codes.end())
      {
        // std::cout << move << std::endl;
        int code_index = codes[move_vector];
        array6488[code_index][7 - from_rank][from_file] = 1;
      }
    }

    return array6488;
  }

  // Movement encoding map (should be initialized globally)
  extern std::map<std::pair<int, int>, int> codes;

  int uci_to_number(const char *uci_move)
  {
    // Move m = uci::uciToMove(board, uci_move);
    // std::cout << uci_move;
    initialize_codes();

    int from_file = uci_move[0] - 'a'; // 'a' -> 0, 'b' -> 1, ..., 'h' -> 7
    int from_rank = uci_move[1] - '1'; // '1' -> 0, '2' -> 1, ..., '8' -> 7
    int to_file = uci_move[2] - 'a';
    int to_rank = uci_move[3] - '1';

    // std::cout << uci_move << " " << to_file - from_file << " " << to_rank - from_rank;

    // Compute move vector (delta file, delta rank)
    std::pair<int, int> move_vector = {to_file - from_file, to_rank - from_rank};

    // Lookup move code
    if (codes.find(move_vector) == codes.end())
    {
      //std::cerr << "Error: Move not found in codes!" << std::endl;
      return -1;
    }
    int move_code = codes[move_vector];

    // Compute flattened index (64 * 8 * rank + 8 * file + move_code)
    int pos = (move_code * 8 * 8) + ((7 - from_rank) * 8) + from_file;

    return pos;
  }

  const char *number_to_uci(int number_move)
  {
    // Extract move_code, from_rank, and from_file from the number
    int move_code = number_move / (8 * 8);
    int from_rank = 7 - (number_move / 8) % 8; // Reverse row index
    int from_file = number_move % 8;           // File index

    // Find the corresponding move vector (delta file, delta rank)
    std::pair<int, int> code;
    for (const auto &[key, value] : codes)
    {
      if (value == move_code)
      {
        code = key;
        break;
      }
    }

    // Convert rank & file to UCI notation (e.g., e2e4)
    static char uci_move[5]; // 4 characters + null terminator
    uci_move[0] = 'a' + from_file;
    uci_move[1] = '1' + from_rank;
    uci_move[2] = 'a' + from_file + code.first;
    uci_move[3] = '1' + from_rank + code.second;
    uci_move[4] = '\0'; // Null-terminate

    return uci_move;
  }

  using Matrix3D77 = std::array<std::array<std::array<int, 8>, 8>, 77>;
  using PackedArray = std::array<uint8_t, 616>;
  PackedArray packed_data = {};

  const PackedArray* concat_fen_legal(const char *fen)
  {
    packed_data = {};
    std::string fen_string(fen);
   // std::cerr << "fen:" << fen_string << std::endl;

    chess::Board board(fen_string); // Load FEN into board

    // Get board representation (13,8,8) and legal moves (64,8,8)
    auto fen_matrix = board_to_3D_matrix(board);
    auto legal_moves = legal_moves_to_64_8_8(board);

    // Concatenate into (77,8,8)
    // Matrix3D77 fen_matrix_legal_moves = {};
    auto *fen_matrix_legal_moves = new Matrix3D77();
    // Copy 13 board layers
    for (int i = 0; i < 13; ++i)
    {
      (*fen_matrix_legal_moves)[i] = fen_matrix[i];
    }

    // Copy 64 legal move layers
    for (int i = 0; i < 64; ++i)
    {
      (*fen_matrix_legal_moves)[13 + i] = legal_moves[i];
    }

    //std::cout << "Concatenated board and legal moves generated!" << std::endl;

    //return &(*fen_matrix_legal_moves)[0][0][0];
    int bit_index = 0;
    for (int i = 0; i < 77; i++) {
        for (int j = 0; j < 8; j++) {
            for (int k = 0; k < 8; k++) {
                int byte_pos = bit_index / 8;
                int bit_pos = bit_index % 8;

                if ((*fen_matrix_legal_moves)[i][j][k]) {
                    packed_data[byte_pos] |= (1 << bit_pos);
                }
                bit_index++;
            }
        }
    }

    return &packed_data;
  }

  int add(int a, int b)
  {
    return a + b;
  }
}

/*
int main () {


    initialize_codes();


    //////////// Pruebas





    Board board = Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    Movelist moves;
    movegen::legalmoves(moves, board);

    for (const auto &move : moves) {
        std::cout << uci::moveToUci(move) << std::endl;
    }

    //chess::Position board;  // Standard chess starting position
    Matrix matrix = board_to_matrix(board, PieceType::KNIGHT); // Example for knights

    // Print the matrix
    for (const auto& row : matrix) {
        for (int cell : row) {
            std::cout << cell << " ";
        }
        std::cout << std::endl;
    }


    Matrix3D m3D = board_to_3D_matrix(board);
    std::cout << "Board representation has " << m3D.size() << " layers." << std::endl;
    for(const auto& matrix: m3D ){
    for (const auto& row : matrix) {
        for (int cell : row) {
            std::cout << cell << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << std::endl;
}

    // Print the codes (for testing)
    for (const auto& [key, value] : codes) {
        std::cout << "(" << key.first << ", " << key.second << ") -> " << value << std::endl;
    }


Matrix3DMoves result = legal_moves_to_64_8_8(board);

    // Print example output (nonzero entries)
    for (int i = 0; i < 64; ++i) {
        for (int r = 0; r < 8; ++r) {
            for (int c = 0; c < 8; ++c) {
                if (result[i][r][c] != 0) {
                    std::cout << "Move " << i << " at (" << r << ", " << c << ") -> " << result[i][r][c] << std::endl;
                }
            }
        }
    }

        std::string uci_move = "e2e4";
    int move_index = uci_to_number(board,uci_move);
    std::cout << "Move index: " << move_index << std::endl;



 int number_move = 564;  // Example move index
    std::string uci_move2 = number_to_uci(number_move);
    std::cout << "UCI Move: " << uci_move2 << std::endl;



std::string fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";  // Starting position

    Matrix3D77 result2 = concat_fen_legal(fen);
    std::cout << "Concatenated board and legal moves generated!" << std::endl;


  for(const auto& matrix: result2 ){
    for (const auto& row : matrix) {
        for (int cell : row) {
            std::cout << cell << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << std::endl;
}

    return 0;
}
*/
