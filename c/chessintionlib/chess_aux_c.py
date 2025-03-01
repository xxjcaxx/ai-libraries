import ctypes
import numpy as np

# Load the shared library (adjust path to the .so file)
chess_extension = ctypes.CDLL('./build/libchessintionlib.so')  # Adjust path to your compiled shared library

# Define the function signatures of the C++ functions in the shared library

# Define matrix types
Matrix3D77 = ctypes.POINTER(ctypes.c_int)

# Define function signatures
chess_extension.board_to_matrix.argtypes = [ctypes.c_void_p, ctypes.c_int]
chess_extension.board_to_matrix.restype = ctypes.POINTER(ctypes.c_int)

chess_extension.board_to_3D_matrix.argtypes = [ctypes.c_void_p]
chess_extension.board_to_3D_matrix.restype = ctypes.POINTER(ctypes.c_int)

chess_extension.legal_moves_to_64_8_8.argtypes = [ctypes.c_void_p]
chess_extension.legal_moves_to_64_8_8.restype = ctypes.POINTER(ctypes.c_int)

chess_extension.uci_to_number.argtypes = [ctypes.c_char_p]
chess_extension.uci_to_number.restype = ctypes.c_int

chess_extension.number_to_uci.argtypes = [ctypes.c_int]
chess_extension.number_to_uci.restype = ctypes.c_char_p


chess_extension.concat_fen_legal.argtypes = [ctypes.c_char_p]
chess_extension.concat_fen_legal.restype = ctypes.POINTER(ctypes.c_int * 77*8*8)  # This should return a pointer to an array



def uci_to_number(uci_move):
    return chess_extension.uci_to_number(uci_move.encode('utf-8'))

def number_to_uci(number_move):
    return chess_extension.number_to_uci(number_move).decode('utf-8')


def concat_fen_legal(fen):
    fen_bytes = fen.encode('utf-8')
    # Call concat_fen_legal from the shared library
    result_ptr = chess_extension.concat_fen_legal(fen_bytes)
    size = 77 * 8 * 8  # Total number of elements
    result_ptr = ctypes.cast(result_ptr, ctypes.POINTER(ctypes.c_int * size)).contents
    result_array = np.ctypeslib.as_array(result_ptr, shape=(77 * 8 * 8,))
    reshaped_result = result_array.reshape(77, 8, 8)
    return reshaped_result


print(uci_to_number('c2c3'))
print(number_to_uci(50))
print(concat_fen_legal('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'))
