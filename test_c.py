import ctypes
import numpy as np

chess_extension = ctypes.CDLL('./c/chessintionlib/build/libchessintionlib.so') 


chess_extension.concat_fen_legal.argtypes = [ctypes.c_char_p]
#chess_extension.concat_fen_legal.restype = ctypes.POINTER(ctypes.c_int * 77*8*8)  # This should return a pointer to an array
chess_extension.concat_fen_legal.restype = ctypes.POINTER(ctypes.c_uint8 * 616)

chess_extension.concat_fen_legal_bits.argtypes = [ctypes.c_char_p]
chess_extension.concat_fen_legal_bits.restype = ctypes.POINTER(ctypes.c_uint8 * 616)

def concat_fen_legal(fen):
    fen_bytes = fen.encode('utf-8')
    # Call concat_fen_legal from the shared library
    result_ptr = chess_extension.concat_fen_legal(fen_bytes)
    compressed_array = np.frombuffer(result_ptr.contents, dtype=np.uint8)
    bit_array = np.unpackbits(compressed_array).astype(np.uint8)
    array_np = bit_array.reshape(77, 8, 8)
    np.set_printoptions(threshold=np.inf)
    print(array_np)

#concat_fen_legal('3k4/1K3B2/1BP5/1n3p2/7p/7P/7b/8 b - - 4 3')

def concat_fen_legal_bits(fen):
    fen_bytes = fen.encode('utf-8')
    
    result_ptr = chess_extension.concat_fen_legal(fen_bytes)
    compressed_array = np.frombuffer(result_ptr.contents, dtype=np.uint8)
    bit_array = np.unpackbits(compressed_array).astype(np.uint8)
    array_np = bit_array.reshape(77, 8, 8)
    np.set_printoptions(threshold=np.inf)
    print(array_np)

    return array_np



concat_fen_legal_bits('3k4/1K3B2/1BP5/1n3p2/7p/7P/7b/8 b - - 4 3')