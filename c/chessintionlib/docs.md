https://github.com/Disservin/chess-library
https://mesonbuild.com/

meson setup build
meson compile -C build

compile in ubuntu 22.04 for libraries

board = chess.Board()
board.set_fen(positions[frame]) 



Parece que **`concat_fen_legal_bits` está tardando 1.3 segundos en ejecutarse** dentro de tu pipeline de Monte Carlo Tree Search (MCTS). Esto sugiere que la conversión de la representación FEN a un tensor binario comprimido es un cuello de botella notable en el rendimiento.  

---

### **Posibles optimizaciones**
1. **Evitar `new` y usar arrays estáticos**  
   - Actualmente, en `concat_fen_legal`, creas dinámicamente `fen_matrix_legal_moves` con `new Matrix3D77()`, lo cual puede ser costoso.  
   - Mejor usa una **variable local estática** para evitar asignaciones dinámicas.

2. **Reducir copias de memoria**  
   - **`std::copy(fen_matrix.begin(), fen_matrix.end(), combinedArray.begin());`** y **`std::copy(legal_moves.begin(), legal_moves.end(), combinedArray.begin() + fen_matrix.size());`** hacen copias innecesarias.  
   - Si los datos ya están organizados en memoria contigua, podrías usar **pointers o `std::memcpy`** en lugar de `std::copy`.

3. **Optimizar la conversión de `uint64_t` a `uint8_t`**  
   - Actualmente, **el bucle que descompone los `uint64_t` en bytes es costoso**.  
   - En lugar de iterar sobre cada byte con `>>` y `&`, podrías hacer **una reinterpretación de memoria (`reinterpret_cast`)** si la alineación lo permite.

4. **Vectorización con SIMD (AVX2, NEON, etc.)**  
   - Si tu CPU soporta instrucciones vectoriales (SIMD), puedes paralelizar la extracción de bytes con AVX2 en x86 o NEON en ARM.

5. **Uso de OpenMP o paralelización con CUDA**  
   - Si `concat_fen_legal_bits` es un cuello de botella importante, podrías paralelizar la conversión usando **OpenMP** (para CPU) o incluso mover la conversión a **CUDA** si estás usando `torch.cuda`.

---

### **Próximos pasos**
1. **Mide el impacto real de `concat_fen_legal_bits`**:  
   - Si es solo **1.3s de 18.4s (~7%)**, tal vez no sea el problema principal.
   - Usa `cProfile` o `torch.profiler` para analizar cuellos de botella más profundos.

2. **Aplica optimizaciones progresivamente**:  
   - Prueba primero la eliminación de `new` y la optimización de copias.
   - Luego intenta `memcpy` y reinterpretación de memoria.
   - Si aún es lento, investiga SIMD y paralelización.

¿Quieres ayuda con una versión optimizada del código? 🚀