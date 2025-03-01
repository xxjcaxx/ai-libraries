project('chessintionlib', 'cpp',
  version : '0.1',
  default_options : ['warning_level=3']
)

glibc_static_path = '/opt/glibc-2.38'  # Path to your static GLIBC

# Create a shared library without statically linking libc (GLIBC)
shared_library('chessintionlib',
    sources: ['chessintionlib.cpp'],
    install: false,
    link_args: [
        '-static-libstdc++',        # Statically link the C++ standard library
        '-static-libgcc',           # Statically link GCC library
        '-L' + glibc_static_path + '/lib',  # Path to the static GLIBC
        '-lc'                       # Link dynamically to libc (GLIBC)
    ]
)
