project('chessintionlib', 'cpp',
  version : '0.1',
  default_options : ['warning_level=3']
)

glibc_static_path = '/usr/lib/x86_64-linux-gnu'


# Create a shared library without statically linking libc (GLIBC)
shared_library('chessintionlib',
    sources: ['chessintionlib.cpp'],
    install: false,
    link_args: [
        '-static-libstdc++',        # Statically link the C++ standard library
        '-static-libgcc',           # Statically link GCC library
        '-static',                  # Statically link everything (including libc)
        '-L' + glibc_static_path + '/lib',  # Path to the static GLIBC (libc.a)
        '-lc'                       # Link statically to libc (GLIBC)
    ]
)

