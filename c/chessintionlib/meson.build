project('chessintionlib', 'cpp',
  version : '0.1',
  default_options : ['warning_level=3'])

glibc_static_path = '/opt/glibc-2.38'  # Change this to your GLIBC install path

shared_library('chessintionlib', sources: ['chessintionlib.cpp'],

    install: false,
    link_args: [
       '-static-libstdc++', '-static-libgcc',   # Statically link C++ and GCC
        '-Wl,-Bstatic', '-L' + glibc_static_path + '/lib', '-lc', '-Wl,-Bdynamic'
    ]  # Statically link stdc++
    )

