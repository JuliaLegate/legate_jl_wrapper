using BinaryBuilder, Pkg

# needed for libjulia_platforms and julia_versions
include("../../L/libjulia/common.jl")


name = "legate_jl_wrapper"
version = v"25.05"
sources = [
    GitSource("https://github.com/JuliaLegate/legate_jl_wrapper.git","dde4d9dbb67653973619f63c9180d1b80e86032b"),
]


script = raw"""
    mkdir build
    cd build
    cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_FIND_ROOT_PATH=${prefix} \
        -DCMAKE_INSTALL_PREFIX=$prefix \
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TARGET_TOOLCHAIN} \
        -DJulia_PREFIX=${prefix} \
        ../legate_jl_wrapper/

    VERBOSE=ON cmake --build . --config Release --target install -- -j${nproc}
    install_license $WORKSPACE/srcdir/legate_jl_wrapper*/LICENSE.md
"""

platforms = vcat(libjulia_platforms.(julia_versions)...)
platforms = filter!(p -> arch(p) == "x86_64" || arch(p) == "aarch64", platforms)

products = [
    LibraryProduct("legate_jl_wrapper", :legate_jl_wrapper)
] 


dependencies = [
    Dependency("legate_jll"; compat = "=25.05"), # Legate versioning is Year.Month
    Dependency("libcxxwrap_julia_jll"; compat="0.14.3"),
    BuildDependency("libjulia_jll"),
]

build_tarballs(
    ARGS, 
    name, 
    version, 
    platform_sources, 
    script, platforms, 
    products, 
    dependencies;
    julia_compat = "1.10", 
    preferred_gcc_version = v"11",
)


