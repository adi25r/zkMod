cmake_minimum_required(VERSION 3.18)

# Self-contained GPU zkGPT fork.
# Builds a single binary `zkgpt_gpu` from zkgpt-gpu/src with the GKR GPU kernels,
# and (Phase 1+) the Blendy matmul + Lasso range-check GPU provers wired in.
# All integration source lives under zkgpt-gpu/ (src/ + gpu_provers/). The Blendy
# kernel headers (blendysc.hpp, blendy_linear.hpp, common.hpp) and common.cpp were
# vendored from core_work into gpu_provers/ + src/, so there is no core_work
# dependency. The only out-of-tree deps are sibling mcl/ and zkmod-gpu/.

# ROCm clang picks the highest-numbered /usr/lib/gcc/<triple>/<N> directory as its
# GCC installation regardless of whether libstdc++ headers are actually present
# there. On hosts with a compiler-only newer gcc package alongside a fully
# installed older one (e.g. gcc-16 with no libstdc++-16-dev, gcc-15 with
# libstdc++-15-dev), that picks the broken one and HIP compiles fail with
# "cmath not found" -- including CMake's own HIP-compiler viability check, which
# runs inside project()/enable_language(HIP), so this has to be set via
# CMAKE_HIP_FLAGS_INIT *before* project() rather than via CMAKE_HIP_FLAGS after.
find_program(ZKGPT_HOST_GXX NAMES g++ c++)
set(ZKGPT_GCC_INSTALL_FLAG "")
if(ZKGPT_HOST_GXX)
    execute_process(
        COMMAND ${ZKGPT_HOST_GXX} -print-libgcc-file-name
        OUTPUT_VARIABLE ZKGPT_LIBGCC_FILE
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    get_filename_component(ZKGPT_GCC_INSTALL_DIR "${ZKGPT_LIBGCC_FILE}" DIRECTORY)
    # ROCm clang only; nvcc hands its flags to the host gcc, which rejects it
    if(EXISTS "${ZKGPT_GCC_INSTALL_DIR}" AND NOT CMAKE_HIP_PLATFORM STREQUAL "nvidia")
        set(ZKGPT_GCC_INSTALL_FLAG "--gcc-install-dir=${ZKGPT_GCC_INSTALL_DIR}")
        set(CMAKE_HIP_FLAGS_INIT "${ZKGPT_GCC_INSTALL_FLAG}")
    endif()
endif()

project(zkgpt_gpu LANGUAGES CXX HIP)

set(OPT_FLAGS "-O2 -g")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${OPT_FLAGS}")
set(CMAKE_HIP_FLAGS "${OPT_FLAGS} ${ZKGPT_GCC_INSTALL_FLAG}")
# ZKGPT_HIP_EXTRA_FLAGS: appended verbatim (the line above overwrites CMAKE_HIP_FLAGS, so
# -D CMAKE_HIP_FLAGS=... on the command line is silently lost). Example for a ROCm tree
# whose clang has no omp.h and no -std default, borrowing ROCm 6.4.3's OpenMP headers:
#   ../zkmod build zkgpt  (the full configure line is rocm_cmake_args() in ../zkmod)
# Compile-only (CMAKE_HIP_FLAGS also reaches the link line, where "-x hip" would make
# clang parse the .o files as source). Applies to the zkmod-gpu sub-build too.
set(ZKGPT_HIP_EXTRA_FLAGS "" CACHE STRING "extra HIP compile flags (semicolon list) appended after the defaults")
if(ZKGPT_HIP_EXTRA_FLAGS)
    add_compile_options("$<$<COMPILE_LANGUAGE:HIP>:${ZKGPT_HIP_EXTRA_FLAGS}>")
endif()

if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES gfx90a)
endif()

# Pick the right relocatable-device-code flag depending on the HIP backend
# (AMD clang uses -fgpu-rdc; CUDA nvcc uses -rdc=true).
if(CMAKE_HIP_PLATFORM STREQUAL "nvidia" OR HIP_PLATFORM STREQUAL "nvidia"
   OR CMAKE_HIP_COMPILER_ID STREQUAL "NVIDIA" OR CMAKE_HIP_COMPILER MATCHES "nvcc")
    set(ZKGPT_RDC_FLAG -rdc=true)
    set(ZKGPT_HIP_NVCC ON)
else()
    set(ZKGPT_RDC_FLAG -fgpu-rdc)
endif()

# ── Link fix for the HIP-on-NVIDIA (nvcc) backend ───────────────────────────
# CMake performs device-side separable linking only on the AMD/clang HIP backend.
# On the nvcc backend it instead (a) routes the final executable link through the
# HOST compiler (g++) and (b) forwards nvcc's -rdc=true COMPILE flag to it, which
# g++ rejects — and even stripped, g++ cannot run the nvlink device-link pass that
# our cross-translation-unit __device__ symbols (blstrs__scalar__Scalar_add, …)
# require. So `HIP_SEPARABLE_COMPILATION ON` is a no-op here and the link breaks.
#
# Fix: drive the executable link through nvcc, which runs the device-link (nvlink)
# AND the host link in a single pass. Also translate CMake's `-Wl,-rpath,<dir>`
# rpath emission into nvcc's `-Xlinker` form (nvcc fatals on bare -Wl,...).
if(ZKGPT_HIP_NVCC)
    # nvcc's default host compiler (/usr/bin/g++) ships an older libstdc++ that
    # lacks symbols (e.g. GLIBCXX_3.4.32) which dependencies like libmcl.so were
    # built against. nvcc is from the conda toolchain, whose c++ has the matching
    # libstdc++ — point nvcc's host link at it via -ccbin. (Overridable.)
    get_filename_component(_zkgpt_nvcc_dir "${CMAKE_HIP_COMPILER}" DIRECTORY)
    find_program(ZKGPT_NVCC_CCBIN
        NAMES x86_64-conda-linux-gnu-c++ c++ g++
        HINTS "${_zkgpt_nvcc_dir}" NO_DEFAULT_PATH)
    if(ZKGPT_NVCC_CCBIN)
        set(_zkgpt_ccbin "-ccbin \"${ZKGPT_NVCC_CCBIN}\"")
    else()
        set(_zkgpt_ccbin "")
    endif()

    set(CMAKE_HIP_LINK_EXECUTABLE
        "<CMAKE_HIP_COMPILER> ${_zkgpt_ccbin} <FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
    # nvcc cannot read CMake's `@objects.rsp` / `@libs.rsp` response files, so put
    # objects and libraries directly on the command line.
    set(CMAKE_HIP_USE_RESPONSE_FILE_FOR_OBJECTS   0)
    set(CMAKE_HIP_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
    set(CMAKE_HIP_USE_RESPONSE_FILE_FOR_INCLUDES  0)
    # nvcc fatals on a bare `-Wl,-rpath,<dir>`; emit rpath in its `-Xlinker` form.
    # Executables use CMAKE_EXECUTABLE_RUNTIME_HIP_FLAG (seeded from the shared-lib
    # flag during project()), so both must be overridden.
    set(CMAKE_SHARED_LIBRARY_RUNTIME_HIP_FLAG     "-Xlinker -rpath -Xlinker ")
    set(CMAKE_SHARED_LIBRARY_RUNTIME_HIP_FLAG_SEP "")
    set(CMAKE_EXECUTABLE_RUNTIME_HIP_FLAG         "-Xlinker -rpath -Xlinker ")
    set(CMAKE_EXECUTABLE_RUNTIME_HIP_FLAG_SEP     "")
endif()

if(NOT DEFINED HIP_DIR)
    if(EXISTS "/opt/rocm/lib/cmake/hip")
        set(HIP_DIR "/opt/rocm/lib/cmake/hip" CACHE PATH "Path to HIP CMake package")
    endif()
endif()
if(DEFINED HIP_DIR)
    list(FIND CMAKE_PREFIX_PATH "${HIP_DIR}" _hip_in_prefix)
    if(${_hip_in_prefix} EQUAL -1)
        list(APPEND CMAKE_PREFIX_PATH "${HIP_DIR}")
    endif()
endif()

# Repo root is the parent of this folder (…/aditya).
set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(MCL_INCLUDE_DIR ${ROOT_DIR}/mcl/include)
set(MCL_LIB_DIR     ${ROOT_DIR}/mcl/lib)

find_library(MCL_LIBRARY NAMES mcl PATHS ${MCL_LIB_DIR} NO_DEFAULT_PATH REQUIRED)
add_library(mcl SHARED IMPORTED)
set_target_properties(mcl PROPERTIES
    IMPORTED_LOCATION             "${MCL_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${MCL_INCLUDE_DIR}"
)

# bls12_381_core: GPU field/curve kernels (blstrs scalar bridged to BN254 via gpu_field.hpp).
add_subdirectory(${ROOT_DIR}/zkmod-gpu zkmod-gpu_build)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(ZKGPT_COMMON_SRC
    ${SRC_DIR}/config.cpp
    ${SRC_DIR}/circuit.cpp
    ${SRC_DIR}/hyrax.cpp
    ${SRC_DIR}/hyrax_rp.cpp
    ${SRC_DIR}/prover.cpp
    ${SRC_DIR}/verifier.cpp
    ${SRC_DIR}/range_prover.cpp
    ${SRC_DIR}/neuralNetwork.cpp
    ${SRC_DIR}/models.cpp
    ${SRC_DIR}/polynomial.cpp
    ${SRC_DIR}/utils.cpp
    ${SRC_DIR}/timer.cpp
    ${SRC_DIR}/gkr_gpu.cpp
    ${SRC_DIR}/gkr_pred_gpu.cpp   # GPU predicate reduction (generic-GKR finalize buckets)
    ${SRC_DIR}/common.cpp
)

set(ZKGPT_GPU_SRC
    ${SRC_DIR}/main_gkr.cpp
    ${SRC_DIR}/config.cpp
    ${SRC_DIR}/circuit.cpp
    ${SRC_DIR}/hyrax.cpp
    ${SRC_DIR}/hyrax_rp.cpp
    ${SRC_DIR}/prover.cpp
    ${SRC_DIR}/verifier.cpp
    ${SRC_DIR}/range_prover.cpp        # NOVEL: GPU chunked-lookup range check (logup)
    ${SRC_DIR}/neuralNetwork.cpp
    ${SRC_DIR}/models.cpp
    ${SRC_DIR}/polynomial.cpp
    ${SRC_DIR}/utils.cpp
    ${SRC_DIR}/timer.cpp
    ${SRC_DIR}/gkr_gpu.cpp
    ${SRC_DIR}/gkr_pred_gpu.cpp   # GPU predicate reduction (generic-GKR finalize buckets)
    ${SRC_DIR}/common.cpp                           # helpers used by Blendy kernels (vendored)
)

set_source_files_properties(${ZKGPT_GPU_SRC} PROPERTIES LANGUAGE HIP)

add_executable(zkgpt_gpu ${ZKGPT_GPU_SRC})
target_compile_definitions(zkgpt_gpu PRIVATE MCL_FP_BIT=384 ZKGPT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(zkgpt_gpu PRIVATE bls12_381_core mcl pthread gomp)
target_include_directories(zkgpt_gpu PRIVATE
    ${SRC_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu_provers
    ${CMAKE_CURRENT_SOURCE_DIR}/witgen_gpu   # NOVEL: GPU witness generation  # NOVEL: GPU matmul/range bridges + vendored Blendy kernel headers
    ${ROOT_DIR}/zkmod-gpu/src            # bls12_381.hpp
    ${MCL_INCLUDE_DIR}
)
set_target_properties(zkgpt_gpu PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
    HIP_SEPARABLE_COMPILATION ON
    LINKER_LANGUAGE HIP)
target_compile_options(zkgpt_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:${ZKGPT_RDC_FLAG}> -fopenmp)
target_link_options(zkgpt_gpu PRIVATE -fopenmp)

# UVM variant: identical to zkgpt_gpu but the matmul product-sumcheck working
# buffers are allocated as managed memory (GKR_UVM). Experiment 3: lets a layer
# oversubscribe physical VRAM via page migration. Run with HSA_XNACK=1.
add_executable(zkgpt_gpu_uvm ${ZKGPT_GPU_SRC})
target_compile_definitions(zkgpt_gpu_uvm PRIVATE MCL_FP_BIT=384 GKR_UVM ZKGPT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(zkgpt_gpu_uvm PRIVATE bls12_381_core mcl pthread gomp)
target_include_directories(zkgpt_gpu_uvm PRIVATE
    ${SRC_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu_provers
    ${CMAKE_CURRENT_SOURCE_DIR}/witgen_gpu   # NOVEL: GPU witness generation
    ${ROOT_DIR}/zkmod-gpu/src
    ${MCL_INCLUDE_DIR}
)
set_target_properties(zkgpt_gpu_uvm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
    HIP_SEPARABLE_COMPILATION ON)
target_compile_options(zkgpt_gpu_uvm PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:${ZKGPT_RDC_FLAG}> -fopenmp)
target_link_options(zkgpt_gpu_uvm PRIVATE -fopenmp)

# Split-block proof runner.
set(ZKGPT_SPLIT_SRC ${SRC_DIR}/main_gkr_split.cpp ${SRC_DIR}/alloc_trace.cpp ${ZKGPT_COMMON_SRC})
set_source_files_properties(${ZKGPT_SPLIT_SRC} PROPERTIES LANGUAGE HIP)
add_executable(main_gkr_split ${ZKGPT_SPLIT_SRC})
target_compile_definitions(main_gkr_split PRIVATE MCL_FP_BIT=384 ZKGPT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(main_gkr_split PRIVATE bls12_381_core mcl pthread gomp)
target_include_directories(main_gkr_split PRIVATE
    ${SRC_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu_provers
    ${CMAKE_CURRENT_SOURCE_DIR}/witgen_gpu   # NOVEL: GPU witness generation
    ${ROOT_DIR}/zkmod-gpu/src
    ${MCL_INCLUDE_DIR}
)
set_target_properties(main_gkr_split PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
    HIP_SEPARABLE_COMPILATION ON)
# -fopenmp at COMPILE time (not just the gomp link) -- without it every
# `#pragma omp` in the tree is silently a no-op. clang's runtime is libomp (__kmpc_*), not
# libgomp, so the flag must reach the link line too.
target_compile_options(main_gkr_split PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:${ZKGPT_RDC_FLAG}> -fopenmp)
target_link_options(main_gkr_split PRIVATE -fopenmp)
# GPU-witness-generation twin of main_gkr_split, as a SEPARATE binary for the same
# reason as the ROCTx one: a sweep is usually running off bin/main_gkr_split, and
# relinking it underneath a live process risks ETXTBSY. Identical sources and flags;
# the GPU witgen path is selected at RUNTIME by GKR_WITGEN_GPU, so this binary is
# byte-equivalent in behaviour to main_gkr_split when the gate is unset.
add_executable(main_gkr_split_witgen ${ZKGPT_SPLIT_SRC})
target_compile_definitions(main_gkr_split_witgen PRIVATE MCL_FP_BIT=384 ZKGPT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(main_gkr_split_witgen PRIVATE bls12_381_core mcl pthread gomp)
target_include_directories(main_gkr_split_witgen PRIVATE
    ${SRC_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu_provers
    ${CMAKE_CURRENT_SOURCE_DIR}/witgen_gpu   # NOVEL: GPU witness generation
    ${ROOT_DIR}/zkmod-gpu/src
    ${MCL_INCLUDE_DIR}
)
set_target_properties(main_gkr_split_witgen PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
    HIP_SEPARABLE_COMPILATION ON)
target_compile_options(main_gkr_split_witgen PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:${ZKGPT_RDC_FLAG}> -fopenmp)
target_link_options(main_gkr_split_witgen PRIVATE -fopenmp)

# ROCTx-instrumented twin of main_gkr_split, as a SEPARATE binary so a profiling build
# never silently replaces the one a sweep is running. Same sources, same flags, plus
# ZKGPT_ROCTX. Drive it with tools/run_profiled.sh.
add_executable(main_gkr_split_roctx ${ZKGPT_SPLIT_SRC})
target_compile_definitions(main_gkr_split_roctx PRIVATE
    MCL_FP_BIT=384 ZKGPT_ROCTX ZKGPT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(main_gkr_split_roctx PRIVATE bls12_381_core mcl pthread gomp
    /opt/rocm/lib/librocprofiler-sdk-roctx.so)
target_include_directories(main_gkr_split_roctx PRIVATE
    ${SRC_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu_provers
    ${CMAKE_CURRENT_SOURCE_DIR}/witgen_gpu   # NOVEL: GPU witness generation
    ${ROOT_DIR}/zkmod-gpu/src
    ${MCL_INCLUDE_DIR}
)
set_target_properties(main_gkr_split_roctx PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
    HIP_SEPARABLE_COMPILATION ON)
target_compile_options(main_gkr_split_roctx PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:${ZKGPT_RDC_FLAG}> -fopenmp)
target_link_options(main_gkr_split_roctx PRIVATE -fopenmp)

# UVM variant of the split runner. Needed because the monolithic zkgpt_gpu_uvm
# CANNOT build Llama-8B at depth 32 -- neuralNetwork.cpp asserts in
# calcInputLayer during circuit construction, before any proving happens. The
# split runner is the only path that builds 8B, so measuring UVM at 8B requires
# a managed-memory build of *it*, not of zkgpt_gpu.
add_executable(main_gkr_split_uvm ${ZKGPT_SPLIT_SRC})
target_compile_definitions(main_gkr_split_uvm PRIVATE MCL_FP_BIT=384 GKR_UVM ZKGPT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(main_gkr_split_uvm PRIVATE bls12_381_core mcl pthread gomp)
target_include_directories(main_gkr_split_uvm PRIVATE
    ${SRC_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu_provers
    ${CMAKE_CURRENT_SOURCE_DIR}/witgen_gpu   # NOVEL: GPU witness generation
    ${ROOT_DIR}/zkmod-gpu/src
    ${MCL_INCLUDE_DIR}
)
set_target_properties(main_gkr_split_uvm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
    HIP_SEPARABLE_COMPILATION ON)
target_compile_options(main_gkr_split_uvm PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:${ZKGPT_RDC_FLAG}> -fopenmp)
target_link_options(main_gkr_split_uvm PRIVATE -fopenmp)

# modular-zkp's split demo driver (LLMSplitBlock, probe + half/half layout, merge flag,
# merged range set), built against THIS repo's prover so the CPU and GPU arms share
# the paper repo's arithmetization exactly; only the prover kernels differ (env gates).
set(ZKGPT_DEMO_SPLIT_SRC ${SRC_DIR}/main_demo_llm_split.cpp ${SRC_DIR}/alloc_trace.cpp ${ZKGPT_COMMON_SRC})
set_source_files_properties(${ZKGPT_DEMO_SPLIT_SRC} PROPERTIES LANGUAGE HIP)
add_executable(main_demo_llm_split ${ZKGPT_DEMO_SPLIT_SRC})
target_compile_definitions(main_demo_llm_split PRIVATE MCL_FP_BIT=384 ZKGPT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(main_demo_llm_split PRIVATE bls12_381_core mcl pthread gomp)
target_include_directories(main_demo_llm_split PRIVATE
    ${SRC_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu_provers
    ${CMAKE_CURRENT_SOURCE_DIR}/witgen_gpu
    ${ROOT_DIR}/zkmod-gpu/src
    ${MCL_INCLUDE_DIR}
)
set_target_properties(main_demo_llm_split PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
    HIP_SEPARABLE_COMPILATION ON)
target_compile_options(main_demo_llm_split PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:${ZKGPT_RDC_FLAG}> -fopenmp)
target_link_options(main_demo_llm_split PRIVATE -fopenmp)

# debug twin (separate binary so a running measurement is never relinked underneath)
set(ZKGPT_DEMO_SPLIT_DBG_SRC ${SRC_DIR}/main_demo_llm_split.cpp ${SRC_DIR}/alloc_trace.cpp ${ZKGPT_COMMON_SRC})
set_source_files_properties(${ZKGPT_DEMO_SPLIT_DBG_SRC} PROPERTIES LANGUAGE HIP)
add_executable(main_demo_llm_split_dbg ${ZKGPT_DEMO_SPLIT_DBG_SRC})
target_compile_definitions(main_demo_llm_split_dbg PRIVATE MCL_FP_BIT=384 ZKGPT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(main_demo_llm_split_dbg PRIVATE bls12_381_core mcl pthread gomp)
target_include_directories(main_demo_llm_split_dbg PRIVATE
    ${SRC_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu_provers
    ${CMAKE_CURRENT_SOURCE_DIR}/witgen_gpu
    ${ROOT_DIR}/zkmod-gpu/src
    ${MCL_INCLUDE_DIR}
)
set_target_properties(main_demo_llm_split_dbg PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
    HIP_SEPARABLE_COMPILATION ON)
target_compile_options(main_demo_llm_split_dbg PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:${ZKGPT_RDC_FLAG}> -fopenmp)
target_link_options(main_demo_llm_split_dbg PRIVATE -fopenmp)

# alloc_trace.cpp replaces the global operator new, which nvcc treats as device-callable, where its `throw` is
# rejected. It is host-only code, so on the NVIDIA backend compile it as plain C++ (after the LANGUAGE HIP lists above).
if(ZKGPT_HIP_NVCC)
    set_source_files_properties(${SRC_DIR}/alloc_trace.cpp PROPERTIES LANGUAGE CXX)
endif()
