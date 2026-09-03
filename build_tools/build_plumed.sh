#!/bin/bash
# Shared PLUMED build steps, sourced by conda_install.sh, custom_install.sh and
# custom_install_sol.sh. All installers need an identical PLUMED, so the versions
# are pinned here in one place.

PLUMED_VERSION="v2.10.1"
OPENMM_PLUMED_VERSION="v2.1"

# macOS does not provide GNU nproc. All supported install routes already run
# inside a Python environment, so use Python's portable CPU-count API instead.
build_job_count() {
    python -c 'import os; count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(); print(count or 1)'
}

# build_linker_flags <system-name> <library-dir>
# Apple ld supports rpath but not GNU ld's rpath-link option.
build_linker_flags() {
    local system_name="$1"
    local library_dir="$2"

    if [[ "${system_name}" == "Darwin" ]]; then
        printf '%s\n' "-Wl,-rpath,${library_dir}"
    else
        printf '%s\n' "-Wl,-rpath-link,${library_dir} -Wl,-rpath,${library_dir}"
    fi
}

# find_plumed_kernel <library-dir> [system-name]
# PLUMED installs a .dylib on macOS and a .so on Linux. Prefer the native
# extension, but accept either so custom toolchains remain usable.
find_plumed_kernel() {
    local library_dir="$1"
    local system_name="${2:-$(uname -s)}"
    local -a extensions
    local extension
    local candidate

    if [[ "${system_name}" == "Darwin" ]]; then
        extensions=("dylib" "so")
    else
        extensions=("so" "dylib")
    fi
    for extension in "${extensions[@]}"; do
        candidate="${library_dir}/libplumedKernel.${extension}"
        if [[ -f "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    echo "Could not find libplumedKernel in ${library_dir}" >&2
    return 1
}

# build_plumed <work_dir>
# Compiles PLUMED (with the opes module) and the OpenMM-PLUMED plugin into $CONDA_PREFIX,
# cloning the sources into <work_dir>. Leaves the shell in <work_dir>.
build_plumed() {
    local work_dir="$1"
    local system_name
    local dependency_linker_flags=""
    local linker_flags

    system_name="$(uname -s)"
    if [[ "${system_name}" != "Darwin" ]]; then
        dependency_linker_flags="-Wl,-rpath-link,${CONDA_PREFIX}/lib"
    fi
    linker_flags="$(build_linker_flags "${system_name}" "${CONDA_PREFIX}/lib")"

    echo "=== Compiling PLUMED ${PLUMED_VERSION} ==="
    cd "${work_dir}"
    git clone --branch "${PLUMED_VERSION}" --depth 1 --filter=blob:none https://github.com/plumed/plumed2.git
    cd plumed2
    # libplumedKernel.so links conda's BLAS and libgomp from ${CONDA_PREFIX}/lib, but
    # ld does not search there when resolving that library's own dependencies at the
    # final `plumed` link (-rpath-link, the same fix conda-forge's recipe uses) or at
    # run time (-rpath, which also lets py-plumed load the kernel without
    # LD_LIBRARY_PATH). --disable-python stops the top-level make from also
    # building the Python interface in-tree: it would bake the source-tree kernel
    # path into build artifacts that build_py_plumed's pip install then reuses,
    # instead of the ${CONDA_PREFIX} kernel it bakes itself.
    ./configure --prefix="${CONDA_PREFIX}" --enable-modules=opes --disable-python \
        LDFLAGS="-L${CONDA_PREFIX}/lib -Wl,-rpath,${CONDA_PREFIX}/lib" \
        STATIC_LIBS="${dependency_linker_flags}"
    make -j"$(build_job_count)"
    make install

    export PLUMED_INCLUDE_DIR="${CONDA_PREFIX}/include/plumed"
    export PLUMED_LIBRARY_DIR="${CONDA_PREFIX}/lib"

    echo "=== Compiling OpenMM-PLUMED ${OPENMM_PLUMED_VERSION} ==="
    cd "${work_dir}"
    git clone --branch "${OPENMM_PLUMED_VERSION}" --depth 1 --filter=blob:none https://github.com/openmm/openmm-plumed.git
    cd openmm-plumed
    mkdir -p build && cd build
    # openmm-plumed v2.1 still declares CMAKE_MINIMUM_REQUIRED(VERSION 2.8), and CMake 4
    # -- what conda-forge now ships -- removed compatibility with anything below 3.5.
    # This restores it without unpinning the tag. The variable arrived in CMake 3.31 and
    # is merely an unused cache entry on older ones, so it is safe to pass either way.
    cmake .. \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_INSTALL_PREFIX="${CONDA_PREFIX}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DOPENMM_DIR="${CONDA_PREFIX}" \
        -DPLUMED_INCLUDE_DIR="${PLUMED_INCLUDE_DIR}" \
        -DPLUMED_LIBRARY_DIR="${PLUMED_LIBRARY_DIR}" \
        -DPYTHON_EXECUTABLE="$(which python)" \
        -DCMAKE_EXE_LINKER_FLAGS="${linker_flags}"
    make -j"$(build_job_count)"
    make install
    # openmm-plumed's setup.py imports numpy without declaring it as a build
    # dependency, so the isolated pip run inside `make PythonInstall` fails; use
    # the target only to generate the swig wrapper, then install against the
    # environment (which has numpy) instead. A swig failure swallowed by the
    # `|| true` still aborts the build at the pip step below.
    make PythonInstall || true
    cd python
    pip install . --no-build-isolation

    cd "${work_dir}"
}

# build_py_plumed <work_dir>
# Builds the PLUMED Python bindings (the `plumed` module) from the plumed2 sources
# that build_plumed left in <work_dir>, against the PLUMED installed in
# $CONDA_PREFIX. The kernel path is baked in as the default, so `import plumed`
# works without PLUMED_KERNEL being set. Requires cython. Leaves the shell in <work_dir>.
build_py_plumed() {
    local work_dir="$1"
    local plumed_kernel

    echo "=== Building py-plumed ${PLUMED_VERSION} ==="
    cd "${work_dir}/plumed2/python"
    # Stages ./PLUMED_VERSION and include/Plumed.h for setup.py.
    make pip
    plumed_kernel="$(find_plumed_kernel "${CONDA_PREFIX}/lib")"
    plumed_default_kernel="${plumed_kernel}" \
        pip install . --no-build-isolation

    cd "${work_dir}"
}
