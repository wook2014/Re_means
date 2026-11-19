#!/bin/sh
set -eu

if ! command -v tar >/dev/null || ! command -v patch >/dev/null; then
    echo "Remeans needs tar and patch to build the bundled Faiss source." >&2
    exit 1
fi

case "${1:-}" in
    win)
        makevars=src/Makevars.win
        blas=-lopenblas
        ;;
    *)
        makevars=src/Makevars
        case "$(uname -s)" in
            Darwin) blas="-framework Accelerate" ;;
            *) blas=-lblas ;;
        esac
        ;;
esac

if [ -L src/.faiss ]; then
    echo "The Faiss build directory must be inside the package source." >&2
    exit 1
fi
mkdir -p src/.faiss
tar -xzf src/vendor/faiss-1.11.0.tar.gz -C src/.faiss
patch -s -p1 -d src/.faiss/faiss-1.11.0 < src/vendor/r-compat.patch
touch src/.faiss/.prepared
sed "s|@FAISS_BLAS@|$blas|" src/Makevars.in > "$makevars"
