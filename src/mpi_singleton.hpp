#pragma once
#include <mpi.h>
#include <stdexcept>

/**
 * MPIContext — Singleton that owns MPI_Init / MPI_Finalize.
 *
 * Safe to call from Python bindings: initialises MPI on first use,
 * finalises on program exit via static destructor.
 */
class MPIContext {
public:
    static MPIContext& instance() {
        static MPIContext inst;
        return inst;
    }

    int      rank() const { return rank_; }
    int      size() const { return size_; }
    MPI_Comm comm() const { return comm_; }

    // Non-copyable, non-movable
    MPIContext(const MPIContext&)            = delete;
    MPIContext& operator=(const MPIContext&) = delete;

private:
    MPIContext() {
        int flag = 0;
        MPI_Initialized(&flag);
        if (!flag) {
            int provided = 0;
            // Request thread-multiple so Python threads don't conflict
            MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
            owns_init_ = true;
        }
        comm_ = MPI_COMM_WORLD;
        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &size_);
    }

    ~MPIContext() {
        int flag = 0;
        MPI_Finalized(&flag);
        if (!flag && owns_init_) {
            MPI_Finalize();
        }
    }

    MPI_Comm comm_      = MPI_COMM_WORLD;
    int      rank_      = 0;
    int      size_      = 1;
    bool     owns_init_ = false;
};
