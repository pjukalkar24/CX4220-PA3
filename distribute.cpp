#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <string>
#include <sstream>
#include "functions.h"

void distribute_matrix_2d(int m, int n, std::vector<std::pair<std::pair<int, int>, int>> &full_matrix,
                          std::vector<std::pair<std::pair<int, int>, int>> &local_matrix,
                          int root, MPI_Comm comm_2d)
{
    // m, n: rows x cols
    // full_matrix: [((i,j), w), ...]
    // local_matrix: [((i,j), w), ...]

    int rank; int size;
    MPI_Comm_rank(comm_2d, &rank);
    MPI_Comm_size(comm_2d, &size);
    int q = sqrt(size);

    // create a custom MPI datatype for each entry: ((i,j), w)
    MPI_Datatype mpi_tmp_t;
    MPI_Datatype mpi_entry_t;
    int blocklengths[] = {1, 1, 1};
    MPI_Datatype types[] = {MPI_INT, MPI_INT, MPI_INT};

    // get displacements (architecture dependent)
    std::pair<std::pair<int, int>, int> dummy = {{0, 0}, 0};
    MPI_Aint base, addr1, addr2, addr3;
    MPI_Get_address(&dummy, &base);
    MPI_Get_address(&dummy.first.first, &addr1);
    MPI_Get_address(&dummy.first.second, &addr2);
    MPI_Get_address(&dummy.second, &addr3);
    MPI_Aint displacements[] = {
        addr1 - base,
        addr2 - base,
        addr3 - base
    };
    MPI_Aint extend = sizeof(dummy);
    MPI_Type_create_struct(3, blocklengths, displacements, types, &mpi_tmp_t);
    MPI_Type_create_resized(mpi_tmp_t, 0, extend, &mpi_entry_t);
    MPI_Type_commit(&mpi_entry_t);

    // distribute the full matrix across all processors, 2D partitioning
    std::vector<int> sendcounts(size, 0);
    std::vector<int> displs(size, 0);
    std::vector<std::pair<std::pair<int, int>, int>> sendbuf;
    sendbuf.reserve(full_matrix.size());

    if (rank == 0) {
        std::vector<std::vector<std::pair<std::pair<int, int>, int>>> partitioned(size);

        // fill in sendcounts, displs, and partitioned
        for (auto entry : full_matrix) {
            // get row and column of element
            int r = entry.first.first; int c = entry.first.second;

            // get destination processor in grid
            int pr = r * q / m;
            int pc = c * q / n;

            // get rank of destination processor
            int dest_coords[] = {pr, pc};
            int dest_rank;
            MPI_Cart_rank(comm_2d, dest_coords, &dest_rank);

            // update sendcounts, displacements
            ++sendcounts[dest_rank];
            partitioned[dest_rank].push_back(entry);
        }

        int curr_displ = 0;
        for (int i = 0; i < size; ++i) {
            displs[i] = curr_displ;
            sendbuf.insert(sendbuf.end(), partitioned[i].begin(), partitioned[i].end());
            curr_displ += sendcounts[i];
        }
    }

    // scatter sendcounts to all processors
    int recvcount;
    MPI_Scatter(sendcounts.data(), 1, MPI_INT, &recvcount, 1, MPI_INT, root, comm_2d);
    local_matrix.resize(recvcount);

    // scatter matrix entries to all processors
    MPI_Scatterv(sendbuf.data(), sendcounts.data(), displs.data(), mpi_entry_t,
                 local_matrix.data(), recvcount, mpi_entry_t, root, comm_2d);


    // if (m == 4) {
    //     std::cout << "==> A ";
    // } else {
    //     std::cout << "==> A_T ";
    // }

    // std::cout << "==> Rank " << rank << "; " << recvcount << " entries; " "local_matrix: [";
    // for (auto [idx, w] : local_matrix) {
    //     std::cout << "((" << idx.first << "," << idx.second << ")," << w << "), ";
    // }
    // std::cout << "]\n";
}