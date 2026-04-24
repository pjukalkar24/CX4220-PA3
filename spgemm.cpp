#include <vector>
#include <map>
#include <algorithm>
#include <utility>
#include <iostream>
#include <mpi.h>
#include <cassert>
#include "functions.h"

void spgemm_2d(int m, int p, int n,
               std::vector<std::pair<std::pair<int,int>, int>> &A,
               std::vector<std::pair<std::pair<int,int>, int>> &B,
               std::vector<std::pair<std::pair<int,int>, int>> &C,
               std::function<int(int, int)> plus, std::function<int(int, int)> times,
               MPI_Comm row_comm, MPI_Comm col_comm)
{
    int q; int pr; int pc;
    MPI_Comm_size(row_comm, &q);
    MPI_Comm_rank(row_comm, &pc);
    MPI_Comm_rank(col_comm, &pr);

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

    // allgather A across the row
    // allgather sendcounts from each proc in row
    std::vector<int> A_sendcounts(q, 0);
    int A_local_size = A.size();
    MPI_Allgather(&A_local_size, 1, MPI_INT, A_sendcounts.data(), 1, MPI_INT, row_comm);
    // calculate displacements
    std::vector<int> A_displs(q, 0);
    int total_A_size = 0;
    for (int i = 0; i < q; i++) {
        A_displs[i] = total_A_size;
        total_A_size += A_sendcounts[i];
    }
    // allgatherv blocks of A across row
    std::vector<std::pair<std::pair<int,int>, int>> recv_A(total_A_size);
    MPI_Allgatherv(A.data(), A.size(), mpi_entry_t,
                    recv_A.data(), A_sendcounts.data(), A_displs.data(), mpi_entry_t, row_comm);


    // allgather B across the col
    // allgather sendcounts from each proc in col
    std::vector<int> B_sendcounts(q, 0);
    int B_local_size = B.size();
    MPI_Allgather(&B_local_size, 1, MPI_INT, B_sendcounts.data(), 1, MPI_INT, col_comm);
    // calculate displs
    std::vector<int> B_displs(q, 0);
    int total_B_size = 0;
    for (int i = 0; i < q; i++) {
        B_displs[i] = total_B_size;
        total_B_size += B_sendcounts[i];
    }
    // allgatherv blocks of B across col
    std::vector<std::pair<std::pair<int,int>, int>> recv_B(total_B_size);
    MPI_Allgatherv(B.data(), B.size(), mpi_entry_t,
                    recv_B.data(), B_sendcounts.data(), B_displs.data(), mpi_entry_t, col_comm);

    // local matrix multiplication
    std::map<std::pair<int, int>, int> C_map;
    for (const auto &a_entry : recv_A) {
        int a_i = a_entry.first.first;
        int a_k = a_entry.first.second;
        int a_w = a_entry.second;

        for (const auto &b_entry : recv_B) {
            int b_k = b_entry.first.first;
            int b_j = b_entry.first.second;
            int b_w = b_entry.second;

            if (a_k == b_k) {
                std::pair<int, int> c_idx = {a_i, b_j};
                int c_val = times(a_w, b_w);
                if (C_map.find(c_idx) != C_map.end()) {
                    C_map[c_idx] = plus(C_map[c_idx], c_val);
                } else {
                    C_map[c_idx] = c_val;
                }
            }
        }
    }

    C.clear();
    for (const auto &entry : C_map) {
        C.push_back({entry.first, entry.second});
    }

    // if (pr == 1 && pc == 0) {
    //     std::cout << "blocks of A received: " << recv_A.size() << "\n";
    //     std::cout << "data received: " << "[";
    //     for (const auto &entry : recv_A) {
    //         std::cout << "((" << entry.first.first << "," << entry.first.second << ")," << entry.second << "), ";
    //     }
    //     std::cout << "]\n\n";

    //     std::cout << "blocks of B received: " << recv_B.size() << "\n";
    //     std::cout << "data received: " << "[";
    //     for (const auto &entry : recv_B) {
    //         std::cout << "((" << entry.first.first << "," << entry.first.second << ")," << entry.second << "), ";
    //     }
    //     std::cout << "]\n";
    // }
}
