#include <vector>
#include <map>
#include <algorithm>
#include <utility>
#include <iostream>
#include <mpi.h>
#include <cassert>
#include "functions.h"

void flatten_matrix(std::vector<std::pair<std::pair<int, int>, int>> &matrix,
                    std::vector<int> &flattened)
{
    flattened.resize(matrix.size() * 3);

    int i = 0;
    for (auto [idx, value] : matrix) {
        flattened[i++] = idx.first;
        flattened[i++] = idx.second;
        flattened[i++] = value;
    }
}

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

    // flatten matrices first
    std::vector<int> flattened_A, flattened_B;
    flatten_matrix(A, flattened_A);
    flatten_matrix(B, flattened_B);

    std::map<std::pair<int, int>, int> C_map;
    for (int i = 0; i < q; ++i) {
        // bcast A block from procs in col i to its entire row
        int A_size = (i == pc) ? static_cast<int>(flattened_A.size()) : 0;
        MPI_Bcast(&A_size, 1, MPI_INT, i, row_comm);
        std::vector<int> recv_A_flat(A_size);
        if (i == pc) {
            recv_A_flat = flattened_A;
        }
        MPI_Bcast(recv_A_flat.data(), A_size, MPI_INT, i, row_comm);

        // bcast B block from procs in row i to its column
        int B_size = (i == pr) ? static_cast<int>(flattened_B.size()) : 0;
        MPI_Bcast(&B_size, 1, MPI_INT, i, col_comm);
        std::vector<int> recv_B_flat(B_size);
        if (i == pr) {
            recv_B_flat = flattened_B;
        }
        MPI_Bcast(recv_B_flat.data(), B_size, MPI_INT, i, col_comm);

        // sparse matrix multiplication
        // separate B by row for quick lookup
        std::map<int, std::vector<std::pair<int, int>>> B_by_row;
        for (int j = 0; j < B_size; j += 3) {
            int b_i = recv_B_flat[j];
            int b_j = recv_B_flat[j + 1];
            int b_w = recv_B_flat[j + 2];
            B_by_row[b_i].push_back({b_j, b_w});
        }

        // local computation loop
        for (int i = 0; i < A_size; i += 3) {
            int a_i = recv_A_flat[i];
            int a_j = recv_A_flat[i + 1];
            int a_w = recv_A_flat[i + 2];

            // find matching entries in B
            auto it = B_by_row.find(a_j);
            if (it != B_by_row.end()) {
                const auto &b_entries = it->second;
                for (const auto &b_entry : b_entries) {
                    int b_j = b_entry.first;
                    int b_w = b_entry.second;

                    // update C_map
                    std::pair<int, int> c_idx = {a_i, b_j};
                    int c_w = times(a_w, b_w);
                    if (C_map.find(c_idx) != C_map.end()) {
                        C_map[c_idx] = plus(C_map[c_idx], c_w);
                    } else {
                        C_map[c_idx] = c_w;
                    }
                }
            }
        }
    }

    // convert map to output vector
    C.clear();
    for (const auto &entry : C_map) {
        C.push_back({entry.first, entry.second});
    }
}
