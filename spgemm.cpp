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

    // cannon's algorithm
    // flatten matrices first
    std::vector<int> flattened_A, flattened_B;
    flatten_matrix(A, flattened_A);
    flatten_matrix(B, flattened_B);

    // shift A left pr positions
    for (int i = 0; i < pr; ++i) {
        int send_to = (pc - 1 + q) % q;
        int recv_from = (pc + 1) % q;

        int local_A_size = flattened_A.size();
        int recv_A_size;
        MPI_Sendrecv(&local_A_size, 1, MPI_INT, send_to, 0,
                     &recv_A_size, 1, MPI_INT, recv_from, 0, row_comm, MPI_STATUS_IGNORE);
        std::vector<int> temp_A;
        temp_A.resize(recv_A_size);
        MPI_Sendrecv(flattened_A.data(), local_A_size, MPI_INT, send_to, 0,
                     temp_A.data(), recv_A_size, MPI_INT, recv_from, 0, row_comm, MPI_STATUS_IGNORE);
        flattened_A = temp_A;
    }

    // shift B up pc positions
    for (int i = 0; i < pc; ++i) {
        int send_to = (pr - 1 + q) % q;
        int recv_from = (pr + 1) % q;

        int local_B_size = flattened_B.size();
        int recv_B_size;
        MPI_Sendrecv(&local_B_size, 1, MPI_INT, send_to, 0,
                     &recv_B_size, 1, MPI_INT, recv_from, 0, col_comm, MPI_STATUS_IGNORE);
        std::vector<int> temp_B;
        temp_B.resize(recv_B_size);
        MPI_Sendrecv(flattened_B.data(), local_B_size, MPI_INT, send_to, 0,
                     temp_B.data(), recv_B_size, MPI_INT, recv_from, 0, col_comm, MPI_STATUS_IGNORE);
        flattened_B = temp_B;
    }

    // loop q times:
    //     compute C += A*B
    //     shift A left by 1
    //     shift B up by 1
    std::map<std::pair<int, int>, int> C_map;
    for (int i = 0; i < q; ++i) {
        // compute local portion of C
        for (int i = 0; i < flattened_A.size(); i += 3) {
            int a_i = flattened_A[i];
            int a_j = flattened_A[i + 1];
            int a_w = flattened_A[i + 2];

            for (int j = 0; j < flattened_B.size(); j += 3) {
                int b_i = flattened_B[j];
                int b_j = flattened_B[j + 1];
                int b_w = flattened_B[j + 2];

                if (a_j == b_i) {
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

        // shift A left 1
        int send_to = (pc - 1 + q) % q;
        int recv_from = (pc + 1) % q;
        int local_A_size = flattened_A.size();
        int recv_A_size;
        MPI_Sendrecv(&local_A_size, 1, MPI_INT, send_to, 0,
                     &recv_A_size, 1, MPI_INT, recv_from, 0, row_comm, MPI_STATUS_IGNORE);
        std::vector<int> temp_A;
        temp_A.resize(recv_A_size);
        MPI_Sendrecv(flattened_A.data(), local_A_size, MPI_INT, send_to, 0,
                     temp_A.data(), recv_A_size, MPI_INT, recv_from, 0, row_comm, MPI_STATUS_IGNORE);
        flattened_A = temp_A;

        // shift B up 1
        send_to = (pr - 1 + q) % q;
        recv_from = (pr + 1) % q;
        int local_B_size = flattened_B.size();
        int recv_B_size;
        MPI_Sendrecv(&local_B_size, 1, MPI_INT, send_to, 0,
                     &recv_B_size, 1, MPI_INT, recv_from, 0, col_comm, MPI_STATUS_IGNORE);
        std::vector<int> temp_B;
        temp_B.resize(recv_B_size);
        MPI_Sendrecv(flattened_B.data(), local_B_size, MPI_INT, send_to, 0,
                     temp_B.data(), recv_B_size, MPI_INT, recv_from, 0, col_comm, MPI_STATUS_IGNORE);
        flattened_B = temp_B;
    }

    // // restore A and B
    // // shift A right pr positions
    // for (int i = 0; i < pr; ++i) {
    //     int send_to = (pc + 1) % q;
    //     int recv_from = (pc - 1 + q) % q;

    //     int local_A_size = A.size();
    //     int recv_A_size;
    //     MPI_Sendrecv(&local_A_size, 1, MPI_INT, send_to, 0,
    //                  &recv_A_size, 1, MPI_INT, recv_from, 0, row_comm, MPI_STATUS_IGNORE);
    //     std::vector<std::pair<std::pair<int,int>, int>> temp_A;
    //     temp_A.resize(recv_A_size);
    //     MPI_Sendrecv(A.data(), local_A_size, mpi_entry_t, send_to, 0,
    //                  temp_A.data(), recv_A_size, mpi_entry_t, recv_from, 0, row_comm, MPI_STATUS_IGNORE);
    //     A = temp_A;
    // }

    // // shift B down pc positions
    // for (int i = 0; i < pc; ++i) {
    //     int send_to = (pr + 1) % q;
    //     int recv_from = (pr - 1 + q) % q;

    //     int local_B_size = B.size();
    //     int recv_B_size;
    //     MPI_Sendrecv(&local_B_size, 1, MPI_INT, send_to, 0,
    //                  &recv_B_size, 1, MPI_INT, recv_from, 0, col_comm, MPI_STATUS_IGNORE);
    //     std::vector<std::pair<std::pair<int,int>, int>> temp_B;
    //     temp_B.resize(recv_B_size);
    //     MPI_Sendrecv(B.data(), local_B_size, mpi_entry_t, send_to, 0,
    //                  temp_B.data(), recv_B_size, mpi_entry_t, recv_from, 0, col_comm, MPI_STATUS_IGNORE);
    //     B = temp_B;
    // }

    // convert map to output vector
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
