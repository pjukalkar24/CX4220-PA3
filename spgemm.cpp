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

    // cannon's algorithm
    // shift A left pr positions
    for (int i = 0; i < pr; ++i) {
        int send_to = (pc - 1 + q) % q;
        int recv_from = (pc + 1) % q;

        int local_A_size = A.size();
        int recv_A_size;
        MPI_Sendrecv(&local_A_size, 1, MPI_INT, send_to, 0,
                     &recv_A_size, 1, MPI_INT, recv_from, 0, row_comm, MPI_STATUS_IGNORE);
        std::vector<std::pair<std::pair<int,int>, int>> temp_A;
        temp_A.resize(recv_A_size);
        MPI_Sendrecv(A.data(), local_A_size, mpi_entry_t, send_to, 0,
                     temp_A.data(), recv_A_size, mpi_entry_t, recv_from, 0, row_comm, MPI_STATUS_IGNORE);
        A = temp_A;
    }

    // shift B up pc positions
    for (int i = 0; i < pc; ++i) {
        int send_to = (pr - 1 + q) % q;
        int recv_from = (pr + 1) % q;

        int local_B_size = B.size();
        int recv_B_size;
        MPI_Sendrecv(&local_B_size, 1, MPI_INT, send_to, 0,
                     &recv_B_size, 1, MPI_INT, recv_from, 0, col_comm, MPI_STATUS_IGNORE);
        std::vector<std::pair<std::pair<int,int>, int>> temp_B;
        temp_B.resize(recv_B_size);
        MPI_Sendrecv(B.data(), local_B_size, mpi_entry_t, send_to, 0,
                     temp_B.data(), recv_B_size, mpi_entry_t, recv_from, 0, col_comm, MPI_STATUS_IGNORE);
        B = temp_B;
    }

    // loop q times:
    //     compute C += A*B
    //     shift A left by 1
    //     shift B up by 1
    std::map<std::pair<int, int>, int> C_map;
    for (int i = 0; i < q; ++i) {
        // compute local portion of C
        for (auto entry : A) {
            int a_i = entry.first.first;
            int a_j = entry.first.second;
            int a_w = entry.second;

            for (auto b_entry : B) {
                int b_i = b_entry.first.first;
                int b_j = b_entry.first.second;
                int b_w = b_entry.second;

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
        int local_A_size = A.size();
        int recv_A_size;
        MPI_Sendrecv(&local_A_size, 1, MPI_INT, send_to, 0,
                     &recv_A_size, 1, MPI_INT, recv_from, 0, row_comm, MPI_STATUS_IGNORE);
        std::vector<std::pair<std::pair<int,int>, int>> temp_A;
        temp_A.resize(recv_A_size);
        MPI_Sendrecv(A.data(), local_A_size, mpi_entry_t, send_to, 0,
                     temp_A.data(), recv_A_size, mpi_entry_t, recv_from, 0, row_comm, MPI_STATUS_IGNORE);
        A = temp_A;

        // shift B up 1
        send_to = (pr - 1 + q) % q;
        recv_from = (pr + 1) % q;
        int local_B_size = B.size();
        int recv_B_size;
        MPI_Sendrecv(&local_B_size, 1, MPI_INT, send_to, 0,
                     &recv_B_size, 1, MPI_INT, recv_from, 0, col_comm, MPI_STATUS_IGNORE);
        std::vector<std::pair<std::pair<int,int>, int>> temp_B;
        temp_B.resize(recv_B_size);
        MPI_Sendrecv(B.data(), local_B_size, mpi_entry_t, send_to, 0,
                     temp_B.data(), recv_B_size, mpi_entry_t, recv_from, 0, col_comm, MPI_STATUS_IGNORE);
        B = temp_B;
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
