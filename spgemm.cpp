#include <vector>
#include <unordered_map>
#include <algorithm>
#include <utility>
#include <iostream>
#include <mpi.h>
#include <cassert>
#include "functions.h"

// Hash function for std::pair<int, int>
namespace std {
    template <>
    struct hash<std::pair<int, int>> {
        size_t operator()(const std::pair<int, int>& p) const {
            return hash<int>()(p.first) ^ (hash<int>()(p.second) << 1);
        }
    };
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

    // Cannon's Algorithm for 2D SpGEMM
    // Initialize result accumulator (sparse, as a map)
    std::unordered_map<std::pair<int, int>, int> C_map;

    // Step 1: Initial Skew Phase
    // Shift A left by pr positions within the row
    std::vector<std::pair<std::pair<int,int>, int>> A_current = A;
    for (int shift = 0; shift < pr; ++shift) {
        int send_to = (pc - 1 + q) % q;
        int recv_from = (pc + 1) % q;
        std::vector<std::pair<std::pair<int,int>, int>> A_recv;
        
        // MPI_Sendrecv to shift A left
        int send_size = A_current.size();
        int recv_size = 0;
        MPI_Sendrecv(&send_size, 1, MPI_INT, send_to, 0,
                     &recv_size, 1, MPI_INT, recv_from, 0, row_comm, MPI_STATUS_IGNORE);
        
        A_recv.resize(recv_size);
        if (send_size > 0 && recv_size > 0) {
            MPI_Sendrecv(A_current.data(), send_size, mpi_entry_t, send_to, 1,
                         A_recv.data(), recv_size, mpi_entry_t, recv_from, 1, row_comm, MPI_STATUS_IGNORE);
        } else if (recv_size > 0) {
            MPI_Recv(A_recv.data(), recv_size, mpi_entry_t, recv_from, 1, row_comm, MPI_STATUS_IGNORE);
        } else if (send_size > 0) {
            MPI_Send(A_current.data(), send_size, mpi_entry_t, send_to, 1, row_comm);
        }
        A_current = A_recv;
    }

    // Shift B up by pc positions within the column
    std::vector<std::pair<std::pair<int,int>, int>> B_current = B;
    for (int shift = 0; shift < pc; ++shift) {
        int send_to = (pr - 1 + q) % q;
        int recv_from = (pr + 1) % q;
        std::vector<std::pair<std::pair<int,int>, int>> B_recv;
        
        // MPI_Sendrecv to shift B up
        int send_size = B_current.size();
        int recv_size = 0;
        MPI_Sendrecv(&send_size, 1, MPI_INT, send_to, 0,
                     &recv_size, 1, MPI_INT, recv_from, 0, col_comm, MPI_STATUS_IGNORE);
        
        B_recv.resize(recv_size);
        if (send_size > 0 && recv_size > 0) {
            MPI_Sendrecv(B_current.data(), send_size, mpi_entry_t, send_to, 1,
                         B_recv.data(), recv_size, mpi_entry_t, recv_from, 1, col_comm, MPI_STATUS_IGNORE);
        } else if (recv_size > 0) {
            MPI_Recv(B_recv.data(), recv_size, mpi_entry_t, recv_from, 1, col_comm, MPI_STATUS_IGNORE);
        } else if (send_size > 0) {
            MPI_Send(B_current.data(), send_size, mpi_entry_t, send_to, 1, col_comm);
        }
        B_current = B_recv;
    }

    // Step 2: Main Computation Loop
    for (int k = 0; k < q; ++k) {
        // Compute C += A_current * B_current
        for (const auto &a_entry : A_current) {
            int a_i = a_entry.first.first;
            int a_k = a_entry.first.second;
            int a_w = a_entry.second;

            for (const auto &b_entry : B_current) {
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

        // Shift A one position left within the row
        if (k < q - 1) {
            int send_to = (pc - 1 + q) % q;
            int recv_from = (pc + 1) % q;
            std::vector<std::pair<std::pair<int,int>, int>> A_recv;
            
            int send_size = A_current.size();
            int recv_size = 0;
            MPI_Sendrecv(&send_size, 1, MPI_INT, send_to, 0,
                         &recv_size, 1, MPI_INT, recv_from, 0, row_comm, MPI_STATUS_IGNORE);
            
            A_recv.resize(recv_size);
            if (send_size > 0 && recv_size > 0) {
                MPI_Sendrecv(A_current.data(), send_size, mpi_entry_t, send_to, 1,
                             A_recv.data(), recv_size, mpi_entry_t, recv_from, 1, row_comm, MPI_STATUS_IGNORE);
            } else if (recv_size > 0) {
                MPI_Recv(A_recv.data(), recv_size, mpi_entry_t, recv_from, 1, row_comm, MPI_STATUS_IGNORE);
            } else if (send_size > 0) {
                MPI_Send(A_current.data(), send_size, mpi_entry_t, send_to, 1, row_comm);
            }
            A_current = A_recv;

            // Shift B one position up within the column
            int send_to_col = (pr - 1 + q) % q;
            int recv_from_col = (pr + 1) % q;
            std::vector<std::pair<std::pair<int,int>, int>> B_recv;
            
            send_size = B_current.size();
            recv_size = 0;
            MPI_Sendrecv(&send_size, 1, MPI_INT, send_to_col, 0,
                         &recv_size, 1, MPI_INT, recv_from_col, 0, col_comm, MPI_STATUS_IGNORE);
            
            B_recv.resize(recv_size);
            if (send_size > 0 && recv_size > 0) {
                MPI_Sendrecv(B_current.data(), send_size, mpi_entry_t, send_to_col, 1,
                             B_recv.data(), recv_size, mpi_entry_t, recv_from_col, 1, col_comm, MPI_STATUS_IGNORE);
            } else if (recv_size > 0) {
                MPI_Recv(B_recv.data(), recv_size, mpi_entry_t, recv_from_col, 1, col_comm, MPI_STATUS_IGNORE);
            } else if (send_size > 0) {
                MPI_Send(B_current.data(), send_size, mpi_entry_t, send_to_col, 1, col_comm);
            }
            B_current = B_recv;
        }
    }

    // Convert result map to output vector
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
