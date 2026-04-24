#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <string>
#include <sstream>
#include "functions.h"

// Read in a testfile for the spgemm 
void read_testfile(std::string &fname, std::string &test_type,
                   std::vector<std::pair<std::pair<int, int>, int>> &A,
                   std::vector<std::pair<std::pair<int, int>, int>> &expected)
{
    std::ifstream ifs {fname};
    std::string buffer;

    if (ifs.fail() || !ifs.is_open()) {
        std::cerr << "Failed to open file " << fname << "\n";
        return;
    }

    int step = 0;
    while (ifs.good()) {
        std::getline(ifs, buffer);
        if (buffer.length() == 0) {
            continue;
        }
        std::stringstream ss {buffer};
        char c = ss.get();

        if (c == 's') {
            ss.unget();
            ss >> test_type;
            assert(test_type == "spgemm");
            continue;
        }

        // Check for matrix delimeter
        if (c == '-') {
            ++step;
            continue;
        }

        // Skip comment lines and lines that don't start with a number
        if (c == '#' || std::isspace(c) || std::isalpha(c)) {
            continue;
        }
        ss.unget();

        // Extract edge i---j with weight w
        int i, j;
        ss >> i;
        ss >> j;
        int w;
        ss >> w;
        auto entry = std::make_pair(std::make_pair(i, j), w);

        assert(step <= 1);
        switch (step) {
        case 0:
            A.push_back(entry);
            break;
        case 1:
            expected.push_back(entry);
            break;
        default:
            // Should not be reachable
            break;
        }
    }
    std::sort(A.begin(), A.end());
    std::sort(expected.begin(), expected.end());
}

// Compute the tranpose of the matrix.
void transpose_matrix(std::vector<std::pair<std::pair<int, int>, int>> &A,
                      std::vector<std::pair<std::pair<int, int>, int>> &A_T)
{
    for (auto [idx, value] : A) {
        auto [i, j] = idx;
        A_T.push_back(std::make_pair(std::make_pair(j, i), value));
    }
    std::sort(A_T.begin(), A_T.end());
}

void flatten_matrix(std::vector<std::pair<std::pair<int, int>, int>> &matrix,
                    std::vector<int> &idx1, std::vector<int> &idx2,
                    std::vector<int> &values)
{
    for (auto [idx, value] : matrix) {
        idx1.push_back(idx.first);
        idx2.push_back(idx.second);
        values.push_back(value);
    }
}

std::pair<int, int> get_matrix_dimensions(std::vector<std::pair<std::pair<int, int>, int>> &A)
{
    int m = A[0].first.first;
    int n = A[0].first.second;
    for (auto [idx, _] : A) {
        m = std::max(m, idx.first);
        n = std::max(n, idx.second);
    }
    return std::make_pair(m + 1, n + 1);
}

// Gather matrix from across ranks to the root.
void gather_matrix(std::vector<std::pair<std::pair<int, int>, int>> &local_matrix,
                   std::vector<std::pair<std::pair<int, int>, int>> &full_matrix,
                   int root, MPI_Comm comm)
{
    int size, rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    int local_count = static_cast<int>(local_matrix.size());
    if (rank == root) {
        std::vector<int> recvcounts;
        recvcounts.resize(size);
        MPI_Gather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, root, comm);

        std::vector<int> disps;
        disps.push_back(0);
        for (int i = 1; i < size; ++i) {
            disps.push_back(disps[i-1] + recvcounts[i-1]);
        }
        int total_count = disps[size - 1] + recvcounts[size - 1];
        assert(static_cast<int>(disps.size()) == size);
        assert(static_cast<int>(recvcounts.size()) == size);

        std::vector<int> send_idx1;
        std::vector<int> send_idx2;
        std::vector<int> send_values;
        flatten_matrix(local_matrix, send_idx1, send_idx2, send_values);

        std::vector<int> recv_idx1;
        recv_idx1.resize(total_count);
        MPI_Gatherv(send_idx1.data(), send_idx1.size(), MPI_INT,
                    recv_idx1.data(), recvcounts.data(), disps.data(), MPI_INT, 0, comm);
        std::vector<int> recv_idx2;
        recv_idx2.resize(total_count);
        MPI_Gatherv(send_idx2.data(), send_idx2.size(), MPI_INT,
                    recv_idx2.data(), recvcounts.data(), disps.data(), MPI_INT, 0, comm);
        std::vector<int> recv_values;
        recv_values.resize(total_count);
        MPI_Gatherv(send_values.data(), send_values.size(), MPI_INT,
                    recv_values.data(), recvcounts.data(), disps.data(), MPI_INT, 0, comm);

        for (int i = 0; i < total_count; ++i) {
            full_matrix.push_back(std::make_pair(std::make_pair(recv_idx1[i], recv_idx2[i]), recv_values[i]));
        }
    } else {
        MPI_Gather(&local_count, 1, MPI_INT, NULL, 1, MPI_INT, root, comm);

        std::vector<int> send_idx1;
        std::vector<int> send_idx2;
        std::vector<int> send_values;
        flatten_matrix(local_matrix, send_idx1, send_idx2, send_values);

        MPI_Gatherv(send_idx1.data(), send_idx1.size(), MPI_INT,
                    NULL, NULL, NULL, MPI_INT, 0, comm);
        MPI_Gatherv(send_idx2.data(), send_idx2.size(), MPI_INT,
                    NULL, NULL, NULL, MPI_INT, 0, comm);
        MPI_Gatherv(send_values.data(), send_values.size(), MPI_INT,
                    NULL, NULL, NULL, MPI_INT, 0, comm);
    }
}

int correctness_check(std::vector<std::pair<std::pair<int, int>, int>> &computed,
                      std::vector<std::pair<std::pair<int, int>, int>> &expected)
{
    bool ok = true;
    std::sort(expected.begin(), expected.end());
    if (computed.size() != expected.size()) {
        ok = false;
    } else {
        for (int i = 0; i < static_cast<int>(expected.size()); ++i) {
            if (expected[i].first.first != computed[i].first.first
                || expected[i].first.second != computed[i].first.second
                || expected[i].second != computed[i].second) {
                ok = false;
                break;
            }
        }
    }
    if (ok) {
        std::cout << "==> correctness_check=ok\n";
    } else {
        std::cout << "==> correctness_check=error\n";
    }
    return ok ? 0 : 1;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Command-line processing
    if (argc != 3) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " [TEST_TYPE] [TEST_INPUT]\n\n";
            std::cerr << "TEST_TYPE is spgemm\n";
        }
        MPI_Finalize();
        return 1;
    }
    std::string test_type {argv[1]};
    assert(test_type == "spgemm");
    std::string fname {argv[2]};

    std::vector<std::pair<std::pair<int, int>, int>> A_complete;
    std::vector<std::pair<std::pair<int, int>, int>> A_T_complete;
    std::vector<std::pair<std::pair<int, int>, int>> expected_result;
    int m = 0, n = 0;
    int file_error = 0;
    if (rank == 0) {
        std::string file_type;
        read_testfile(fname, file_type, A_complete, expected_result);
        if (file_type != test_type) {
            std::cerr << "File " << fname << " is the wrong file type: got "
                      << file_type << " but expected " << test_type << "\n";
            file_error = 1;
        } else {
            assert(A_complete.size() > 0);
            assert(expected_result.size() > 0);
            auto [tmp_m, tmp_n] = get_matrix_dimensions(A_complete);
            m = tmp_m;
            n = tmp_n;
            // Compute the transpose of A
            transpose_matrix(A_complete, A_T_complete);
        }
    }
    MPI_Bcast(&file_error, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (file_error) {
        MPI_Finalize();
        return 1;
    }
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Comm comm_2d;
    int q = 0;
    // Set up a 2d cartesian topology for the 2d spgemm
    q = sqrt(size);
    assert(size == (q * q));
    int dims[] = {q, q};
    int periods[] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &comm_2d);

    // Set up row and column communicators
    int co[2];
    MPI_Cart_get(comm_2d, 2, dims, periods, co);
    MPI_Comm row_comm;
    MPI_Comm_split(comm_2d, co[0], rank, &row_comm);
    MPI_Comm col_comm;
    MPI_Comm_split(comm_2d, co[1], rank, &col_comm);

    std::vector<std::pair<std::pair<int, int>, int>> A;
    std::vector<std::pair<std::pair<int, int>, int>> A_T;
    distribute_matrix_2d(m, n, A_complete, A, 0, comm_2d);
    if (test_type == "spgemm") {
        distribute_matrix_2d(n, m, A_T_complete, A_T, 0, comm_2d);
    }
    // Maybe we can free up some memory with this
    A_complete.clear();
    A_T_complete.clear();

    std::vector<std::pair<std::pair<int, int>, int>> spgemm_result;
    //std::vector<std::pair<std::pair<int, int>, int>> computed_dist;
    // Run and time the code
    double start = MPI_Wtime();
    spgemm_2d(m, n, m, A, A_T, spgemm_result,
                  [](int a, int b){ return a + b; },
                  [](int a, int b){ return a * b; }, row_comm, col_comm);
    /*
    if (test_type == "spgemm") {
        spgemm_2d(m, n, m, A, A_T, spgemm_result,
                  [](int a, int b){ return a + b; },
                  [](int a, int b){ return a * b; }, row_comm, col_comm);
    } else {
        apsp(std::max(m, n), A, computed_dist, row_comm, col_comm);
    }
        */
    double time = MPI_Wtime() - start;
    double avg_time = 0.0;
    MPI_Reduce(&time, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        avg_time = avg_time / size;
        std::cout << "==> time_taken=" << avg_time << "s\n";
    }

    // Gather matrix from across ranks to root 0
    std::vector<std::pair<std::pair<int, int>, int>> complete_spgemm_result;
    //std::vector<std::pair<std::pair<int, int>, int>> complete_computed_dist;
    gather_matrix(spgemm_result, complete_spgemm_result, 0, MPI_COMM_WORLD);
    /*
    if (test_type == "spgemm") {
        gather_matrix(spgemm_result, complete_spgemm_result, 0, MPI_COMM_WORLD);
    } else {
        gather_matrix(computed_dist, complete_computed_dist, 0, MPI_COMM_WORLD);
    }
        */

    MPI_Barrier(MPI_COMM_WORLD);
    int ret = 0;
    if (rank == 0) {
        std::sort(complete_spgemm_result.begin(), complete_spgemm_result.end());
        ret = correctness_check(complete_spgemm_result, expected_result);
        /*
        if (test_type == "spgemm") {
            std::sort(complete_spgemm_result.begin(), complete_spgemm_result.end());
            ret = correctness_check(complete_spgemm_result, expected_result);
        } else {
            std::sort(complete_computed_dist.begin(), complete_computed_dist.end());
            ret = correctness_check(complete_computed_dist, expected_result);
        }
            */
    }
    MPI_Finalize();
    return ret;
}
