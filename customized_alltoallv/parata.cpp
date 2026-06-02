/*
 * Self-contained ParAta prototype extracted from twophase_tunable_rbruckv.cpp.
 */

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace {

int my_pow(int x, unsigned int p)
{
    if (p == 0) {
        return 1;
    }
    if (p == 1) {
        return x;
    }

    int tmp = my_pow(x, p / 2);
    return (p % 2 == 0) ? tmp * tmp : x * tmp * tmp;
}

size_t alloc_bytes(size_t requested)
{
    return requested == 0 ? 1 : requested;
}

} /* namespace */

int twophase_rbruck_alltoallv(int r, char *sendbuf, int *sendcounts,
                              int *sdispls, MPI_Datatype sendtype, char *recvbuf,
                              int *recvcounts, int *rdispls, MPI_Datatype recvtype,
                              MPI_Comm comm)
{
    (void) recvtype;

    int rank, nprocs, typesize;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
    MPI_Type_size(sendtype, &typesize);

    if (nprocs <= 1) {
        std::memcpy(recvbuf + rdispls[0] * typesize, sendbuf + sdispls[0] * typesize,
                    recvcounts[0] * typesize);
        return MPI_SUCCESS;
    }

    if (r < 2) {
        return -1;
    }

    if (r > nprocs) {
        r = nprocs;
    }

    int w = static_cast<int>(std::ceil(std::log(static_cast<double>(nprocs)) /
                                       std::log(static_cast<double>(r))));
    int nlpow = my_pow(r, w - 1);
    int d = (my_pow(r, w) - nprocs) / nlpow;

    int local_max_count = 0;
    for (int i = 0; i < nprocs; i++) {
        local_max_count = std::max(local_max_count, sendcounts[i]);
    }

    int max_send_count = 0;
    MPI_Allreduce(&local_max_count, &max_send_count, 1, MPI_INT, MPI_MAX, comm);

    std::vector<int> send_n_copy(sendcounts, sendcounts + nprocs);
    std::vector<int> rotate_index_array(nprocs);
    std::vector<int> pos_status(nprocs, 0);
    std::vector<int> sent_blocks(nlpow);

    for (int i = 0; i < nprocs; i++) {
        rotate_index_array[i] = (2 * rank - i + nprocs) % nprocs;
    }

    size_t extra_bytes = alloc_bytes(static_cast<size_t>(max_send_count) * typesize * nprocs);
    size_t temp_bytes = alloc_bytes(static_cast<size_t>(max_send_count) * typesize * nlpow);
    std::vector<char> extra_buffer(extra_bytes);
    std::vector<char> temp_send_buffer(temp_bytes);
    std::vector<char> temp_recv_buffer(temp_bytes);

    std::memcpy(recvbuf + rdispls[rank] * typesize, sendbuf + sdispls[rank] * typesize,
                recvcounts[rank] * typesize);

    int distance = my_pow(r, w - 1);
    int next_distance = distance * r;

    for (int x = w - 1; x >= 0; x--) {
        int ze = (x == w - 1) ? (r - d) : r;

        for (int z = ze - 1; z > 0; z--) {
            int di = 0;
            int spoint = z * distance;

            for (int i = spoint; i < nprocs; i += next_distance) {
                for (int j = i; j < i + distance; j++) {
                    if (j > nprocs - 1) {
                        break;
                    }
                    sent_blocks[di++] = (j + rank) % nprocs;
                }
            }

            std::vector<int> metadata_send(di);
            int send_count = 0;
            int offset = 0;

            for (int i = 0; i < di; i++) {
                int send_index = rotate_index_array[sent_blocks[i]];
                int bytes = send_n_copy[send_index] * typesize;

                metadata_send[i] = send_n_copy[send_index];
                if (pos_status[send_index] == 0) {
                    std::memcpy(temp_send_buffer.data() + offset,
                                sendbuf + sdispls[send_index] * typesize, bytes);
                } else {
                    std::memcpy(temp_send_buffer.data() + offset,
                                extra_buffer.data() +
                                    static_cast<size_t>(sent_blocks[i]) * max_send_count * typesize,
                                bytes);
                }
                offset += bytes;
            }

            int recvrank = (rank + spoint) % nprocs;
            int sendrank = (rank - spoint + nprocs) % nprocs;

            std::vector<int> metadata_recv(di);
            MPI_Sendrecv(metadata_send.data(), di, MPI_INT, sendrank, 0,
                         metadata_recv.data(), di, MPI_INT, recvrank, 0, comm,
                         MPI_STATUS_IGNORE);

            for (int i = 0; i < di; i++) {
                send_count += metadata_recv[i];
            }

            MPI_Sendrecv(temp_send_buffer.data(), offset, MPI_CHAR, sendrank, 1,
                         temp_recv_buffer.data(), send_count * typesize, MPI_CHAR,
                         recvrank, 1, comm, MPI_STATUS_IGNORE);

            offset = 0;
            for (int i = 0; i < di; i++) {
                int send_index = rotate_index_array[sent_blocks[i]];
                int bytes = metadata_recv[i] * typesize;
                int origin_index = (sent_blocks[i] - rank + nprocs) % nprocs;

                if (origin_index % next_distance == (recvrank - rank + nprocs) % nprocs) {
                    std::memcpy(recvbuf + rdispls[sent_blocks[i]] * typesize,
                                temp_recv_buffer.data() + offset, bytes);
                } else {
                    std::memcpy(extra_buffer.data() +
                                    static_cast<size_t>(sent_blocks[i]) * max_send_count * typesize,
                                temp_recv_buffer.data() + offset, bytes);
                }

                offset += bytes;
                pos_status[send_index] = 1;
                send_n_copy[send_index] = metadata_recv[i];
            }
        }

        distance /= r;
        next_distance /= r;
    }

    return MPI_SUCCESS;
}
