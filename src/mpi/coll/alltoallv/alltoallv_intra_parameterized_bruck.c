/*
 * Self-contained ParAta prototype extracted from twophase_tunable_rbruckv.cpp.
 */

#include "mpiimpl.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

static int my_pow(int x, unsigned int p)
{
    if (p == 0) {
        return 1;
    }
    if (p == 1) {
        return x;
    }

    {
        int tmp = my_pow(x, p / 2);
        return (p % 2 == 0) ? tmp * tmp : x * tmp * tmp;
    }
}

static size_t alloc_bytes(MPI_Aint requested)
{
    return requested <= 0 ? 1 : (size_t) requested;
}

int MPIR_Alltoallv_intra_parameterized_bruck(const void *sendbuf,
                                             const MPI_Aint *sendcounts,
                                             const MPI_Aint *sdispls,
                                             MPI_Datatype sendtype,
                                             void *recvbuf,
                                             const MPI_Aint *recvcounts,
                                             const MPI_Aint *rdispls,
                                             MPI_Datatype recvtype,
                                             MPIR_Comm *comm_ptr,
                                             int coll_attr)
{
    int mpi_errno = MPI_SUCCESS;
    int rank, nprocs;
    int r = MPIR_CVAR_ALLTOALLV_PARATA_RADIX;
    char *recvbuf_c = recvbuf;
    const char *sendbuf_c = sendbuf;
    MPI_Aint sendtype_size = 0, recvtype_size = 0, send_extent = 0, recv_extent = 0;
    MPI_Aint typesize = 0;
    int sendtype_is_contig = 0, recvtype_is_contig = 0;
    MPI_Aint local_max_count = 0, max_send_count = 0;
    MPI_Aint *send_n_copy = NULL;
    int *rotate_index_array = NULL;
    int *pos_status = NULL;
    int *sent_blocks = NULL;
    MPI_Aint *metadata_send = NULL;
    MPI_Aint *metadata_recv = NULL;
    char *extra_buffer = NULL;
    char *temp_send_buffer = NULL;
    char *temp_recv_buffer = NULL;
    int w = 0, nlpow = 0, d = 0;
    int distance = 0, next_distance = 0;

    MPIR_COMM_RANK_SIZE(comm_ptr, rank, nprocs);

    if (sendbuf == MPI_IN_PLACE) {
        //printf("ParAta - Warning: MPI_IN_PLACE is not supported by MPIR_Alltoallv_intra_parameterized_bruck, falling back to MPIR_Alltoallv_intra_pairwise_sendrecv_replace\n");
        mpi_errno = MPIR_Alltoallv_intra_pairwise_sendrecv_replace(sendbuf, sendcounts,
                                                                   sdispls, sendtype, recvbuf,
                                                                   recvcounts, rdispls, recvtype,
                                                                   comm_ptr, coll_attr);
        goto fn_exit;
    }

    MPIR_Datatype_get_size_macro(sendtype, sendtype_size);
    MPIR_Datatype_get_size_macro(recvtype, recvtype_size);
    MPIR_Datatype_get_extent_macro(sendtype, send_extent);
    MPIR_Datatype_get_extent_macro(recvtype, recv_extent);
    MPIR_Datatype_is_contig(sendtype, &sendtype_is_contig);
    MPIR_Datatype_is_contig(recvtype, &recvtype_is_contig);

    if (!sendtype_is_contig || !recvtype_is_contig || sendtype_size != recvtype_size ||
        send_extent != sendtype_size || recv_extent != recvtype_size) {
        //printf("ParAta - Warning: non-contiguous or non-matching datatypes, falling back to MPIR_Alltoallv_intra_scattered\n");
        mpi_errno = MPIR_Alltoallv_intra_scattered(sendbuf, sendcounts, sdispls, sendtype,
                                                   recvbuf, recvcounts, rdispls, recvtype,
                                                   comm_ptr, coll_attr);
        goto fn_exit;
    }

    typesize = sendtype_size;

    if (nprocs <= 1) {
        memcpy(recvbuf_c + rdispls[0] * recv_extent, sendbuf_c + sdispls[0] * send_extent,
               recvcounts[0] * typesize);
        goto fn_exit;
    }

    if (r < 2) {
        mpi_errno = MPIR_Alltoallv_intra_scattered(sendbuf, sendcounts, sdispls, sendtype,
                                                   recvbuf, recvcounts, rdispls, recvtype,
                                                   comm_ptr, coll_attr);
        goto fn_exit;
    }

    if (r > nprocs) {
        r = nprocs;
    }

    w = (int) ceil(log((double) nprocs) / log((double) r));
    nlpow = my_pow(r, w - 1);
    d = (my_pow(r, w) - nprocs) / nlpow;

    for (int i = 0; i < nprocs; i++) {
        if (sendcounts[i] > local_max_count) {
            local_max_count = sendcounts[i];
        }
    }

    mpi_errno = MPIR_Allreduce(&local_max_count, &max_send_count, 1, MPIR_AINT_INTERNAL, MPI_MAX,
                               comm_ptr, coll_attr);
    MPIR_ERR_CHECK(mpi_errno);

    send_n_copy = MPL_malloc(alloc_bytes(nprocs * sizeof(MPI_Aint)), MPL_MEM_COLL);
    rotate_index_array = MPL_malloc(alloc_bytes(nprocs * sizeof(int)), MPL_MEM_COLL);
    pos_status = MPL_malloc(alloc_bytes(nprocs * sizeof(int)), MPL_MEM_COLL);
    sent_blocks = MPL_malloc(alloc_bytes(nlpow * sizeof(int)), MPL_MEM_COLL);
    metadata_send = MPL_malloc(alloc_bytes(nlpow * sizeof(MPI_Aint)), MPL_MEM_COLL);
    metadata_recv = MPL_malloc(alloc_bytes(nlpow * sizeof(MPI_Aint)), MPL_MEM_COLL);
    extra_buffer = MPL_malloc(alloc_bytes(max_send_count * typesize * nprocs), MPL_MEM_COLL);
    temp_send_buffer = MPL_malloc(alloc_bytes(max_send_count * typesize * nlpow), MPL_MEM_COLL);
    temp_recv_buffer = MPL_malloc(alloc_bytes(max_send_count * typesize * nlpow), MPL_MEM_COLL);
    MPIR_ERR_CHKANDJUMP(!send_n_copy || !rotate_index_array || !pos_status || !sent_blocks ||
                        !metadata_send || !metadata_recv || !extra_buffer || !temp_send_buffer ||
                        !temp_recv_buffer, mpi_errno, MPI_ERR_OTHER, "**nomem");

    memcpy(send_n_copy, sendcounts, nprocs * sizeof(MPI_Aint));
    memset(pos_status, 0, nprocs * sizeof(int));

    for (int i = 0; i < nprocs; i++) {
        rotate_index_array[i] = (2 * rank - i + nprocs) % nprocs;
    }

    memcpy(recvbuf_c + rdispls[rank] * recv_extent, sendbuf_c + sdispls[rank] * send_extent,
           recvcounts[rank] * typesize);

    distance = my_pow(r, w - 1);
    next_distance = distance * r;

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

            MPI_Aint send_count = 0;
            MPI_Aint offset = 0;

            for (int i = 0; i < di; i++) {
                int send_index = rotate_index_array[sent_blocks[i]];
                MPI_Aint bytes = send_n_copy[send_index] * typesize;

                metadata_send[i] = send_n_copy[send_index];
                if (pos_status[send_index] == 0) {
                    memcpy(temp_send_buffer + offset,
                           sendbuf_c + sdispls[send_index] * send_extent, bytes);
                } else {
                    memcpy(temp_send_buffer + offset,
                           extra_buffer + sent_blocks[i] * max_send_count * typesize,
                           bytes);
                }
                offset += bytes;
            }

            int recvrank = (rank + spoint) % nprocs;
            int sendrank = (rank - spoint + nprocs) % nprocs;

            mpi_errno = MPIC_Sendrecv(metadata_send, di, MPIR_AINT_INTERNAL, sendrank,
                                      MPIR_ALLTOALLV_TAG, metadata_recv, di, MPIR_AINT_INTERNAL,
                                      recvrank, MPIR_ALLTOALLV_TAG, comm_ptr, MPI_STATUS_IGNORE,
                                      coll_attr);
            MPIR_ERR_CHECK(mpi_errno);

            for (int i = 0; i < di; i++) {
                send_count += metadata_recv[i];
            }

            mpi_errno = MPIC_Sendrecv(temp_send_buffer, offset, MPIR_BYTE_INTERNAL, sendrank,
                                      MPIR_ALLTOALLV_TAG, temp_recv_buffer, send_count * typesize,
                                      MPIR_BYTE_INTERNAL, recvrank, MPIR_ALLTOALLV_TAG, comm_ptr,
                                      MPI_STATUS_IGNORE, coll_attr);
            MPIR_ERR_CHECK(mpi_errno);

            offset = 0;
            for (int i = 0; i < di; i++) {
                int send_index = rotate_index_array[sent_blocks[i]];
                MPI_Aint bytes = metadata_recv[i] * typesize;
                int origin_index = (sent_blocks[i] - rank + nprocs) % nprocs;

                if (origin_index % next_distance == (recvrank - rank + nprocs) % nprocs) {
                    memcpy(recvbuf_c + rdispls[sent_blocks[i]] * recv_extent,
                           temp_recv_buffer + offset, bytes);
                } else {
                    memcpy(extra_buffer + sent_blocks[i] * max_send_count * typesize,
                           temp_recv_buffer + offset, bytes);
                }

                offset += bytes;
                pos_status[send_index] = 1;
                send_n_copy[send_index] = metadata_recv[i];
            }
        }

        distance /= r;
        next_distance /= r;
    }

  fn_exit:
    MPL_free(temp_recv_buffer);
    MPL_free(temp_send_buffer);
    MPL_free(extra_buffer);
    MPL_free(metadata_recv);
    MPL_free(metadata_send);
    MPL_free(sent_blocks);
    MPL_free(pos_status);
    MPL_free(rotate_index_array);
    MPL_free(send_n_copy);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
