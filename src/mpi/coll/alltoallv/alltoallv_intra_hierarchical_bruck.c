#include "mpiimpl.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

static int myPow(int x, unsigned int p)
{
	if (p == 0)
	{
		return 1;
	}
	if (p == 1)
	{
		return x;
	}

	{
		int tmp = myPow(x, p / 2);
		return (p % 2 == 0) ? tmp * tmp : x * tmp * tmp;
	}
}

static size_t alloc_bytes(MPI_Aint requested)
{
	return requested <= 0 ? 1 : (size_t) requested;
}

//  coalesced
/**
n: number of process per node, 			n \in [1, p]
r: radix of algorithm 					r \in [2, num_proc_pernode]
// bblock								bblock \in [1, num_node-1]
**/

/* parameters removed from the header and need addressing
int n
int r
int bblock
*/
int MPIR_Alltoallv_intra_hierarchical_bruck(
	const void *sendbuf,
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
	char *recvbuf_c = recvbuf;
	const char *sendbuf_c = sendbuf;
	char *temp_send_buffer = NULL;
	char *extra_buffer = NULL;
	char *temp_recv_buffer = NULL;
	MPIR_Request **req = NULL;
	MPI_Status *stat = NULL;
	MPI_Aint *updated_sendcounts = NULL;
	int *rotate_index_array = NULL;
	int *pos_status = NULL;
	int *sent_blocks = NULL;
	MPI_Aint *metadata_send = NULL;
	MPI_Aint *metadata_recv = NULL;
	MPI_Aint *nsend = NULL;
	MPI_Aint *nrecv = NULL;
	MPI_Aint *nsdisp = NULL;
	MPI_Aint *nrdisp = NULL;
	int n = 0, ngroup = 0, gid = 0, grank = 0;
	int r = 0, bblock = 0;
	MPI_Aint sendtype_size = 0, recvtype_size = 0, send_extent = 0, recv_extent = 0;
	MPI_Aint typesize = 0, expected_rdisp = 0;
	int sendtype_is_contig = 0, recvtype_is_contig = 0;
	int sw = 0, imax = 0, max_sd = 0;
	MPI_Aint local_max_count = 0, max_send_count = 0;
	int id = 0;
	int spoint = 1, distance = 1, next_distance = 0, di = 0;
	MPI_Aint index = 0, soffset = 0, roffset = 0;
	int req_cnt = 0, ss = 0;

	MPIR_COMM_RANK_SIZE(comm_ptr, rank, nprocs);

	if (sendbuf == MPI_IN_PLACE)
	{
		//printf("HieAta - Warning: MPI_IN_PLACE is not supported by MPIR_Alltoallv_intra_hierarchical_bruck, falling back to MPIR_Alltoallv_intra_pairwise_sendrecv_replace\n");
		mpi_errno = MPIR_Alltoallv_intra_pairwise_sendrecv_replace(sendbuf, sendcounts,
																   sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm_ptr, coll_attr);
		goto fn_exit;
	}

	if (!MPII_Comm_is_node_canonical(comm_ptr))
	{
		//printf("HieAta - Warning: non-canonical node-aware communicator, falling back to MPIR_Alltoallv_intra_scattered\n");
		mpi_errno = MPIR_Alltoallv_intra_scattered(sendbuf, sendcounts, sdispls, sendtype,
												   recvbuf, recvcounts, rdispls, recvtype, comm_ptr, coll_attr);
		goto fn_exit;
	}

	n = comm_ptr->num_local;
	ngroup = comm_ptr->num_external;
	gid = comm_ptr->external_rank;
	grank = comm_ptr->local_rank;
	r = MPIR_CVAR_ALLTOALLV_HIEATA_RADIX;
	bblock = MPIR_CVAR_ALLTOALLV_HIEATA_BTHSIZE;

	if (n <= 1 || ngroup <= 0 || nprocs % ngroup != 0 || nprocs / ngroup != n || r < 2)
	{
		mpi_errno = MPIR_Alltoallv_intra_scattered(sendbuf, sendcounts, sdispls, sendtype,
												   recvbuf, recvcounts, rdispls, recvtype, comm_ptr, coll_attr);
		goto fn_exit;
	}
	if (r > n)
	{
		r = n;
	}
	if (bblock <= 0 || bblock > ngroup)
	{
		bblock = ngroup;
	}

	MPIR_Datatype_get_size_macro(sendtype, sendtype_size);
	MPIR_Datatype_get_size_macro(recvtype, recvtype_size);
	MPIR_Datatype_get_extent_macro(sendtype, send_extent);
	MPIR_Datatype_get_extent_macro(recvtype, recv_extent);

	MPIR_Datatype_is_contig(sendtype, &sendtype_is_contig);
	MPIR_Datatype_is_contig(recvtype, &recvtype_is_contig);
	if (!sendtype_is_contig || !recvtype_is_contig || sendtype_size != recvtype_size ||
		send_extent != sendtype_size || recv_extent != recvtype_size)
	{
		//printf("HieAra - Warning: non-contiguous or non-matching datatypes, falling back to MPIR_Alltoallv_intra_scattered\n");
		mpi_errno = MPIR_Alltoallv_intra_scattered(sendbuf, sendcounts, sdispls, sendtype,
												   recvbuf, recvcounts, rdispls, recvtype, comm_ptr, coll_attr);
		goto fn_exit;
	}

	typesize = sendtype_size;
	expected_rdisp = 0;
	for (int i = 0; i < nprocs; i++)
	{
		if (rdispls[i] != expected_rdisp)
		{
			mpi_errno = MPIR_Alltoallv_intra_scattered(sendbuf, sendcounts, sdispls,
													   sendtype, recvbuf, recvcounts, rdispls, recvtype, comm_ptr, coll_attr);
			goto fn_exit;
		}
		expected_rdisp += recvcounts[i];
	}

	sw = (int) ceil(log((double) n) / log((double) r));	// required digits for intra-Bruck

	imax = myPow(r, sw - 1) * ngroup;
	max_sd = (ngroup > imax) ? ngroup : imax; // max send data block count

	updated_sendcounts = MPL_malloc(alloc_bytes(nprocs * sizeof(MPI_Aint)), MPL_MEM_COLL);
	rotate_index_array = MPL_malloc(alloc_bytes(nprocs * sizeof(int)), MPL_MEM_COLL);
	pos_status = MPL_malloc(alloc_bytes(nprocs * sizeof(int)), MPL_MEM_COLL);
	sent_blocks = MPL_malloc(alloc_bytes(max_sd * sizeof(int)), MPL_MEM_COLL);
	metadata_send = MPL_malloc(alloc_bytes(max_sd * sizeof(MPI_Aint)), MPL_MEM_COLL);
	metadata_recv = MPL_malloc(alloc_bytes(max_sd * sizeof(MPI_Aint)), MPL_MEM_COLL);
	nsend = MPL_malloc(alloc_bytes(ngroup * sizeof(MPI_Aint)), MPL_MEM_COLL);
	nrecv = MPL_malloc(alloc_bytes(ngroup * sizeof(MPI_Aint)), MPL_MEM_COLL);
	nsdisp = MPL_malloc(alloc_bytes(ngroup * sizeof(MPI_Aint)), MPL_MEM_COLL);
	nrdisp = MPL_malloc(alloc_bytes(ngroup * sizeof(MPI_Aint)), MPL_MEM_COLL);
	MPIR_ERR_CHKANDJUMP(!updated_sendcounts || !rotate_index_array || !pos_status ||
							!sent_blocks || !metadata_send || !metadata_recv || !nsend || !nrecv ||
							!nsdisp || !nrdisp,
						mpi_errno, MPI_ERR_OTHER, "**nomem");
	memcpy(updated_sendcounts, sendcounts, nprocs * sizeof(MPI_Aint));
	memset(pos_status, 0, nprocs * sizeof(int));

	// 1. Find max send elements per data-block
	for (int i = 0; i < nprocs; i++)
	{
		if (sendcounts[i] > local_max_count)
			local_max_count = sendcounts[i];
	}
	mpi_errno = MPIR_Allreduce(&local_max_count, &max_send_count, 1, MPIR_AINT_INTERNAL, MPI_MAX,
							   comm_ptr, coll_attr);
	MPIR_ERR_CHECK(mpi_errno);

	// 2. create local index array after rotation
	for (int i = 0; i < ngroup; i++)
	{
		int gsp = i * n;
		for (int j = 0; j < n; j++)
		{
			rotate_index_array[id++] = gsp + (2 * grank - j + n) % n;
		}
	}

	temp_send_buffer = MPL_malloc(alloc_bytes(max_send_count * typesize * nprocs), MPL_MEM_COLL);
	extra_buffer = MPL_malloc(alloc_bytes(max_send_count * typesize * nprocs), MPL_MEM_COLL);
	temp_recv_buffer = MPL_malloc(alloc_bytes(max_send_count * typesize * max_sd), MPL_MEM_COLL);
	MPIR_ERR_CHKANDJUMP(!temp_send_buffer || !extra_buffer || !temp_recv_buffer,
						mpi_errno, MPI_ERR_OTHER, "**nomem");

	// Intra-Bruck 
	spoint = 1;
	distance = 1;
	next_distance = r;
	di = 0;
	for (int x = 0; x < sw; x++)
	{
		for (int z = 1; z < r; z++)
		{
			di = 0;
			spoint = z * distance;
			if (spoint > n - 1)
			{
				break;
			}

			// get the sent data-blocks
			for (int g = 0; g < ngroup; g++)
			{
				for (int i = spoint; i < n; i += next_distance)
				{
					for (int j = i; j < (i + distance); j++)
					{
						if (j > n - 1)
						{
							break;
						}
						int id = g * n + (j + grank) % n;
						sent_blocks[di++] = id;
					}
				}
			}

			// 2) prepare metadata and send buffer
			MPI_Aint sendCount = 0, offset = 0;
			for (int i = 0; i < di; i++)
			{
				int send_index = rotate_index_array[sent_blocks[i]];
				metadata_send[i] = updated_sendcounts[send_index];

				if (pos_status[send_index] == 0)
					memcpy(&temp_send_buffer[offset], sendbuf_c + sdispls[send_index] * send_extent,
						   updated_sendcounts[send_index] * typesize);
				else
					memcpy(&temp_send_buffer[offset], &extra_buffer[sent_blocks[i] * max_send_count * typesize], updated_sendcounts[send_index] * typesize);
				offset += updated_sendcounts[send_index] * typesize;
			}

			int recv_proc = gid * n + (grank + spoint) % n;		// receive data from rank + 2^step process
			int send_proc = gid * n + (grank - spoint + n) % n; // send data from rank - 2^k process

			// 3) exchange metadata
			mpi_errno = MPIC_Sendrecv(metadata_send, di, MPIR_AINT_INTERNAL, send_proc,
									  MPIR_ALLTOALLV_TAG, metadata_recv, di, MPIR_AINT_INTERNAL,
									  recv_proc, MPIR_ALLTOALLV_TAG, comm_ptr, MPI_STATUS_IGNORE,
									  coll_attr);
			MPIR_ERR_CHECK(mpi_errno);

			for (int i = 0; i < di; i++)
			{
				sendCount += metadata_recv[i];
			}

			// 4) exchange data
			mpi_errno = MPIC_Sendrecv(temp_send_buffer, offset, MPIR_BYTE_INTERNAL, send_proc,
									  MPIR_ALLTOALLV_TAG, temp_recv_buffer, sendCount * typesize,
									  MPIR_BYTE_INTERNAL, recv_proc, MPIR_ALLTOALLV_TAG, comm_ptr,
									  MPI_STATUS_IGNORE, coll_attr);
			MPIR_ERR_CHECK(mpi_errno);

			// 5) replace
			offset = 0;
			for (int i = 0; i < di; i++)
			{
				int send_index = rotate_index_array[sent_blocks[i]];

				memcpy(&extra_buffer[sent_blocks[i] * max_send_count * typesize], &temp_recv_buffer[offset], metadata_recv[i] * typesize);

				offset += metadata_recv[i] * typesize;
				pos_status[send_index] = 1;
				updated_sendcounts[send_index] = metadata_recv[i];
			}
		}
		distance *= r;
		next_distance *= r;
	}

	// organize data
	index = 0;
	for (int i = 0; i < nprocs; i++)
	{
		MPI_Aint d = updated_sendcounts[rotate_index_array[i]] * typesize;
		if (grank == (i % n))
		{
			memcpy(&temp_send_buffer[index], sendbuf_c + sdispls[i] * send_extent, d);
		}
		else
		{
			memcpy(&temp_send_buffer[index], &extra_buffer[i * max_send_count * typesize], d);
		}
		index += d;
	}

	soffset = 0;
	roffset = 0;
	for (int i = 0; i < ngroup; i++)
	{
		nsend[i] = 0, nrecv[i] = 0;
		for (int j = 0; j < n; j++)
		{
			int id = i * n + j;
			MPI_Aint sn = updated_sendcounts[rotate_index_array[id]];
			nsend[i] += sn;
			nrecv[i] += recvcounts[id];
		}
		nsdisp[i] = soffset, nrdisp[i] = roffset;
		soffset += nsend[i] * typesize, roffset += nrecv[i] * typesize;
	}

	req = MPL_malloc(2 * bblock * sizeof(MPIR_Request *), MPL_MEM_COLL);
	stat = MPL_malloc(2 * bblock * sizeof(MPI_Status), MPL_MEM_COLL);
	MPIR_ERR_CHKANDJUMP(!req || !stat, mpi_errno, MPI_ERR_OTHER, "**nomem");
	req_cnt = 0;
	ss = 0;

	for (int ii = 0; ii < ngroup; ii += bblock)
	{
		req_cnt = 0;
		ss = ngroup - ii < bblock ? ngroup - ii : bblock;

		for (int i = 0; i < ss; i++)
		{
			int nsrc = (gid + i + ii) % ngroup;
			int src = nsrc * n + grank; // avoid always to reach first master node

			mpi_errno = MPIC_Irecv(recvbuf_c + nrdisp[nsrc], nrecv[nsrc] * typesize,
								   MPIR_BYTE_INTERNAL, src, MPIR_ALLTOALLV_TAG, comm_ptr, &req[req_cnt++]);
			MPIR_ERR_CHECK(mpi_errno);
		}

		for (int i = 0; i < ss; i++)
		{
			int ndst = (gid - i - ii + ngroup) % ngroup;
			int dst = ndst * n + grank;

			mpi_errno = MPIC_Isend(&temp_send_buffer[nsdisp[ndst]], nsend[ndst] * typesize,
								   MPIR_BYTE_INTERNAL, dst, MPIR_ALLTOALLV_TAG, comm_ptr, &req[req_cnt++],
								   coll_attr);
			MPIR_ERR_CHECK(mpi_errno);
		}

		mpi_errno = MPIC_Waitall(req_cnt, req, stat);
		MPIR_ERR_CHECK(mpi_errno);

		if (mpi_errno == MPI_ERR_IN_STATUS)
		{
			for (int i = 0; i < req_cnt; i++)
			{
				if (stat[i].MPI_ERROR != MPI_SUCCESS)
				{
					mpi_errno = stat[i].MPI_ERROR;
					MPIR_ERR_CHECK(mpi_errno);
				}
			}
		}
	}

fn_exit:
	MPL_free(nrdisp);
	MPL_free(nsdisp);
	MPL_free(nrecv);
	MPL_free(nsend);
	MPL_free(metadata_recv);
	MPL_free(metadata_send);
	MPL_free(sent_blocks);
	MPL_free(pos_status);
	MPL_free(rotate_index_array);
	MPL_free(updated_sendcounts);
	MPL_free(req);
	MPL_free(stat);
	MPL_free(temp_recv_buffer);
	MPL_free(extra_buffer);
	MPL_free(temp_send_buffer);
	return mpi_errno;
fn_fail:
	goto fn_exit;
}
