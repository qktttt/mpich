/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

/* -- THIS FILE IS AUTO-GENERATED -- */

#ifndef COLL_AUTOGEN_H_INCLUDED
#define COLL_AUTOGEN_H_INCLUDED


/* conditional CSEL conditions */

MPL_STATIC_INLINE_PREFIX bool MPIR_CSEL_check_MPIDI_CH4_release_gather(MPIR_Csel_coll_sig_s * coll_sig)
{
#if defined(MPIDI_CH4_SHM_POSIX)
    return MPIDI_POSIX_check_release_gather(coll_sig);
#else
    return false;
#endif
}
#endif /* COLL_AUTOGEN_H_INCLUDED */
