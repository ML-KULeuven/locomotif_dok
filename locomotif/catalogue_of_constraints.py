# catalogue of constraints that can be used in LoCoMotif-DoK 
# 
# Note that any constraints of the following types are supported as numba-compiled functions: 
# h_mot_repr, h_mot, h_mots_same, h_mset_all, h_mots_diff_all, h_msets_pairwise_all, desir_all.

import numpy as np

from numba import njit
from numba import int32, float64, float32, boolean
from numba.core.types import FunctionType
import inspect
from functools import partial

import warnings
from numba.core.errors import NumbaExperimentalFeatureWarning


# Suppress the NumbaExperimentalFeatureWarning
warnings.simplefilter('ignore', NumbaExperimentalFeatureWarning)


#######################################################################################
# Constraints for motifs 
#######################################################################################

# Combine constraints with an 'and' operator:
def combine_h_mot(*constraints):
    @njit(boolean(int32, int32))
    def combined_constraint(b, e):
        for constraint in constraints:
            if not constraint(b, e):
                return False
        return True
    return combined_constraint

# Commbine constraints with an 'or' operator: 
def combine_h_mot_or(*constraints):
    @njit(boolean(int32, int32))
    def combined_constraint(b, e):
        for constraint in constraints:
            if constraint(b, e):
                return True
        return False
    return combined_constraint

# Start mask:
def h_mot_start_mask(start_mask):
    @njit(boolean(int32, int32))
    def h_mot_start_mask_jit(b, e):
        if start_mask[b]:
            return True
        else:
            return False
    return h_mot_start_mask_jit

# End mask:
def h_mot_end_mask(end_mask):
    @njit(boolean(int32, int32))
    def h_mot_end_mask_jit(b, e):
        if end_mask[e-1]:
            return True
        else:
            return False
    return h_mot_end_mask_jit

# Start and end mask:
def h_mot_start_end_mask(start_mask, end_mask):
    @njit(boolean(int32, int32))
    def h_mot_start_end_mask_jit(b, e):
        if start_mask[b] and end_mask[e-1]:
            return True
        else:
            return False
    return h_mot_start_end_mask_jit

# Length range:
def h_mot_l_min_l_max(l_min, l_max):
    assert 1 <= l_min <= l_max, "Invalid length range (l_min, l_max)."
    @njit(boolean(int32, int32))
    def h_mot_l_min_l_max_jit(b, e):
        if e - b >= l_min and e - b <= l_max:
            return True
        else:
            return False
    return h_mot_l_min_l_max_jit

# Standard deviation: 
def h_mot_std_dev_min(std_dev_min, series):
    series = np.array(series).copy()
    assert std_dev_min >= 0, "Invalid minimum standard deviation."
    
    if np.ndim(series) == 1:
        @njit(boolean(int32, int32))
        def h_mot_std_dev_jit(b, e):
            if np.std(series[b:e]) >= std_dev_min:
                return True
            else:
                return False
    
    else:
        @njit(boolean(int32, int32))
        def h_mot_std_dev_jit(b, e): # allow if any of the dimensions has a standard deviation above the threshold
            n_dim = series.shape[1]
            for dim in range(n_dim):
                if np.std(series[b:e, dim]) >= std_dev_min:
                    return True
            return False
        
        # @njit(boolean(int32, int32))
        # def h_mot_std_dev_jit(b, e): # allow if the average of the standard deviations of the dimensions is above the threshold
        #     n_dim = series.shape[1]
        #     stds = np.zeros(n_dim)
        #     for dim in range(n_dim):
        #         stds[dim] = np.std(series[b:e, dim])
        #     if np.mean(stds) >= std_dev_min:
        #         return True
        #     return False
    
    
    return h_mot_std_dev_jit

# Barriers:
def h_mot_barriers(barriers):
    @njit(boolean(int32, int32))
    def h_mot_barriers_jit(b, e):
        for barrier in barriers:
            if b <= barrier < e:
                return False
        return True
    return h_mot_barriers_jit


#######################################################################################
# Constraints for pairs of motifs (within and across motif sets)
#######################################################################################

# Combine constraints with an 'and' operator:
def combine_h_mots(*constraints):
    @njit(boolean(int32, int32, int32, int32))
    def combined_constraint(b1, e1, b2, e2):
        for constraint in constraints:
            if not constraint(b1, e1, b2, e2):
                return False
        return True
    return combined_constraint

# Combine constraints with an 'or' operator:
def combine_h_mots_or(*constraints):
    @njit(boolean(int32, int32, int32, int32))
    def combined_constraint(b1, e1, b2, e2):
        for constraint in constraints:
            if constraint(b1, e1, b2, e2):
                return True
        return False
    return combined_constraint

# Overlap between motifs (check if the first motif is beta-coincident to the second) (which causes overlapping motifs in a motif set sto be discarded):
def h_mots_overlap(beta): # beta=0 disallows overlaps at all, beta=1 does not check for any overlaps
    assert 0 <= beta <= 1, "The beta (overlap) parameter needs to be between 0 and 1."
    @njit(boolean(int32, int32, int32, int32))
    def h_mots_overlap_jit(b1, e1, b2, e2): 
        # Check if (b1, e1) is beta-coincident to (b2, e2):
        overlap_length = max(0, min(e1, e2) - max(b1, b2))
        interval_length = e2 - b2
        return overlap_length <= beta * interval_length # False when there is too much overlap
    return h_mots_overlap_jit

# A motif cannot start within a buffer_length of the end of an existing motif, and cannot start within an existing motif:
def h_mots_cannot_follow(buffer_length):
    assert 0 <= buffer_length, "The buffer length needs to be a positive integer."
    @njit(boolean(int32, int32, int32, int32))
    def h_mots_cannot_follow_jit(b1, e1, b2, e2): 
        return not (b1 <= b2 <= e1 + buffer_length)
    return h_mots_cannot_follow_jit

# Prevent consecutive motifs by applying h_mots_cannot_follow in both directions: 
def h_mots_non_consecutive(buffer_length): # prevent consecutive motifs: a motif cannot start during an existing motif or within a buffer_length of the end of an existing motif
    assert 0 <= buffer_length, "The buffer length needs to be a positive integer."
    @njit(boolean(int32, int32, int32, int32))
    def h_mots_non_consecutive_jit(b1, e1, b2, e2): 
        return not ( (b1 <= b2 <= e1 + buffer_length) or (b2 <= b1 <= e2 + buffer_length) )
    return h_mots_non_consecutive_jit


#######################################################################################
# Constraints for motif sets
#######################################################################################

# Combine constraints with an 'and' operator:
def combine_h_mset(*constraints):
    @njit(boolean(int32[:], int32[:]))
    def combined_constraint(bs, es):
        for constraint in constraints:
            if not constraint(bs, es):
                return False
        return True
    return combined_constraint

# Combine constraints with an 'or' operator:
def combine_h_mset_or(*constraints):
    @njit(boolean(int32[:], int32[:]))
    def combined_constraint(bs, es):
        for constraint in constraints:
            if constraint(bs, es):
                return True
        return False
    return combined_constraint

# Min and max cardinality:
def h_mset_cardinality_min_max(k_min, k_max):
    assert 1 <= k_min <= k_max, "Invalid cardinality range (k_min, k_max)."
    @njit(boolean(int32[:], int32[:]))
    def h_mset_cardinality_min_max_jit(bs, es):
        if len(bs) >= k_min and len(bs) <= k_max:
            return True
        else:
            return False
    return h_mset_cardinality_min_max_jit

# Min and max time coverage: 
def h_mset_time_coverage_min_max(cov_min, cov_max, n):
    assert 0 <= cov_min <= cov_max <= n, "Invalid time coverage range (cov_min, cov_max)."
    @njit(boolean(int32[:], int32[:]))
    def h_mset_time_coverage_min_max_jit(bs, es):
        covered = np.full(n, False) 
        for b_, e_ in zip(bs, es):
            covered[b_:e_] = True
        if np.sum(covered) >= cov_min and np.sum(covered) <= cov_max:
            return True
        else:
            return False
    return h_mset_time_coverage_min_max_jit

# Overlap between motifs in a motif set (which causes the entire motif set to be discarded in case of any excessive overlap):
def h_mset_overlap(beta): # beta=0 disallows overlaps at all, beta=1 does not check for any overlaps
    assert 0 <= beta <= 1, "The beta (overlap) parameter needs to be between 0 and 1."
    @njit(boolean(int32[:], int32[:]))
    def h_mset_overlap_jit(bs, es):
        for i in range(len(bs)):
            for j in range(i+1, len(bs)):
                b1, e1, b2, e2 = bs[i], es[i], bs[j], es[j]
                # Check if (b1, e1) is beta-coincident to (b2, e2):
                overlap_length = max(0, min(e1, e2) - max(b1, b2))
                interval_length = e2 - b2
                if overlap_length > beta * interval_length: # False when there is too much overlap
                    return False
        return True
    return


#######################################################################################
# Constraints for pairs of motif sets
#######################################################################################

# Combine constraints with an 'and' operator:
def combine_h_msets(*constraints):
    @njit(boolean(int32[:], int32[:], int32[:], int32[:]))
    def combined_constraint(bs1, es1, bs2, es2):
        for constraint in constraints:
            if not constraint(bs1, es1, bs2, es2):
                return False
        return True
    return combined_constraint

# Combine constraints with an 'or' operator:
def combine_h_msets_or(*constraints):
    @njit(boolean(int32[:], int32[:], int32[:], int32[:]))
    def combined_constraint(bs1, es1, bs2, es2):
        for constraint in constraints:
            if constraint(bs1, es1, bs2, es2):
                return True
        return False
    return combined_constraint

# Overlaps between motifs in two motif sets (which causes one of the motif sets to be discarded in case of any excessive overlap):
def h_msets_overlap(beta): # beta=0 disallows overlaps at all, beta=1 does not check for any overlaps
    assert 0 <= beta <= 1, "The beta (overlap) parameter needs to be between 0 and 1."
    @njit(boolean(int32[:], int32[:], int32[:], int32[:]))
    def h_msets_overlap_jit(bs1, es1, bs2, es2):
        for i in range(len(bs1)):
            for j in range(len(bs2)):
                b1, e1, b2, e2 = bs1[i], es1[i], bs2[j], es2[j]
                # Check if (b1, e1) is beta-coincident to (b2, e2):
                overlap_length = max(0, min(e1, e2) - max(b1, b2))
                interval_length = e2 - b2
                if overlap_length > beta * interval_length: # False when there is too much overlap
                    return False
        return True
    return h_msets_overlap_jit

# Cardinalities of motif sets cannot differ more than k_diff:
def h_msets_cardinality_diff(k_diff):
    assert k_diff >= 0, "The k_diff parameter needs to be non-negative."
    @njit(boolean(int32[:], int32[:], int32[:], int32[:]))
    def h_msets_cardinality_diff_jit(bs1, es1, bs2, es2):
        if abs(len(bs1) - len(bs2)) > k_diff:
            return False
        return True
    return h_msets_cardinality_diff_jit


#######################################################################################
# Desirability functions
#######################################################################################

# Combine desirability functions by multiplication:
def combine_desir(*desirabilities):
    @njit(float64(int32[:], int32[:]))
    def combined_constraint(bs, es):
        d = 1.0
        for desirability in desirabilities:
            d *= desirability(bs, es)
        return d
    return combined_constraint

# Minimum cardinality: 
def desir_cardinality_min(k_min):
    assert k_min >= 2, "Invalid minimum cardinality."
    @njit(float64(int32[:], int32[:]))
    def desir_cardinality_min_jit(bs, es):
        if len(bs) < k_min:
            return len(bs) / k_min
        return 1.0
    return desir_cardinality_min_jit

# Maximum cardinality (with a multiplicative penalty of varrho for every excess motif):
def desir_cardinality_max(k_max, varrho):
    assert k_max >= 2, "Invalid maximum cardinality."
    assert 0 <= varrho <= 1, "Invalid varrho."
    @njit(float64(int32[:], int32[:]))
    def desir_cardinality_max_jit(bs, es):
        if len(bs) > k_max:
            return varrho**(len(bs) - k_max)
        return 1.0
    return desir_cardinality_max_jit

# Minimum time coverage:
def desir_time_coverage_min(cov_min, n):
    assert 0 <= cov_min <= n, "Invalid minimum time coverage."
    @njit(float64(int32[:], int32[:]))
    def desir_time_coverage_min_jit(bs, es):
        covered = np.full(n, False)
        for b_, e_ in zip(bs, es):
            covered[b_:e_] = True
        if np.sum(covered) < cov_min:
            return np.sum(covered) / cov_min
        return 1.0
    return desir_time_coverage_min_jit

# Maximum time coverage (with a multiplicative penalty of varrho for every excess time point):
def desir_time_coverage_max(cov_max, varrho, n):
    assert 0 <= cov_max <= n, "Invalid maximum time coverage."
    assert 0 <= varrho <= 1, "Invalid varrho."
    @njit(float64(int32[:], int32[:]))
    def desir_time_coverage_max_jit(bs, es):
        covered = np.full(n, False)
        for b_, e_ in zip(bs, es):
            covered[b_:e_] = True
        if np.sum(covered) > cov_max:
            return varrho**(np.sum(covered) - cov_max)
        return 1.0
    return desir_time_coverage_max_jit

# Minimum motif length:
def desir_motif_length_min(l_min):
    assert l_min >= 1, "Invalid minimum motif length."
    @njit(float64(int32[:], int32[:]))
    def desir_motif_length_min_jit(bs, es):
        lengths = es - bs
        d = 1.0
        for length in lengths:
            if length < l_min:
                d *= length / l_min
                if d == 0:
                    return 0.0
        return d
    return desir_motif_length_min_jit

# Maximum motif length (with a multiplicative penalty of varrho for every excess time point):
def desir_motif_length_max(l_max, varrho):
    assert l_max >= 1, "Invalid maximum motif length."
    assert 0 <= varrho <= 1, "Invalid varrho."
    @njit(float64(int32[:], int32[:]))
    def desir_motif_length_max_jit(bs, es):
        lengths = es - bs
        d = 1.0
        for length in lengths:
            if length > l_max:
                d *= varrho**(length/l_max - 1)
                if d == 0:
                    return 0.0
        return d
    return desir_motif_length_max_jit

# Minimum standard deviation:
def desir_std_dev_min(std_dev_min, series):
    series = np.array(series).copy()
    assert std_dev_min >= 0, "Invalid minimum standard deviation."
    if np.ndim(series) == 1:
        @njit(float64(int32[:], int32[:]))
        def desir_std_dev_min_jit(bs, es):
            d = 1.0
            for b, e in zip(bs, es):
                if np.std(series[b:e]) < std_dev_min:
                    d *= np.std(series[b:e]) / std_dev_min
                    if d == 0:
                        return 0.0
            return d
    else:
        @njit(float64(int32[:], int32[:]))
        def desir_std_dev_min_jit(bs, es): # penalize when all dimensions have a std below the threshold
            d = 1.0
            for b, e in zip(bs, es):
                n_dim = series.shape[1]
                stds = np.zeros(n_dim)
                for dim in range(n_dim):
                    stds[dim] = np.std(series[b:e, dim])
                if np.all(stds < std_dev_min):
                    d *= np.std(series[b:e, dim]) / std_dev_min
                    if d == 0:
                        return 0.0
            return d
    return desir_std_dev_min_jit

# Soft start mask: 
def desir_start_mask(start_mask):
    @njit(float64(int32[:], int32[:]))
    def desir_start_mask_jit(bs, es):
        d = 1.0
        for b in bs:
            d *= start_mask[b]
            if d == 0:
                return 0.0
        return d
    return desir_start_mask_jit

# Soft end mask:
def desir_end_mask(end_mask):
    @njit(float64(int32[:], int32[:]))
    def desir_end_mask_jit(bs, es):
        d = 1.0
        for e in es:
            d *= end_mask[e-1]
            if d == 0:
                return 0.0
        return d
    return desir_end_mask_jit

# Soft average mask (averaged over all the time indices covered by motifs):
def desir_avg_mask(avg_mask):
    @njit(float64(int32[:], int32[:]))
    def desir_avg_mask_jit(bs, es):
        total = 0.0
        count = 0
        for b, e in zip(bs, es):
            total += np.sum(avg_mask[b:e])
            count += e - b
        avg = total / count if count > 0 else 0.0
        return avg
    return desir_avg_mask_jit

# Annotation vector (or guidance) applied on the beginning point of the representative motif:
def desir_annotation_vector(annotation_vector):
    @njit(float64(int32[:], int32[:]))
    def desir_annotation_vector_jit(bs, es):
        b_repr = bs[0]
        if b_repr >= len(annotation_vector):
            return 0.0
        else:
            return annotation_vector[b_repr]
    return desir_annotation_vector_jit