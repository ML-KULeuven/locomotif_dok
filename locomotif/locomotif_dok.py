# LoCoMotif-DoK algorithm
# an extension of LoCoMotif to incorporate domain knowledge into the motif discovery process

import numpy as np
import warnings
import copy
import time

import numba
from numba import int32, float64, float32, boolean, optional
from numba import njit
from numba.typed import List
from numba.types import bool_, int_, float64, Tuple

import locomotif.locomotif as locomotif
from locomotif.locomotif import Path
from . import util
import locomotif.catalogue_of_constraints as cat



def apply_locomotifdok(ts, l_min, l_max, rho=None, nb=None, start_mask=None, end_mask=None, overlap=0.0, warping=True, 
                       tau=None, 
                       h_mot_repr_all=None, h_mot_all=None, h_mots_same_all=None, h_mset_all=None, h_mots_diff_all=None, h_msets_pairwise_all=None, desir_all=None, k_max_discard_all=None, 
                       assume_symmetric_constraints=True,
                       verbose=True):
    """Apply the LoCoMotif-DoK algorithm to find motif sets in the given time ts by leveraging the provided domain knowledge.

    :param ts: Univariate or multivariate time series, with the time axis being the 0-th dimension.
    :param l_min: Minimum length of the representative motifs.
    :param l_max: Maximum length of the representative motifs.
    :param rho: The strictness parameter between 0 and 1. It is the quantile of the similarity matrix to use as the threshold for the LoCo algorithm.
    :param nb: Maximum number of motif sets to find.
    :param start_mask: Mask for the starting time points of representative motifs, where True means allowed. If None, all points are allowed.
    :param end_mask: Mask for the ending time points of representative motifs, where True means allowed. If None, all points are allowed.
    :param overlap: Maximum allowed overlap between motifs, between 0 and 1. A new motif β can be discovered only when |β ∩ β'|/|β'| is less than this value for all existing motifs β'. 
            Default is 0, where no overlaps are allowed.
    :param warping: Whether warping is allowed (True) or not (False).
    :param tau: The threshold for the similarity matrix. If None, it is estimated based on the similarity matrix and rho.
    :param h_mot_repr_all: hard motif constraints for representative motifs in discovered motif sets
            (list of boolean functions of length nb, or a boolean function to apply to (the representative motif in) every motif set)
    :param h_mot_all: hard motif constraints for every motif in discovered motif sets
            (list of boolean functions of length nb, or a boolean function to apply to (motifs in) every motif set)
    :param h_mots_same_all: hard motif constraints for pairs of motifs within every discovered motif set
            (list of boolean functions of length nb, or a boolean function to apply to (every pair of motifs in) every motif set)
    :param h_mset_all: hard motif set constraints for motif sets to discover 
            (list of boolean functions of length nb, or a boolean function to apply to every motif set)
    :param h_mots_diff_all: hard motif constraints for pairs of motifs across discovered motif sets
            (list of lists of boolean functions of size nb×nb, or a boolean function to apply to every pair of motifs in distinct motif sets)
    :param h_msets_pairwise_all: pairwise hard motif set constraints for distinct pairs of motif sets to discover 
            (list of lists of boolean functions of size nb×nb, or a boolean function to apply to every pair of distinct motif sets)
    :param desir_all: desirability functions (soft constraints) for motif sets to discover 
            (list of scalar functions of length nb, or a scalar function to apply to every motif set)
    :param k_max_discard_all: maximum number of motifs for every motif set, beyond which motifs are discarded (0 or None means no limit)
            (list of integers of length nb, or an integer to apply to every motif set)
    :param assume_symmetric_constraints: Whether to assume that pairwise constraints are symmetric (i.e., h(x, y) = h(y, x) ∀x,y) or not. If False, the constraints are applied in both directions.
    :param verbose: Whether to print the progress of the algorithm or not.
    
    :return: motif_sets: a list of motif sets, where each motif set contains the representative segment, a list of all segments (i.e., motifs), and the fitness (scaled by desirability).
    """   
    
    start_time_LoCoMotif_DoK = time.time()
    
    # Get a locomotif instance
    lcm = get_locomotifdok_instance(ts, l_min, l_max, rho=rho, warping=warping, nb=nb, 
                                    overlap=overlap, start_mask=start_mask, end_mask=end_mask, 
                                    tau=tau, 
                                    h_mot_repr_all=h_mot_repr_all, h_mot_all=h_mot_all, h_mots_same_all=h_mots_same_all, h_mset_all=h_mset_all, h_mots_diff_all=h_mots_diff_all, h_msets_pairwise_all=h_msets_pairwise_all, desir_all=desir_all, k_max_discard_all=k_max_discard_all, 
                                    assume_symmetric_constraints=assume_symmetric_constraints)
    if verbose:
        if lcm.same_constraints_for_all_motif_sets:
            print('Applying LoCoMotif-DoK with the same constraints for all (pairs of) motif sets...')
        else:
            print('Applying LoCoMotif-DoK with different constraints for (pairs of) motif sets...')
    
    # Apply LoCo:
    start_time = time.time()
    if verbose:
        print('\nApplying LoCo...')
    lcm.align()
    if verbose:
        print(f' • Cumulative similarities computed in {time_duration_str(start_time)}.')
    start_time = time.time()
    lcm.find_best_paths(vwidth=l_min // 2)
    if verbose:
        print(f' • Local warping paths found in {time_duration_str(start_time)}.')
    
    if verbose:
        print(f'\nSearching for {lcm.nb} motif sets...')
    motif_sets = lcm.find_multiple_best_motif_sets(verbose=verbose)
    
    if verbose:
        print(f'LoCoMotif-DoK completed in {time_duration_str(start_time_LoCoMotif_DoK)}.')
            
    return motif_sets


def get_locomotifdok_instance(ts, l_min, l_max, rho=None, warping=True, nb=None, ts2=None, 
                              overlap=None, start_mask=None, end_mask=None, 
                              tau=None, 
                              h_mot_repr_all=None, h_mot_all=None, h_mots_same_all=None, h_mset_all=None, h_mots_diff_all=None, h_msets_pairwise_all=None, desir_all=None, k_max_discard_all=None, assume_symmetric_constraints=True):
    return LoCoMotifDoK.instance_from_rho_or_tau(ts, l_min=l_min, l_max=l_max, rho=rho, warping=warping, nb=nb, ts2=ts2, 
                                                 overlap=overlap, start_mask=start_mask, end_mask=end_mask, 
                                                 tau=tau, 
                                                 h_mot_repr_all=h_mot_repr_all, h_mot_all=h_mot_all, h_mots_same_all=h_mots_same_all, h_mset_all=h_mset_all, h_mots_diff_all=h_mots_diff_all, h_msets_pairwise_all=h_msets_pairwise_all, desir_all=desir_all, k_max_discard_all=k_max_discard_all, 
                                                 assume_symmetric_constraints=assume_symmetric_constraints)


class LoCoMotifDoK(locomotif.LoCoMotif):
    
    def __init__(self, ts, l_min, l_max, gamma=1.0, tau=0.5, delta_a=1.0, delta_m=0.5, step_sizes=None, nb=None, ts2=None, 
                 overlap=0.0, start_mask=None, end_mask=None, 
                 h_mot_repr_all=None, h_mot_all=None, h_mots_same_all=None, h_mset_all=None, h_mots_diff_all=None, h_msets_pairwise_all=None, desir_all=None, k_max_discard_all=None, 
                 assume_symmetric_constraints=True):
        super().__init__(ts, l_min, l_max, gamma, tau, delta_a, delta_m, step_sizes, ts2)
        
        self.nb = nb
        
        assert 0.0 <= overlap and overlap <= 1 #0.5
        self.overlap = overlap
        
        n = len(self.ts)
        if start_mask is None:
            start_mask = np.full(n, True) # True means allowed
        if end_mask is None:
            end_mask   = np.full(n, True) # True means allowed
        assert start_mask.shape == (n,)
        assert end_mask.shape   == (n,)
        self.start_mask = start_mask.copy() if start_mask is not None else None
        self.end_mask   = end_mask.copy()   if end_mask   is not None else None
        self.assume_symmetric_constraints = assume_symmetric_constraints
        
        # Check constraints for consistency:
        if not isinstance(h_mot_repr_all, list) and not isinstance(h_mot_all, list) and not isinstance(h_mots_same_all, list) and not isinstance(h_mset_all, list) and not isinstance(h_mots_diff_all, list) and not isinstance(h_msets_pairwise_all, list) and not isinstance(desir_all, list) and not isinstance(k_max_discard_all, (list, np.ndarray)): # all constraints are single
            self.same_constraints_for_all_motif_sets = True
            
            k_max_discard_all = k_max_discard_all if k_max_discard_all is not None and np.isfinite(k_max_discard_all) and k_max_discard_all>=0 else 0
            
        else: # there are multiple constraints, so make sure that each of them is a list of correct size
            self.same_constraints_for_all_motif_sets = False
            
            # TODO: Consider getting h_mots_same_all and h_mots_diff_all as a single list of lists
            
            # Set nb if not provided:
            if self.nb is None:
                for h in [h_mot_repr_all, h_mot_all, h_mots_same_all, h_mset_all, h_mots_diff_all, h_msets_pairwise_all, desir_all]:
                    if h is not None and isinstance(h, list):
                        self.nb = len(h)
                        break
                if k_max_discard_all is not None and isinstance(k_max_discard_all, (list, np.ndarray)):
                    self.nb = len(k_max_discard_all)
            if self.nb is None:
                raise ValueError("Cannot infer nb from constraints.")
            
            
            # Ensure h_mot_repr_all is a list of length nb:
            if not isinstance(h_mot_repr_all, list):
                h_mot_repr_all = [h_mot_repr_all] * self.nb
            # Ensure h_mot_all is a list of length nb:
            if not isinstance(h_mot_all, list):
                h_mot_all = [h_mot_all] * self.nb
            # Ensure h_mots_same_all is a list of length nb:
            if not isinstance(h_mots_same_all, list):
                h_mots_same_all = [h_mots_same_all] * self.nb
            # Ensure h_mset_all is a list of length nb:
            if not isinstance(h_mset_all, list):
                h_mset_all = [h_mset_all] * self.nb
            # Ensure h_mots_diff_all is a list of lists (size nb×nb):
            if not isinstance(h_mots_diff_all, list):
                h_mots_diff_all = [[None if i == j else h_mots_diff_all for j in range(self.nb)] for i in range(self.nb)] # keep the diagonal None to apply the constraint on distinct motif sets
            # Ensure h_msets_pairwise_all is a list of lists (size nb×nb):
            if not isinstance(h_msets_pairwise_all, list):
                h_msets_pairwise_all = [[None if i == j else h_msets_pairwise_all for j in range(self.nb)] for i in range(self.nb)] # keep the diagonal None to apply the constraint on distinct motif sets
            # Ensure desir_all is a list of length nb:
            if not isinstance(desir_all, list):
                desir_all = [desir_all] * self.nb
            # Ensure k_max_discard_all is a np.array of length nb of type int32, where values that are not positive are replaced by 0:
            if not isinstance(k_max_discard_all, (list, np.ndarray)):
                k_max_discard_all = np.full(self.nb, k_max_discard_all if k_max_discard_all is not None and np.isfinite(k_max_discard_all) and k_max_discard_all>=0 else 0, dtype=np.int32)
            else:
                k_max_discard_all = np.array([k if k is not None and np.isfinite(k) and k>=0 else 0 for k in k_max_discard_all], dtype=np.int32)
            

            # Ensure h_mot_repr_all, h_mot_all, h_mots_same_all, h_mset_all, and desir_all have the same length:
            if not (len(h_mot_repr_all) == len(h_mot_all) == len(h_mots_same_all) == len(h_mset_all) == len(desir_all) == self.nb):
                raise ValueError("h_mot_repr_all, h_mot_all, h_mots_same_all, h_mset_all, and desir_all must all have the same length and be consistent with nb_motifs (if provided).")
            
            # Ensure h_mots_diff_all and h_msets_pairwise_all are square arrays represented as a list of lists:
            if not all(isinstance(row, list) for row in h_mots_diff_all) or not all(len(row) == len(h_mots_diff_all) for row in h_mots_diff_all):
                raise ValueError("h_mots_diff_all must be a square array represented as a list of lists, with a size consistent with nb_motifs (if provided).")
            if not all(isinstance(row, list) for row in h_msets_pairwise_all) or not all(len(row) == len(h_msets_pairwise_all) for row in h_msets_pairwise_all):
                raise ValueError("h_msets_pairwise_all must be a square array represented as a list of lists, with a size consistent with nb_motifs (if provided).")
            
            # Warn if h_mots_diff_all and h_msets_pairwise_all are not None on the diagonal:
            if any(h_mots_diff_all[i][i] is not None for i in range(self.nb)):
                warnings.warn("The diagonal of h_mots_diff_all is ignored. Pairwise motif constraints within motif sets can be specified using h_mots_same_all.")
            if any(h_msets_pairwise_all[i][i] is not None for i in range(self.nb)):
                warnings.warn("The diagonal of h_msets_pairwise_all is ignored. Pairwise motif set constraints can be specified using h_mset_all.")
        
        self.h_mot_repr_all = h_mot_repr_all
        self.h_mot_all = h_mot_all
        self.h_mots_same_all = h_mots_same_all
        self.h_mset_all = h_mset_all
        self.h_mots_diff_all = h_mots_diff_all
        self.h_msets_pairwise_all = h_msets_pairwise_all
        self.desir_all = desir_all
        self.k_max_discard_all = k_max_discard_all
    
    
    @classmethod
    def instance_from_rho_or_tau(cls, ts, l_min, l_max, rho=None, warping=True, ts2=None, nb=None, 
                          overlap=None, start_mask=None, end_mask=None, 
                          tau=None, 
                          h_mot_repr_all=None, h_mot_all=None, h_mots_same_all=None, h_mset_all=None, h_mots_diff_all=None, h_msets_pairwise_all=None, desir_all=None, k_max_discard_all=None, 
                          assume_symmetric_constraints=True):
        # Handle default rho value
        if rho is None:
            rho = 0.8 if warping else 0.5
        # Make ts of shape (n,) of shape (n, 1) such that it can be handled as a multivariate ts
        if ts.ndim == 1:
            ts = np.expand_dims(ts, axis=1)
        ts = np.array(ts, dtype=np.float32)
        # if ts2 is None:
        #     ts2 = ts
        #     issym = True
        # else:
        if ts2 is not None:
            if ts2.ndim == 1:
                ts2 = np.expand_dims(ts2, axis=1)
            ts2 = np.array(ts2, dtype=np.float32)
        #     issym = False
        # Check whether the time series is z-normalized. If not, give a warning.
        if not util.is_unitstd(ts): # util.is_znormalized(ts):
            warnings.warn(
                "It is highly recommended to z-normalize the input time series so that it has a standard deviation of 1 before applying LoCoMotif to it.")

        gamma = 1
        # Determine values for tau, delta_a, delta_m based on the ssm and rho
        sm = locomotif.similarity_matrix_ndim(ts, ts if ts2 is None else ts2, gamma, only_triu=(ts2 is None))
        if tau is None:
            tau = locomotif.estimate_tau_from_sm(sm, rho, only_triu=(ts2 is None))

        delta_a = 2 * tau
        delta_m = 0.5
        # Determine step_sizes based on warping
        step_sizes = np.array([(1, 1), (2, 1), (1, 2)]) if warping else np.array([(1, 1)])
        lcm = cls(ts=ts, l_min=l_min, l_max=l_max, gamma=gamma, tau=tau, delta_a=delta_a, delta_m=delta_m,
                  step_sizes=step_sizes, nb=nb, ts2=ts2, 
                  overlap=overlap, start_mask=start_mask, end_mask=end_mask, 
                  h_mot_repr_all=h_mot_repr_all, h_mot_all=h_mot_all, h_mots_same_all=h_mots_same_all, h_mset_all=h_mset_all, h_mots_diff_all=h_mots_diff_all, h_msets_pairwise_all=h_msets_pairwise_all, desir_all=desir_all, k_max_discard_all=k_max_discard_all, 
                  assume_symmetric_constraints=assume_symmetric_constraints)
        lcm._sm = sm
        return lcm
    
    
    # Iteratively finds multiple best motif sets (without using a generator, i.e., without the "yield" keyword):
    def find_multiple_best_motif_sets(self, verbose=True):
        if self.same_constraints_for_all_motif_sets:
            return self.find_best_multiple_motif_sets_with_same_constraints(verbose=verbose)
        else:
            return self.find_best_multiple_motif_sets_with_different_constraints(verbose=verbose)


    def find_best_multiple_motif_sets_with_same_constraints(self, verbose=True):
        assume_symmetric_constraints = self.assume_symmetric_constraints
        start_mask = self.start_mask
        end_mask = self.end_mask
        
        n = len(self.ts)

        # Iteratively find best motif sets:
        current_nb = 0
        motif_sets = []
        mask       = np.full(n, False) # False means allowed
        
        h_mot_all_iter  = [copy.deepcopy(self.h_mot_all)]  # motif constraints for all motifs in all iterations (a list of constraints if self.same_constraints_for_all_motif_sets, or a list of lists of constraints otherwise)
        h_mset_all_iter = [copy.deepcopy(self.h_mset_all)] # motif set constraints for           all iterations (a list of constraints if self.same_constraints_for_all_motif_sets, or a list of lists of constraints otherwise)
        # TODO: copying may not be necessary
        
        while (self.nb is None or current_nb < self.nb):

            if np.all(mask) or not np.any(start_mask) or not np.any(end_mask):
                break

            start_mask[mask] = False
            end_mask[mask]   = False
            
            # if verbose:
            #     print(f" • Searching for motif set {current_nb + 1}...")
            start_time = time.time()

            # Run the standard LoCoMotif with constraints applied in _find_best_candidate_with_dok:
            (b, e), best_fitness, best_bs, best_es, fitnesses_of_candidate_motif_sets = \
                _find_best_candidate_with_dok(
                    start_mask, end_mask, mask, paths=self._paths, l_min=self.l_min, l_max=self.l_max, overlap=self.overlap, 
                    h_mot_repr=self.h_mot_repr_all if self.h_mot_repr_all is not None else h_mot_dummy, 
                    h_mot=h_mot_all_iter[-1] if h_mot_all_iter[-1] is not None else h_mot_dummy, # self.h_mot_all if self.h_mot_all is not None else h_mot_dummy, 
                    h_mots_same=self.h_mots_same_all if self.h_mots_same_all is not None else h_mots_same_dummy, 
                    h_mset=h_mset_all_iter[-1] if h_mset_all_iter[-1] is not None else h_mset_dummy, #self.h_mset_all if self.h_mset_all is not None else h_mset_dummy, 
                    desir=self.desir_all if self.desir_all is not None else desir_dummy, 
                    k_max_discard=self.k_max_discard_all,
                    check_h_mots_same=self.h_mots_same_all is not None,
                    keep_fitnesses=False, 
                    assume_symmetric_h_mots_same=assume_symmetric_constraints
                )
            
            motif_set = list(zip(best_bs, best_es))

            if best_fitness == 0.0:
                break
            
            for (b_m, e_m) in motif_set:
                l = e_m - b_m
                l_mask = max(1, int((1 - 2*self.overlap) * l)) # mask length must be lower bounded by 1 (otherwise, nothing is masked when overlap=0.5)
                mask[b_m + (l - l_mask)//2 : b_m + (l - l_mask)//2 + l_mask] = True

            current_nb += 1
            motif_sets.append(((b, e), motif_set, best_fitness))
            
            if verbose:
                print(f" • Discovered motif set {current_nb} with cardinality k = {len(motif_set)} and weighted fitness {best_fitness:.4f} in {time_duration_str(start_time)}.")
            
            # TODO: consider appending constraints incrementally for every newly discovered motif set
            # Convert the pairwise motif constraints across motif sets (h_mots_diff_all) to motif constraints and incorporate them into h_mot_all for the next iteration:
            if self.h_mots_diff_all is not None:
                h_mots_diff_all = self.h_mots_diff_all
                h_mot_all_new = []
                for ms in motif_sets:
                    for b_existing, e_existing in ms[1]:
                        @njit(boolean(int32, int32))
                        def h_mot_conv1(b, e):
                            return h_mots_diff_all(b, e, b_existing, e_existing)
                        h_mot_all_new.append(h_mot_conv1)
                        if not assume_symmetric_constraints:
                            @njit(boolean(int32, int32))
                            def h_mot_conv2(b, e):
                                return h_mots_diff_all(b_existing, e_existing, b, e)
                            h_mot_all_new.append(h_mot_conv2)
                h_mot_new = cat.combine_h_mot(*h_mot_all_new, self.h_mot_all) if self.h_mot_all is not None else cat.combine_h_mot(*h_mot_all_new)
                h_mot_all_iter.append(h_mot_new)
            else:
                h_mot_all_iter.append(self.h_mot_all)
            
            # Convert the pairwise motif set constraints (h_msets_pairwise_all) to motif set constraints and incorporate them into h_mset_all for the next iteration:
            if self.h_msets_pairwise_all is not None:
                h_msets_pairwise_all = self.h_msets_pairwise_all
                h_mset_all_new = []
                for ms in motif_sets:
                    bs_existing_ = np.array([m[0] for m in ms[1]], dtype=np.int32)
                    es_existing_ = np.array([m[1] for m in ms[1]], dtype=np.int32)
                    @njit(boolean(int32[:], int32[:]))
                    def h_mset_conv1(bs, es):
                        bs_existing = np.copy(bs_existing_)
                        es_existing = np.copy(es_existing_)
                        return h_msets_pairwise_all(bs, es, bs_existing, es_existing)
                    h_mset_all_new.append(h_mset_conv1)
                    if not assume_symmetric_constraints:
                        @njit(boolean(int32[:], int32[:]))
                        def h_mset_conv2(bs, es):
                            bs_existing = np.copy(bs_existing_)
                            es_existing = np.copy(es_existing_)
                            return h_msets_pairwise_all(bs_existing, es_existing, bs, es)
                        h_mset_all_new.append(h_mset_conv2)
                h_mset_new = cat.combine_h_mset(*h_mset_all_new, self.h_mset_all) if self.h_mset_all is not None else cat.combine_h_mset(*h_mset_all_new)
                h_mset_all_iter.append(h_mset_new)
            else:
                h_mset_all_iter.append(self.h_mset_all)
            
        return motif_sets # TODO: provide an option to return more results


    def find_best_multiple_motif_sets_with_different_constraints(self, verbose=True):
        assume_symmetric_constraints = self.assume_symmetric_constraints
        start_mask = self.start_mask
        end_mask = self.end_mask
        
        n = len(self.ts)

        # Iteratively find best motif sets:
        motif_sets       = [None for _ in range(self.nb)] # initialize the motif sets to discover
        inds_to_discover = list(range(self.nb)) # indices of motif sets to discover
        mask             = np.full(n, False) # False means allowed
        
        # h_mot_all_iter  = [copy.deepcopy(self.h_mot_all)]  # motif constraints for all motifs in all iterations (a list of constraints if self.same_constraints_for_all_motif_sets, or a list of lists of constraints otherwise)
        # h_mset_all_iter = [copy.deepcopy(self.h_mset_all)] # motif set constraints for           all iterations (a list of constraints if self.same_constraints_for_all_motif_sets, or a list of lists of constraints otherwise)
        h_mot_all = copy.deepcopy(self.h_mot_all)
        h_mset_all = copy.deepcopy(self.h_mset_all)
        # TODO: copying may not be necessary
        
        while len(inds_to_discover) > 0: # there are motif sets to discover

            if np.all(mask) or not np.any(start_mask) or not np.any(end_mask):
                break

            start_mask[mask] = False
            end_mask[mask]   = False
            
            start_time = time.time()

            # Run _find_best_candidate_with_dok multiple times with different constraints in every iteration:
            motif_sets_current_iter = [None for _ in inds_to_discover] # initialize the motif sets to discover in the current iteration
            fitnesses_current_iter = np.full(len(inds_to_discover), np.nan, dtype=np.float64)
            # TODO: can parallelize this for loop
            for j, i in enumerate(inds_to_discover): # i is the index of the motif set to discover
                (b, e), best_fitness, best_bs, best_es, fitnesses_of_candidate_motif_sets = \
                    _find_best_candidate_with_dok(
                        start_mask, end_mask, mask, paths=self._paths, l_min=self.l_min, l_max=self.l_max, overlap=self.overlap, 
                        h_mot_repr=self.h_mot_repr_all[i] if self.h_mot_repr_all[i] is not None else h_mot_dummy, 
                        h_mot=h_mot_all[i] if h_mot_all[i] is not None else h_mot_dummy, 
                        h_mots_same=self.h_mots_same_all[i] if self.h_mots_same_all[i] is not None else h_mots_same_dummy, 
                        h_mset=h_mset_all[i] if h_mset_all[i] is not None else h_mset_dummy, 
                        desir=self.desir_all[i] if self.desir_all[i] is not None else desir_dummy, 
                        k_max_discard=self.k_max_discard_all[i],
                        check_h_mots_same=self.h_mots_same_all[i] is not None,
                        keep_fitnesses=False, 
                        assume_symmetric_h_mots_same=assume_symmetric_constraints
                    )
                motif_set = list(zip(best_bs, best_es))
                motif_sets_current_iter[j] = ((b, e), motif_set, best_fitness)
                fitnesses_current_iter[j] = best_fitness
            
            # Select and save the best motif set among the discovered ones:
            if np.max(fitnesses_current_iter) <= 0.0:
                break # nothing found
            best_j = np.argmax(fitnesses_current_iter)
            best_i = inds_to_discover[best_j] # the discovered motif set in the current iteration
            motif_sets[best_i] = motif_sets_current_iter[best_j]
            inds_to_discover.remove(best_i)
            
            if verbose:
                print(f" • Discovered motif set {best_i + 1} with cardinality k = {len(motif_sets[best_i][1])} and weighted fitness {motif_sets[best_i][2]:.4f} in {time_duration_str(start_time)}.")
            
            for (b_m, e_m) in motif_sets[best_i][1]:
                l = e_m - b_m
                l_mask = max(1, int((1 - 2*overlap) * l)) # mask length must be lower bounded by 1 (otherwise, nothing is masked when overlap=0.5)
                mask[b_m + (l - l_mask)//2 : b_m + (l - l_mask)//2 + l_mask] = True
            
            if len(inds_to_discover) == 0: 
                break
            
            for i in inds_to_discover: # constraints of motif set i to discover will be updated
                # Convert the pairwise motif constraints across motif sets (h_mots_diff_all) that involve motif set i into motif constraints and incorporate them into h_mot_all[i] for the next iterations:
                h_mots_diff_all = self.h_mots_diff_all
                h_mot_all_new = []
                if h_mots_diff_all[i][best_i] is not None:
                    for b_existing, e_existing in motif_sets[best_i][1]:
                        h = h_mots_diff_all[i][best_i]
                        @njit(boolean(int32, int32))
                        def h_mot_conv1(b, e):
                            return h(b, e, b_existing, e_existing)
                        h_mot_all_new.append(h_mot_conv1)
                if not assume_symmetric_constraints:
                    if h_mots_diff_all[best_i][i] is not None:
                        for b_existing, e_existing in motif_sets[best_i][1]:
                            h = h_mots_diff_all[best_i][i]
                            @njit(boolean(int32, int32))
                            def h_mot_conv2(b, e):
                                return h(b_existing, e_existing, b, e)
                            h_mot_all_new.append(h_mot_conv2)
                if h_mot_all[i] is not None:
                    h_mot_all_new.append(h_mot_all[i])
                if len(h_mot_all_new) > 0:
                    h_mot_all[i] = cat.combine_h_mot(*h_mot_all_new)
                
                # Convert the pairwise motif set constraints (h_msets_pairwise_all) that involve motif set i into motif set constraints and incorporate them into h_mset_all[i] for the next iterations:
                h_msets_pairwise_all = self.h_msets_pairwise_all
                bs_existing_ = np.array([m[0] for m in motif_sets[best_i][1]], dtype=np.int32)
                es_existing_ = np.array([m[1] for m in motif_sets[best_i][1]], dtype=np.int32)
                h_mset_all_new = []
                if h_msets_pairwise_all[i][best_i] is not None:
                    h = h_msets_pairwise_all[i][best_i]
                    @njit(boolean(int32[:], int32[:]))
                    def h_mset_conv1(bs, es):
                        bs_existing = np.copy(bs_existing_)
                        es_existing = np.copy(es_existing_)
                        return h(bs, es, bs_existing, es_existing)
                    h_mset_all_new.append(h_mset_conv1)
                if not assume_symmetric_constraints:
                    if h_msets_pairwise_all[best_i][i] is not None:
                        h = h_msets_pairwise_all[best_i][i]
                        @njit(boolean(int32[:], int32[:]))
                        def h_mset_conv2(bs, es):
                            bs_existing = np.copy(bs_existing_)
                            es_existing = np.copy(es_existing_)
                            return h(bs_existing, es_existing, bs, es)
                        h_mset_all_new.append(h_mset_conv2)
                if h_mset_all[i] is not None: 
                    h_mset_all_new.append(h_mset_all[i])
                if len(h_mset_all_new) > 0:
                    h_mset_all[i] = cat.combine_h_mset(*h_mset_all_new)
            
            
        return motif_sets # TODO: provide an option to return more results



@njit(boolean(int32, int32))
def h_mot_dummy(b, e):
    return True


@njit(boolean(int32, int32, int32, int32))
def h_mots_same_dummy(b1, e1, b2, e2):
    return True


@njit(boolean(int32[:], int32[:]))
def h_mset_dummy(bs, es):
    return True


@njit(float64(int32[:], int32[:]))
def desir_dummy(bs, es):
    return 1.0


@njit(numba.types.Tuple((numba.types.UniTuple(int32, 2), float32, int32[:], int32[:], float32[:, :]))\
      (boolean[:], boolean[:], boolean[:], numba.types.ListType(Path.class_type.instance_type), int32, int32, float64, 
       numba.types.FunctionType(boolean(int32, int32)), numba.types.FunctionType(boolean(int32, int32)), numba.types.FunctionType(boolean(int32, int32, int32, int32)), numba.types.FunctionType(boolean(int32[:], int32[:])), numba.types.FunctionType(float64(int32[:], int32[:])), optional(int32),
       optional(boolean),
       optional(boolean),
       optional(boolean)))
def _find_best_candidate_with_dok(start_mask, end_mask, mask, paths, l_min, l_max, overlap=0.0, 
                                  h_mot_repr=h_mot_dummy, h_mot=h_mot_dummy, h_mots_same=h_mots_same_dummy, h_mset=h_mset_dummy, desir=desir_dummy, k_max_discard=0,
                                  check_h_mots_same=True,
                                  keep_fitnesses=False, 
                                  assume_symmetric_h_mots_same=True):
    fitnesses = []    
    n = len(start_mask)

    # j1s and jls respectively contain the column index of the first and last position of all paths
    j1s = np.array([path.j1 for path in paths])
    jls = np.array([path.jl for path in paths])

    nbp = len(paths)

    # bs and es will respectively contain the start and end indices of the motifs in the  candidate motif set of the current candidate [b : e].
    bs  = np.zeros(nbp, dtype=np.int32)
    es  = np.zeros(nbp, dtype=np.int32)

    # kbs and kes will respectively contain the index on the path (\in [0 : len(path)]) where the path crosses the vertical line through b and e.
    kbs = np.zeros(nbp, dtype=np.int32)
    kes = np.zeros(nbp, dtype=np.int32)
    
    covered = np.full(n, False) 

    best_fitness   = 0.0
    best_candidate = (0, n)
    best_bs_sorted_by_score = np.array([-1], dtype=np.int32)
    best_es_sorted_by_score = np.array([-1], dtype=np.int32)

    for b in np.arange(n - l_min + 1, dtype=np.int32):
        
        if not start_mask[b]:
            continue
            
        smask = j1s <= b        

        for e in np.arange(b + l_min, min(n + 1, b + l_max + 1), dtype=np.int32):
            
            if not end_mask[e-1]:
                continue

            if np.any(mask[b:e]):
                break

            emask = jls >= e
            pmask = smask & emask # mask of paths span (b, e) after checking start_mask, end_mask, and mask

            # If there are not paths that cross both the vertical line through b and e, skip the candidate.
            if not np.any(pmask[1:]):
                break

            # Check representative segment in terms of the representative motif constraint and the motif constraint:
            if not h_mot_repr(b, e) or not h_mot(b, e):
                continue

            for p in np.flatnonzero(pmask):
                path = paths[p]
                kbs[p] = pi = path.find_j(b)   # index of the first occurrence of b   in the path
                kes[p] = pj = path.find_j(e-1) # index of the first occurrence of e-1 in the path
                bs[p] = path[pi][0]     # starting point of the induced segment of the path
                es[p] = path[pj][0] + 1 # end      point of the induced segment of the path
                # Check overlap with previously found motifs.
                if np.any(mask[bs[p]:es[p]]): # or es[p] - bs[p] < l_min or es[p] - bs[p] > l_max:
                    pmask[p] = False
                # Check the motif constraint:
                if not h_mot(bs[p], es[p]):
                    pmask[p] = False

            # If the candidate only matches with itself, skip it.
            if not np.any(pmask[1:]):
                break

            bs_ = bs[pmask]
            es_ = es[pmask]

            # Sort bs and es on bs such that overlaps can be calculated efficiently:
            perm = np.argsort(bs_)
            bs_chron = bs_[perm] # in chronological order according to bs_
            es_chron = es_[perm] # in chronological order according to bs_
            
            # Calculate the overlaps   
            len_     = es_chron - bs_chron
            len_[:-1] = np.minimum(len_[:-1], len_[1:])
            overlaps  = np.maximum(es_chron[:-1] - bs_chron[1:] - 1, 0) 
            
            # Overlap check within motif set
            if np.any(overlaps > overlap * len_[:-1]): 
                continue

            # Compute the score of each motif:
            scores = np.full(np.sum(pmask), np.nan, dtype=np.float64)
            for i, p in enumerate(np.flatnonzero(pmask)):
                scores[i] = paths[p].cumsim[kes[p]+1] - paths[p].cumsim[kbs[p]]
            
            # Sort the motifs by their scores in descending order:
            perm_score = np.argsort(scores)[::-1]
            scores_sorted = scores[perm_score]
            bs_sorted_by_score = bs_[perm_score]
            es_sorted_by_score = es_[perm_score]
            
            # Make pairwise motif constraint check optional for efficiency
            if check_h_mots_same or k_max_discard != 0:
                # Start with an empty set and add motifs that satisfy the pairwise motif constraint one by one up do k_max_discard motifs:
                admissible_motif_mask = np.zeros(len(bs_sorted_by_score), dtype=np.bool_)
                for i in range(len(admissible_motif_mask)):
                    if k_max_discard != 0 and np.sum(admissible_motif_mask) >= k_max_discard: # there are at least k_max_discard motifs, so we shouldn't add more
                        break
                    is_admissible = True
                    for j in np.flatnonzero(admissible_motif_mask):
                        if not h_mots_same(bs_sorted_by_score[i], es_sorted_by_score[i], bs_sorted_by_score[j], es_sorted_by_score[j]) or (not assume_symmetric_h_mots_same and not h_mots_same(bs_sorted_by_score[j], es_sorted_by_score[j], bs_sorted_by_score[i], es_sorted_by_score[i])):
                            is_admissible = False
                            break
                    if is_admissible:
                        admissible_motif_mask[i] = True
            else:
                admissible_motif_mask = np.ones(len(bs_sorted_by_score), dtype=np.bool_)
                
            scores_sorted = scores_sorted[admissible_motif_mask]
            bs_sorted_by_score = bs_sorted_by_score[admissible_motif_mask]
            es_sorted_by_score = es_sorted_by_score[admissible_motif_mask]
            
            # Check the motif set constraint:
            if not h_mset(bs_sorted_by_score, es_sorted_by_score):
                continue
            
            # Apply admissible_motif_mask to pmask
            pmask[np.flatnonzero(pmask)[perm_score]] = admissible_motif_mask
            
            # Calculate normalized score
            n_score = (np.sum(scores_sorted) - (e - b)) / float(np.sum(kes[pmask] - kbs[pmask] + 1))
            
            # Calculate normalized coverage (using a more robust way that supports any value of overlap from 0 to 1):
            covered[:] = False
            for b_, e_ in zip(bs_sorted_by_score, es_sorted_by_score):
                covered[b_:e_] = True
            n_coverage = ( np.sum(covered) - (e - b) ) / float(n)
            
            # Calculate the fitness value
            fit = 0.0
            if n_coverage > 0 and n_score > 0:
                fit = 2 * (n_coverage * n_score) / (n_coverage + n_score)
            
            if fit <= 0.0:
                continue
            
            # Scale the fitness by the desirability of the candidate motif set:
            fit *= desir(bs_sorted_by_score, es_sorted_by_score)
    
            if fit <= 0.0:
                continue

            # Update best fitness
            if fit > best_fitness:
                best_candidate = (b, e)
                best_fitness   = fit
                best_bs_sorted_by_score = bs_sorted_by_score
                best_es_sorted_by_score = es_sorted_by_score

            # Store fitness if necessary
            if keep_fitnesses:
                fitnesses.append((b, e, fit, n_coverage, n_score))
    
    fitnesses = np.array(fitnesses, dtype=np.float32) if keep_fitnesses else np.empty((0, 5), dtype=np.float32)
    
    return best_candidate, best_fitness, best_bs_sorted_by_score, best_es_sorted_by_score, fitnesses


def time_duration_str(start_time):
    return time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)) + ' sec'


# def _is_constraint_defined(constraint_all): 
#     if constraint_all is None: 
#         return False 
#     elif isinstance(constraint_all, list): 
#         return all(_is_constraint_defined(item) for item in constraint_all) 
#     return True

# def _is_constraint_undefined_or_single_or_multiple(constraint_all):
#     if not _is_constraint_defined(constraint_all):
#         return 0
#     elif isinstance(constraint_all, list) and len(constraint_all) > 1:
#         return 2
#     else:
#         return 1




# def _check_constraint_for_every_motif_set(constraint_all):
#     if not _is_constraint_defined(constraint_all):
#         constraint_all = None
#     else:
#         if not _is_multiple_constraints(constraint_all):
#             return constraint_all
#     if _is_multiple_constraints(constraint_all) or not _is_constraint_defined(constraint_all):
#         return
#     pass


# def _check_constraint_for_every_pair_of_motif_sets():
#     pass

    
# def _check_constraints(hmset_all, hmsets_pairwise_all, desir_all):
#     are_undefined_or_single_or_multiple = np.array([_is_constraint_undefined_or_single_or_multiple(constraint_all) for constraint_all in [hmset_all, hmsets_pairwise_all, desir_all]])
#     same_constraints_for_all_motif_sets = False if any(are_undefined_or_single_or_multiple==2) else True
    
#     if same_constraints_for_all_motif_sets: 
#         hmset_all = hmset_all[0] if isinstance(hmset_all, list) else hmset_all