"""
Objective Functions for Stage-2 Batching
Implements Theorem 2.1 Multi-Objective Formulation from OR-Tools Bridge
"""

import numpy as np
from typing import List


class ObjectiveFunctionBuilder:
    """
    Builds multi-objective functions for CP-SAT optimization.
    
    Implements Theorem 2.1 from OR-Tools CP-SAT Bridge Foundation.
    """
    
    def __init__(self, model, model_builder):
        """
        Initialize objective function builder.
        
        Args:
            model: CP-SAT model instance
            model_builder: CPSATBatchingModel instance
        """
        self.model = model
        self.mb = model_builder
    
    def build_f1_objective(self, target_sizes: List[int]):
        """
        Definition 4.1: Batch Size Optimization (f1)
        
        minimize: Σj |batch_size[j] - τj|²
        
        CP-SAT encoding with absolute deviation linearization.
        
        Args:
            target_sizes: List of target sizes for each batch
        
        Returns:
            Linear expression representing f1 objective
        """
        obj1_terms = []
        
        for j in range(self.mb.m):
            # Create positive and negative deviation variables
            size_dev_pos = self.model.NewIntVar(0, self.mb.n, f'size_dev_pos_{j}')
            size_dev_neg = self.model.NewIntVar(0, self.mb.n, f'size_dev_neg_{j}')
            
            # Constraint: batch_size[j] - target_sizes[j] = size_dev_pos - size_dev_neg
            self.model.Add(
                self.mb.batch_size[j] - target_sizes[j] == size_dev_pos - size_dev_neg
            )
            
            # Add to objective (absolute deviation)
            obj1_terms.append(size_dev_pos + size_dev_neg)
        
        return sum(obj1_terms)
    
    def build_f2_objective(self, similarity_matrix: np.ndarray):
        """
        Definition 4.3: Academic Homogeneity (f2)
        
        maximize: -Σj Σi,i'∈Bj sim(si, si')
        
        Encoded as minimization in CP-SAT (minimize negative).
        
        Args:
            similarity_matrix: NxN similarity matrix
        
        Returns:
            Linear expression representing -f2 objective
        """
        obj2_terms = []
        
        for j in range(self.mb.m):
            homogeneity_sum = []
            
            # Sum similarities for all pairs in batch j
            for i in range(self.mb.n):
                for i_prime in range(i + 1, self.mb.n):
                    # Indicator: both students in same batch
                    both_in_batch = self.model.NewBoolVar(f'both_{i}_{i_prime}_{j}')
                    
                    # Create AND constraint: x[i,j] AND x[i_prime,j] → both_in_batch
                    self.model.AddBoolAnd([
                        self.mb.x[i, j],
                        self.mb.x[i_prime, j]
                    ]).OnlyEnforceIf(both_in_batch)
                    
                    # Similarity contribution (scaled to integer)
                    sim_value = int(similarity_matrix[i, i_prime] * 10000)
                    
                    # Contribution only if both in batch
                    similarity_contrib = self.model.NewIntVar(
                        0, sim_value, f'sim_contrib_{i}_{i_prime}_{j}'
                    )
                    
                    self.model.Add(similarity_contrib == sim_value).OnlyEnforceIf(both_in_batch)
                    self.model.Add(similarity_contrib == 0).OnlyEnforceIf(both_in_batch.Not())
                    
                    homogeneity_sum.append(similarity_contrib)
            
            # Set homogeneity[j] = sum of all pair similarities
            if homogeneity_sum:
                self.model.Add(self.mb.homogeneity[j] == sum(homogeneity_sum))
            else:
                self.model.Add(self.mb.homogeneity[j] == 0)
            
            obj2_terms.append(self.mb.homogeneity[j])
        
        # Return negative (since we're maximizing homogeneity)
        return -sum(obj2_terms)
    
    def build_f3_objective(self, resource_demands: List[float]):
        """
        Definition 4.7: Resource Utilization Balance (f3)
        
        minimize: Variance of resource demands across batches
        
        Implemented via mean absolute deviation.
        
        Args:
            resource_demands: List of resource demands for each batch
        
        Returns:
            Linear expression representing f3 objective
        """
        # Deprecated; use build_f3_from_intvars with batch demand IntVars
        obj3_terms = []
        return sum(obj3_terms)

    def build_f3_from_intvars(self, batch_demands: list) -> any:
        """
        Definition 4.7: Resource Utilization Balance (f3)

        minimize: Variance (approximated via mean absolute deviation) of resource
        demands across batches.

        Args:
            batch_demands: List of IntVar, demand per batch
        """
        obj3_terms = []
        # Compute integer mean demand as constant: total_demand / m
        total_demand = 0
        # Approximate total_demand by sum of batch upper bounds; alternatively pass in precomputed
        # For rigor, rely on precomputed sum in model builder stored as self.mb.total_student_demand
        mean_demand = int(self.mb.total_student_demand // max(1, self.mb.m))

        # For each batch, compute absolute deviation from mean
        for j in range(self.mb.m):
            # dev_pos, dev_neg >= 0
            max_dev = self.mb.total_student_demand
            dev_pos = self.model.NewIntVar(0, max_dev, f'res_dev_pos_{j}')
            dev_neg = self.model.NewIntVar(0, max_dev, f'res_dev_neg_{j}')
            # demand_j - mean = dev_pos - dev_neg
            self.model.Add(batch_demands[j] - mean_demand == dev_pos - dev_neg)
            obj3_terms.append(dev_pos + dev_neg)

        return sum(obj3_terms)

