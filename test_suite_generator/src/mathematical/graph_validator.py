"""
Graph Theory Validation

Validates graph properties including DAG verification, cycle detection,
path length analysis, and prerequisite graph validation using NetworkX.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
import logging

import networkx as nx

logger = logging.getLogger(__name__)


class GraphType(Enum):
    """Types of graphs."""
    DIRECTED = "directed"
    UNDIRECTED = "undirected"
    DAG = "dag"


@dataclass
class GraphMetrics:
    """
    Graph structure metrics.
    
    Attributes:
        num_nodes: Number of nodes
        num_edges: Number of edges
        density: Graph density
        avg_in_degree: Average in-degree
        avg_out_degree: Average out-degree
        max_in_degree: Maximum in-degree
        max_out_degree: Maximum out-degree
        is_dag: Whether graph is a DAG
        longest_path_length: Length of longest path
        num_connected_components: Number of connected components
    """
    num_nodes: int
    num_edges: int
    density: float
    avg_in_degree: float
    avg_out_degree: float
    max_in_degree: int
    max_out_degree: int
    is_dag: bool
    longest_path_length: Optional[int]
    num_connected_components: int


@dataclass
class GraphValidationResult:
    """
    Result of graph validation.
    
    Attributes:
        graph_name: Name of graph
        passed: Whether validation passed
        violations: List of violation messages
        metrics: Graph metrics
        details: Additional validation details
    """
    graph_name: str
    passed: bool
    violations: List[str]
    metrics: GraphMetrics
    details: Dict[str, Any]


class GraphValidator:
    """
    Graph theory validator using NetworkX.
    
    Validates graph properties for prerequisite graphs,
    dependency graphs, and other relationships.
    """
    
    def __init__(self):
        """Initialize graph validator."""
        self.graphs: Dict[str, nx.DiGraph] = {}
        logger.info("GraphValidator initialized")
    
    def create_graph(
        self,
        name: str,
        graph_type: GraphType = GraphType.DIRECTED
    ) -> nx.DiGraph:
        """
        Create a new graph.
        
        Args:
            name: Graph name
            graph_type: Type of graph
            
        Returns:
            NetworkX graph object
        """
        if graph_type == GraphType.DIRECTED or graph_type == GraphType.DAG:
            graph = nx.DiGraph()
        else:
            graph = nx.Graph()
        
        self.graphs[name] = graph
        logger.debug(f"Created graph: {name} ({graph_type.value})")
        return graph
    
    def add_nodes(
        self,
        graph_name: str,
        nodes: List[Any],
        **attributes
    ) -> None:
        """
        Add nodes to graph.
        
        Args:
            graph_name: Name of graph
            nodes: List of node identifiers
            **attributes: Node attributes
        """
        if graph_name not in self.graphs:
            raise ValueError(f"Graph not found: {graph_name}")
        
        graph = self.graphs[graph_name]
        graph.add_nodes_from(nodes, **attributes)
        logger.debug(f"Added {len(nodes)} nodes to {graph_name}")
    
    def add_edge(
        self,
        graph_name: str,
        source: Any,
        target: Any,
        **attributes
    ) -> None:
        """
        Add edge to graph.
        
        Args:
            graph_name: Name of graph
            source: Source node
            target: Target node
            **attributes: Edge attributes
        """
        if graph_name not in self.graphs:
            raise ValueError(f"Graph not found: {graph_name}")
        
        graph = self.graphs[graph_name]
        graph.add_edge(source, target, **attributes)
        logger.debug(f"Added edge {source} -> {target} to {graph_name}")
    
    def add_edges(
        self,
        graph_name: str,
        edges: List[Tuple[Any, Any]],
        **attributes
    ) -> None:
        """
        Add multiple edges to graph.
        
        Args:
            graph_name: Name of graph
            edges: List of (source, target) tuples
            **attributes: Edge attributes
        """
        if graph_name not in self.graphs:
            raise ValueError(f"Graph not found: {graph_name}")
        
        graph = self.graphs[graph_name]
        graph.add_edges_from(edges, **attributes)
        logger.debug(f"Added {len(edges)} edges to {graph_name}")
    
    def is_dag(self, graph_name: str) -> bool:
        """
        Check if graph is a Directed Acyclic Graph.
        
        Args:
            graph_name: Name of graph
            
        Returns:
            True if graph is a DAG, False otherwise
        """
        if graph_name not in self.graphs:
            raise ValueError(f"Graph not found: {graph_name}")
        
        graph = self.graphs[graph_name]
        is_dag = nx.is_directed_acyclic_graph(graph)
        logger.debug(f"Graph {graph_name} is {'a' if is_dag else 'not a'} DAG")
        return is_dag
    
    def find_cycles(self, graph_name: str) -> List[List[Any]]:
        """
        Find all cycles in graph.
        
        Args:
            graph_name: Name of graph
            
        Returns:
            List of cycles (each cycle is a list of nodes)
        """
        if graph_name not in self.graphs:
            raise ValueError(f"Graph not found: {graph_name}")
        
        graph = self.graphs[graph_name]
        
        try:
            cycles = list(nx.simple_cycles(graph))
            if cycles:
                logger.warning(f"Found {len(cycles)} cycles in {graph_name}")
            else:
                logger.debug(f"No cycles found in {graph_name}")
            return cycles
        except Exception as e:
            logger.error(f"Error finding cycles in {graph_name}: {e}")
            return []
    
    def longest_path(self, graph_name: str) -> List[Any]:
        """
        Find longest path in DAG.
        
        Args:
            graph_name: Name of graph (must be DAG)
            
        Returns:
            Longest path as list of nodes
        """
        if graph_name not in self.graphs:
            raise ValueError(f"Graph not found: {graph_name}")
        
        graph = self.graphs[graph_name]
        
        if not nx.is_directed_acyclic_graph(graph):
            logger.warning(f"Graph {graph_name} is not a DAG, cannot find longest path")
            return []
        
        try:
            path = nx.dag_longest_path(graph)
            logger.debug(f"Longest path in {graph_name}: length {len(path)}")
            return path
        except Exception as e:
            logger.error(f"Error finding longest path in {graph_name}: {e}")
            return []
    
    def longest_path_length(self, graph_name: str) -> int:
        """
        Get length of longest path in DAG.
        
        Args:
            graph_name: Name of graph (must be DAG)
            
        Returns:
            Length of longest path
        """
        path = self.longest_path(graph_name)
        return len(path) - 1 if len(path) > 0 else 0
    
    def topological_sort(self, graph_name: str) -> List[Any]:
        """
        Perform topological sort on DAG.
        
        Args:
            graph_name: Name of graph (must be DAG)
            
        Returns:
            Topologically sorted list of nodes
        """
        if graph_name not in self.graphs:
            raise ValueError(f"Graph not found: {graph_name}")
        
        graph = self.graphs[graph_name]
        
        if not nx.is_directed_acyclic_graph(graph):
            logger.warning(f"Graph {graph_name} is not a DAG, cannot topological sort")
            return []
        
        try:
            sorted_nodes = list(nx.topological_sort(graph))
            logger.debug(f"Topological sort of {graph_name}: {len(sorted_nodes)} nodes")
            return sorted_nodes
        except Exception as e:
            logger.error(f"Error topological sorting {graph_name}: {e}")
            return []
    
    def compute_metrics(self, graph_name: str) -> GraphMetrics:
        """
        Compute comprehensive graph metrics.
        
        Args:
            graph_name: Name of graph
            
        Returns:
            GraphMetrics object
        """
        if graph_name not in self.graphs:
            raise ValueError(f"Graph not found: {graph_name}")
        
        graph = self.graphs[graph_name]
        
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        density = nx.density(graph) if num_nodes > 0 else 0.0
        
        # Degree metrics
        in_degrees = [d for n, d in graph.in_degree()]
        out_degrees = [d for n, d in graph.out_degree()]
        
        avg_in_degree = sum(in_degrees) / num_nodes if num_nodes > 0 else 0.0
        avg_out_degree = sum(out_degrees) / num_nodes if num_nodes > 0 else 0.0
        max_in_degree = max(in_degrees) if in_degrees else 0
        max_out_degree = max(out_degrees) if out_degrees else 0
        
        # DAG properties
        is_dag = nx.is_directed_acyclic_graph(graph)
        longest_path_len = self.longest_path_length(graph_name) if is_dag else None
        
        # Connected components
        if isinstance(graph, nx.DiGraph):
            num_components = nx.number_weakly_connected_components(graph)
        else:
            num_components = nx.number_connected_components(graph)
        
        metrics = GraphMetrics(
            num_nodes=num_nodes,
            num_edges=num_edges,
            density=density,
            avg_in_degree=avg_in_degree,
            avg_out_degree=avg_out_degree,
            max_in_degree=max_in_degree,
            max_out_degree=max_out_degree,
            is_dag=is_dag,
            longest_path_length=longest_path_len,
            num_connected_components=num_components
        )
        
        logger.debug(f"Computed metrics for {graph_name}: {metrics}")
        return metrics
    
    def validate_prerequisite_graph(
        self,
        graph_name: str,
        max_depth: int = 4,
        max_prerequisites: int = 3
    ) -> GraphValidationResult:
        """
        Validate prerequisite graph according to foundation specifications.
        
        Args:
            graph_name: Name of prerequisite graph
            max_depth: Maximum allowed path length (default 4)
            max_prerequisites: Maximum prerequisites per course (default 3)
            
        Returns:
            GraphValidationResult
        """
        if graph_name not in self.graphs:
            raise ValueError(f"Graph not found: {graph_name}")
        
        violations = []
        graph = self.graphs[graph_name]
        
        # Compute metrics
        metrics = self.compute_metrics(graph_name)
        
        # Check 1: Must be a DAG
        if not metrics.is_dag:
            violations.append("Graph is not a DAG")
            cycles = self.find_cycles(graph_name)
            for i, cycle in enumerate(cycles[:5]):  # Show first 5 cycles
                violations.append(f"Cycle {i+1}: {' -> '.join(map(str, cycle))}")
        
        # Check 2: Maximum path length
        if metrics.is_dag and metrics.longest_path_length is not None:
            if metrics.longest_path_length > max_depth:
                violations.append(
                    f"Longest path length {metrics.longest_path_length} "
                    f"exceeds maximum {max_depth}"
                )
                longest = self.longest_path(graph_name)
                violations.append(f"Longest path: {' -> '.join(map(str, longest))}")
        
        # Check 3: Maximum in-degree (prerequisites per course)
        if metrics.max_in_degree > max_prerequisites:
            # Find nodes with too many prerequisites
            high_prereq_nodes = [
                (node, degree) 
                for node, degree in graph.in_degree()
                if degree > max_prerequisites
            ]
            violations.append(
                f"Maximum prerequisites {metrics.max_in_degree} "
                f"exceeds limit {max_prerequisites}"
            )
            for node, degree in high_prereq_nodes[:5]:  # Show first 5
                violations.append(f"Node {node} has {degree} prerequisites")
        
        # Check 4: Average prerequisites should be 1-2
        if not (1.0 <= metrics.avg_in_degree <= 2.0):
            violations.append(
                f"Average prerequisites {metrics.avg_in_degree:.2f} "
                f"outside recommended range [1.0, 2.0]"
            )
        
        passed = len(violations) == 0
        
        details = {
            "max_depth_allowed": max_depth,
            "max_prerequisites_allowed": max_prerequisites,
            "is_dag": metrics.is_dag,
            "actual_longest_path": metrics.longest_path_length,
            "actual_max_prerequisites": metrics.max_in_degree,
            "actual_avg_prerequisites": metrics.avg_in_degree
        }
        
        result = GraphValidationResult(
            graph_name=graph_name,
            passed=passed,
            violations=violations,
            metrics=metrics,
            details=details
        )
        
        if not passed:
            logger.warning(
                f"Prerequisite graph validation failed for {graph_name}: "
                f"{len(violations)} violations"
            )
        else:
            logger.info(f"Prerequisite graph validation passed for {graph_name}")
        
        return result
    
    def validate_no_self_loops(self, graph_name: str) -> bool:
        """
        Validate that graph has no self-loops.
        
        Args:
            graph_name: Name of graph
            
        Returns:
            True if no self-loops, False otherwise
        """
        if graph_name not in self.graphs:
            raise ValueError(f"Graph not found: {graph_name}")
        
        graph = self.graphs[graph_name]
        self_loops = list(nx.selfloop_edges(graph))
        
        if self_loops:
            logger.warning(f"Found {len(self_loops)} self-loops in {graph_name}")
            return False
        
        return True
    
    def get_reachable_nodes(
        self,
        graph_name: str,
        source: Any
    ) -> Set[Any]:
        """
        Get all nodes reachable from source node.
        
        Args:
            graph_name: Name of graph
            source: Source node
            
        Returns:
            Set of reachable nodes
        """
        if graph_name not in self.graphs:
            raise ValueError(f"Graph not found: {graph_name}")
        
        graph = self.graphs[graph_name]
        
        if source not in graph:
            return set()
        
        reachable = nx.descendants(graph, source)
        reachable.add(source)  # Include source itself
        return reachable
    
    def get_transitive_closure(self, graph_name: str) -> nx.DiGraph:
        """
        Compute transitive closure of graph.
        
        Args:
            graph_name: Name of graph
            
        Returns:
            New graph with transitive closure
        """
        if graph_name not in self.graphs:
            raise ValueError(f"Graph not found: {graph_name}")
        
        graph = self.graphs[graph_name]
        closure = nx.transitive_closure(graph)
        logger.debug(f"Computed transitive closure for {graph_name}")
        return closure
    
    def get_graph(self, graph_name: str) -> Optional[nx.DiGraph]:
        """Get graph by name."""
        return self.graphs.get(graph_name)
    
    def clear_graph(self, graph_name: str) -> None:
        """Clear a graph."""
        if graph_name in self.graphs:
            self.graphs[graph_name].clear()
            logger.debug(f"Cleared graph: {graph_name}")
    
    def remove_graph(self, graph_name: str) -> None:
        """Remove a graph."""
        if graph_name in self.graphs:
            del self.graphs[graph_name]
            logger.debug(f"Removed graph: {graph_name}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get graph validation statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_graphs": len(self.graphs),
            "graph_names": list(self.graphs.keys()),
            "graphs_detail": {
                name: {
                    "nodes": graph.number_of_nodes(),
                    "edges": graph.number_of_edges(),
                    "is_dag": nx.is_directed_acyclic_graph(graph)
                }
                for name, graph in self.graphs.items()
            }
        }
    
    def __repr__(self) -> str:
        return f"GraphValidator(graphs={len(self.graphs)})"
