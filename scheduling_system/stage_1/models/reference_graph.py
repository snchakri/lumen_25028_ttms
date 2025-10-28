"""
Reference graph structures for referential integrity analysis.

Implements:
- Definition 5.1: Reference Graph Theory
- Theorem 5.5: Tarjan's Strongly Connected Components Algorithm
  for circular dependency detection in O(|V| + |E|) time
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict


@dataclass
class GraphNode:
    """Node in the reference graph."""
    node_id: str  # Entity ID
    entity_type: str  # Table name
    outgoing_edges: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class ReferenceGraph:
    """
    Reference graph G_R = (V, E) per Definition 5.1.
    
    V = entity types (tables)
    E = foreign key relationships
    
    Implements Tarjan's SCC algorithm for cycle detection (Theorem 5.5).
    """
    
    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.adjacency_list: Dict[str, List[str]] = defaultdict(list)
        self.reverse_adjacency: Dict[str, List[str]] = defaultdict(list)
        
        # Tarjan's algorithm state
        self._index = 0
        self._stack: List[str] = []
        self._indices: Dict[str, int] = {}
        self._lowlinks: Dict[str, int] = {}
        self._on_stack: Set[str] = set()
        self._sccs: List[List[str]] = []
    
    def add_node(self, node_id: str, entity_type: str, metadata: Optional[Dict] = None):
        """Add a node to the graph."""
        if node_id not in self.nodes:
            self.nodes[node_id] = GraphNode(
                node_id=node_id,
                entity_type=entity_type,
                metadata=metadata or {}
            )
    
    def add_edge(self, from_node: str, to_node: str):
        """
        Add a directed edge (foreign key reference).
        
        from_node references to_node (foreign key → primary key).
        """
        if from_node not in self.nodes or to_node not in self.nodes:
            raise ValueError(f"Both nodes must exist before adding edge: {from_node} → {to_node}")
        
        self.adjacency_list[from_node].append(to_node)
        self.reverse_adjacency[to_node].append(from_node)
        self.nodes[from_node].outgoing_edges.append(to_node)
    
    def has_edge(self, from_node: str, to_node: str) -> bool:
        """Check if edge exists."""
        return to_node in self.adjacency_list.get(from_node, [])
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get node by ID."""
        return self.nodes.get(node_id)
    
    def get_outgoing_edges(self, node_id: str) -> List[str]:
        """Get all outgoing edges from a node."""
        return self.adjacency_list.get(node_id, [])
    
    def get_incoming_edges(self, node_id: str) -> List[str]:
        """Get all incoming edges to a node."""
        return self.reverse_adjacency.get(node_id, [])
    
    def node_count(self) -> int:
        """Get number of nodes."""
        return len(self.nodes)
    
    def edge_count(self) -> int:
        """Get number of edges."""
        return sum(len(edges) for edges in self.adjacency_list.values())
    
    def find_strongly_connected_components(self) -> List[List[str]]:
        """
        Find all strongly connected components using Tarjan's algorithm.
        
        Implements Theorem 5.5: Cycle Detection Correctness.
        Time Complexity: O(|V| + |E|)
        
        Returns:
            List of SCCs, where each SCC is a list of node IDs.
            SCCs with more than one node contain cycles.
        """
        # Reset Tarjan's algorithm state
        self._index = 0
        self._stack = []
        self._indices = {}
        self._lowlinks = {}
        self._on_stack = set()
        self._sccs = []
        
        # Run Tarjan's algorithm on each unvisited node
        for node_id in self.nodes:
            if node_id not in self._indices:
                self._tarjan_strongconnect(node_id)
        
        return self._sccs
    
    def _tarjan_strongconnect(self, v: str):
        """
        Tarjan's strongly connected components algorithm (recursive).
        
        Proof of Correctness (Theorem 5.5):
        - Maintains DFS tree with discovery times (indices)
        - Tracks lowest reachable ancestor (lowlinks)
        - Stack maintains potential SCC nodes
        - When v.lowlink == v.index, v is SCC root
        - All nodes on stack above v form complete SCC
        """
        # Set depth index for v
        self._indices[v] = self._index
        self._lowlinks[v] = self._index
        self._index += 1
        
        # Push v onto stack
        self._stack.append(v)
        self._on_stack.add(v)
        
        # Consider successors of v
        for w in self.adjacency_list.get(v, []):
            if w not in self._indices:
                # Successor w not yet visited; recurse
                self._tarjan_strongconnect(w)
                self._lowlinks[v] = min(self._lowlinks[v], self._lowlinks[w])
            elif w in self._on_stack:
                # Successor w is on stack (part of current SCC)
                self._lowlinks[v] = min(self._lowlinks[v], self._indices[w])
        
        # If v is a root node, pop the stack to get SCC
        if self._lowlinks[v] == self._indices[v]:
            scc = []
            while True:
                w = self._stack.pop()
                self._on_stack.remove(w)
                scc.append(w)
                if w == v:
                    break
            self._sccs.append(scc)
    
    def find_cycles(self) -> List[List[str]]:
        """
        Find all cycles in the graph.
        
        Returns only SCCs that contain more than one node (actual cycles).
        Self-loops are also considered cycles.
        """
        sccs = self.find_strongly_connected_components()
        cycles = []
        
        for scc in sccs:
            if len(scc) > 1:
                # Multiple nodes - definite cycle
                cycles.append(scc)
            elif len(scc) == 1:
                # Check for self-loop
                node = scc[0]
                if self.has_edge(node, node):
                    cycles.append(scc)
        
        return cycles
    
    def has_cycles(self) -> bool:
        """Check if graph contains any cycles."""
        return len(self.find_cycles()) > 0
    
    def get_cycle_details(self) -> List[Dict]:
        """
        Get detailed information about all cycles.
        
        Returns list of cycle details including nodes and paths.
        """
        cycles = self.find_cycles()
        cycle_details = []
        
        for idx, cycle in enumerate(cycles):
            cycle_info = {
                "cycle_id": idx + 1,
                "node_count": len(cycle),
                "nodes": cycle,
                "entity_types": [self.nodes[n].entity_type for n in cycle],
                "is_self_loop": len(cycle) == 1 and self.has_edge(cycle[0], cycle[0]),
            }
            
            # Try to find a simple path through the cycle for display
            if len(cycle) > 1:
                path = self._find_cycle_path(cycle)
                cycle_info["example_path"] = path
            else:
                cycle_info["example_path"] = [cycle[0], cycle[0]]
            
            cycle_details.append(cycle_info)
        
        return cycle_details
    
    def _find_cycle_path(self, scc_nodes: List[str]) -> List[str]:
        """
        Find a simple path through nodes in an SCC that demonstrates the cycle.
        
        Uses BFS to find path from first node back to itself through SCC nodes.
        """
        if not scc_nodes:
            return []
        
        start = scc_nodes[0]
        scc_set = set(scc_nodes)
        
        # BFS to find path from start back to start through SCC
        queue = [(start, [start])]
        visited = {start}
        
        while queue:
            current, path = queue.pop(0)
            
            for neighbor in self.adjacency_list.get(current, []):
                if neighbor == start and len(path) > 1:
                    # Found cycle back to start
                    return path + [start]
                
                if neighbor in scc_set and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        # Couldn't find path (shouldn't happen for valid SCC)
        return scc_nodes + [scc_nodes[0]]
    
    def find_orphan_references(
        self, 
        foreign_keys: Dict[str, str],
        primary_keys: Set[str]
    ) -> List[Tuple[str, str]]:
        """
        Find foreign key values that don't reference valid primary keys.
        
        Args:
            foreign_keys: Dict mapping {record_id: foreign_key_value}
            primary_keys: Set of valid primary key values
        
        Returns:
            List of (record_id, missing_foreign_key) tuples
        """
        orphans = []
        for record_id, fk_value in foreign_keys.items():
            if fk_value and fk_value not in primary_keys:
                orphans.append((record_id, fk_value))
        return orphans
    
    def verify_referential_integrity(
        self,
        table_data: Dict[str, Dict[str, List]]
    ) -> Dict[str, Any]:
        """
        Comprehensive referential integrity verification per Theorem 5.3.
        
        Time Complexity: O(|I| log |I|) with hash table lookups.
        
        Args:
            table_data: Dict mapping {table_name: {
                'primary_keys': [list of PKs],
                'foreign_keys': {fk_column: [list of FK values]}
            }}
        
        Returns:
            Verification results including orphan references and cycles.
        """
        from collections import defaultdict
        
        # Build primary key sets for O(1) lookup (Theorem 5.3)
        pk_sets = {}
        for table, data in table_data.items():
            if 'primary_keys' in data:
                pk_sets[table] = set(data['primary_keys'])
        
        # Check all foreign key references
        orphan_results = defaultdict(list)
        for table, data in table_data.items():
            if 'foreign_keys' not in data:
                continue
            
            for fk_column, fk_values in data['foreign_keys'].items():
                # Determine referenced table from foreign key metadata
                # This should be provided in the data structure
                if 'fk_references' not in data:
                    continue
                
                ref_table = data['fk_references'].get(fk_column)
                if not ref_table or ref_table not in pk_sets:
                    continue
                
                # Check each FK value
                ref_pk_set = pk_sets[ref_table]
                for idx, fk_value in enumerate(fk_values):
                    if fk_value and fk_value not in ref_pk_set:
                        orphan_results[f"{table}.{fk_column}"].append({
                            "row_index": idx,
                            "foreign_key_value": fk_value,
                            "referenced_table": ref_table,
                        })
        
        # Find cycles
        cycle_details = self.get_cycle_details()
        
        return {
            "has_orphans": len(orphan_results) > 0,
            "orphan_count": sum(len(v) for v in orphan_results.values()),
            "orphan_details": dict(orphan_results),
            "has_cycles": len(cycle_details) > 0,
            "cycle_count": len(cycle_details),
            "cycle_details": cycle_details,
            "referential_integrity_satisfied": (
                len(orphan_results) == 0 and len(cycle_details) == 0
            ),
        }
    
    def to_dict(self) -> Dict:
        """Convert graph to dictionary for serialization."""
        return {
            "node_count": self.node_count(),
            "edge_count": self.edge_count(),
            "nodes": {
                node_id: {
                    "entity_type": node.entity_type,
                    "outgoing_edges": node.outgoing_edges,
                    "metadata": node.metadata,
                }
                for node_id, node in self.nodes.items()
            },
        }

