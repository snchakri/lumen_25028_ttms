<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Stage 3 Data Compilation Onboarding Guide for Cursor IDE

This document provides a comprehensive overview of Stage 3, detailing every module, implementation changes, theoretical foundations, and integration notes. It ensures you understand the codebase structure, design decisions, and mathematical guarantees.

***

## Table of Contents

1. Overview \& Architectural Context
2. Directory Structure
3. Module Descriptions
4. Key Theoretical Foundations \& Theorems
5. Recent Fixes \& Mock Removal
6. Integration \& Usage Guide
7. References \& Further Reading

***

## 1. Overview \& Architectural Context

Stage 3 (“Data Compilation”) transforms validated inputs (Stage 1) and batch outputs (Stage 2) into a universal, solver-agnostic data foundation for Stages 4–7. It adheres to:

- **Single-threaded compile-and-forward** model
- **O(N log² N)** time, **O(N log N)** space
- **512 MB RAM** peak constraint
- Rigorous **mathematical guarantees** (information preservation, query completeness, normalization, relationship completeness, index access bounds)
- Strict modularity: four primary layers + support modules

***

## 2. Directory Structure

```
stage_3/
├── __init__.py
├── data_normalizer/         # Layer 1
│   ├── __init__.py
│   ├── csv_ingestor.py
│   ├── schema_validator.py
│   ├── dependency_validator.py
│   ├── redundancy_eliminator.py
│   ├── checkpoint_manager.py
│   └── normalization_engine.py
├── relationship_engine.py    # Layer 2
├── index_builder.py          # Layer 3
├── optimization_views.py     # Layer 4
├── memory_optimizer.py       # Support: cache-oblivious design
├── validation_engine.py      # Support: transitional theorem checks
├── compilation_engine.py     # Orchestrator: invokes all layers
├── performance_monitor.py    # Support: runtime profiling & enforcement
├── storage_manager.py        # Support: file persistence & manifests
└── api_interface.py          # Support: FastAPI endpoints & WebSocket
```


***

## 3. Module Descriptions

### Layer 1: data_normalizer/

- **csv_ingestor.py**
Loads validated CSVs; detects dialect statistically; computes SHA-256 checksums; chunked reads; strict I/O error handling.
- **schema_validator.py**
Uses Pydantic models auto-generated from `hei_timetabling_datamodel.sql`; enforces types, nullability, ENUMs, cross-table referential integrity.
- **dependency_validator.py**
Implements BCNF decomposition (Theorem 3.3) with lossless-join tests; loads functional dependencies; verifies preservation.
- **redundancy_eliminator.py**
Detects exact and near-duplicates via hash grouping, Jaccard/Levenshtein measures; preserves multiplicity; guarantees information preservation (Theorem 5.1).
- **checkpoint_manager.py**
Creates atomic checkpoints after each sub-step; stores checksums and row counts; rollback on failure; lightweight validation of state transitions.
- **normalization_engine.py**
Orchestrates Layer 1 end-to-end; records performance/memory metrics; integrates all above modules sequentially.


### Layer 2: relationship_engine.py

Discovers PK–FK edges syntactically; infers semantic links via cosine similarity; computes statistical correlations; runs Floyd-Warshall for transitive closure (Theorem 3.6 completeness ≥ 99.4%).

### Layer 3: index_builder.py

Builds HashIndex (O(1) expected), BTreeIndex (O(log N + k)), GraphIndex (O(d)), BitmapIndex (O(N/word_size)); verifies Theorem 3.9 bounds and memory limits.

### Layer 4: optimization_views.py

Assembles entities, relationships, and indices into a `CompiledDataStructure`; exposes solver-agnostic API; enforces information preservation (Theorem 5.1) and query completeness (Theorem 5.2).

### Support Modules

- **memory_optimizer.py**: Implements cache-oblivious layouts, real-time 512 MB monitoring, memory-mapped files (Theorems 4.2 \& 4.4).
- **validation_engine.py**: Runs lightweight theorem checks (entropy comparison, representative query tests).
- **compilation_engine.py**: High-level pipeline driver calling Layers 1–4 in sequence with checkpoint rollback.
- **performance_monitor.py**: Profiles CPU, memory; logs complexity deviations.
- **storage_manager.py**: Persists outputs into `Lraw.parquet`, `Lrel.graphml`, `Lidx.*`, `manifest.json`; atomic writes and metadata.
- **api_interface.py**: FastAPI service exposing `/compile`, `/status`, `/download`; WebSockets for progress; authentication middleware.

***

## 4. Key Theoretical Foundations \& Theorems

- **Normalization Theorem 3.3**: Lossless BCNF \& dependency preservation
- **Information Preservation (Theorem 5.1)**: $I_{compiled} ≥ I_{source} - R + I_{relationships}$
- **Query Completeness (Theorem 5.2)**: All CSV queries remain answerable in ≤ O(log N)
- **Relationship Discovery (Theorem 3.6)**: P(found ⊇ true) ≥ 0.994
- **Index Access (Theorem 3.9)**: Expected O(1) point, O(log N + k) range queries
- **Cache-Oblivious (Theorems 4.2 \& 4.4)**: Miss rate ≤ 1/B, optimal block transfers

***

## 5. Recent Fixes \& Mock Removal

- **All mock functions eliminated**: No placeholder returns, no `asyncio.sleep` simulations, no fake data.
- **Algorithms completed**: B-tree splits, Floyd-Warshall, BCNF join tests, Shannon entropy, statistical correlation.
- **MockCompilationEngine** and fake memory managers removed.
- **Production-grade error handling and structured logging** added across modules.

Refer to **STAGE3_COMPREHENSIVE_ISSUES_REPORT.md** for prior inconsistencies and how each was addressed.

***

## 6. Integration \& Usage Guide

1. **Install Dependencies**

```bash
pip install pandas numpy scipy networkx pydantic fastapi uvicorn python-json-logger
```

2. **Verify SQL Schema**
Ensure `hei_timetabling_datamodel.sql` is up to date; regenerate Pydantic models if modified.
3. **Run Compilation via API**

```bash
uvicorn stage_3.api_interface:app --reload
curl -X POST http://localhost:8000/compile -F "input_dir=path/to/csvs" -F "output_dir=path/to/compiled"
```

4. **Monitor Progress**
Connect to WebSocket at `/ws/status` for real-time updates.
5. **Load Compiled Data**
Use `storage_manager.load_compiled(output_dir)` to get a `CompiledDataStructure` instance.
6. **Error Recovery**
On failure, check `checkpoints/` to resume from last valid state.

***

## 7. References \& Further Reading

- Stage 3 PDF: **Stage-3 DATA COMPILATION – Theoretical Foundations \& Mathematical Framework**
- Stage 1 \& 2 PDFs for validation and batching context
- Theorems and algorithms cited inline within each module’s docstrings
- **STAGE3_COMPREHENSIVE_ISSUES_REPORT.md** for historical bugs and fixes

***

*Cursor IDE should use this document to navigate the Stage 3 codebase, understand design rationales, and ensure all modules align with the project’s rigorous mathematical and production requirements.*

