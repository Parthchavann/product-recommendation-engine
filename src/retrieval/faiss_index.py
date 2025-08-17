import faiss
import numpy as np
import pickle
import os
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
import time

from ..utils.logger import get_logger


logger = get_logger(__name__)


class FAISSIndex:
    """FAISS index for fast similarity search"""
    
    def __init__(self, 
                 dimension: int,
                 index_type: str = "IVF",
                 metric: str = "inner_product",
                 nlist: int = 100,
                 nprobe: int = 10):
        """
        Initialize FAISS index
        
        Args:
            dimension: Vector dimension
            index_type: Type of index (IVF, HNSW, Flat)
            metric: Distance metric (inner_product, L2)  
            nlist: Number of Voronoi cells (for IVF)
            nprobe: Number of cells to search (for IVF)
        """
        
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.nlist = nlist
        self.nprobe = nprobe
        
        self.index = None
        self.id_map = {}  # Map from FAISS internal id to actual item id
        self.reverse_id_map = {}  # Map from item id to FAISS internal id
        self.is_trained = False
        
        logger.info(f"Initialized FAISS index: {index_type}, dimension={dimension}, metric={metric}")
    
    def _create_index(self) -> faiss.Index:
        """Create appropriate FAISS index based on configuration"""
        
        if self.metric == "inner_product":
            metric_type = faiss.METRIC_INNER_PRODUCT
        elif self.metric == "L2":
            metric_type = faiss.METRIC_L2
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        
        if self.index_type == "Flat":
            # Exact search (brute force)
            if self.metric == "inner_product":
                index = faiss.IndexFlatIP(self.dimension)
            else:
                index = faiss.IndexFlatL2(self.dimension)
            
        elif self.index_type == "IVF":
            # IVF (Inverted File) index for approximate search
            quantizer = faiss.IndexFlatIP(self.dimension) if self.metric == "inner_product" else faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, metric_type)
            
        elif self.index_type == "HNSW":
            # Hierarchical Navigable Small World
            index = faiss.IndexHNSWFlat(self.dimension, 32, metric_type)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 100
            
        elif self.index_type == "PQ":
            # Product Quantization for memory efficiency
            m = min(8, self.dimension // 8)  # Number of subquantizers
            index = faiss.IndexPQ(self.dimension, m, 8, metric_type)
            
        elif self.index_type == "IVF_PQ":
            # Combination of IVF and PQ
            quantizer = faiss.IndexFlatIP(self.dimension) if self.metric == "inner_product" else faiss.IndexFlatL2(self.dimension)
            m = min(8, self.dimension // 8)
            index = faiss.IndexIVFPQ(quantizer, self.dimension, self.nlist, m, 8, metric_type)
            
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        return index
    
    def build_index(self, 
                   embeddings: np.ndarray,
                   item_ids: List[int],
                   train_size: Optional[int] = None):
        """
        Build FAISS index from embeddings
        
        Args:
            embeddings: Embedding vectors [num_items, dimension]
            item_ids: Corresponding item IDs
            train_size: Number of vectors to use for training (for IVF indices)
        """
        
        if len(embeddings) != len(item_ids):
            raise ValueError("Number of embeddings and item_ids must match")
        
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} doesn't match index dimension {self.dimension}")
        
        logger.info(f"Building FAISS index with {len(embeddings)} vectors...")
        
        # Ensure embeddings are float32 (FAISS requirement)
        embeddings = embeddings.astype('float32')
        
        # Normalize embeddings for inner product search
        if self.metric == "inner_product":
            faiss.normalize_L2(embeddings)
        
        # Create index
        self.index = self._create_index()
        
        # Train index if needed
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            logger.info("Training index...")
            
            if train_size and len(embeddings) > train_size:
                # Use random subset for training
                train_indices = np.random.choice(len(embeddings), train_size, replace=False)
                train_vectors = embeddings[train_indices]
            else:
                train_vectors = embeddings
            
            start_time = time.time()
            self.index.train(train_vectors)
            train_time = time.time() - start_time
            
            logger.info(f"Training completed in {train_time:.2f}s")
        
        # Add vectors to index
        logger.info("Adding vectors to index...")
        start_time = time.time()
        self.index.add(embeddings)
        add_time = time.time() - start_time
        
        logger.info(f"Added vectors in {add_time:.2f}s")
        
        # Set search parameters for IVF indices
        if self.index_type in ["IVF", "IVF_PQ"]:
            self.index.nprobe = self.nprobe
        
        # Create ID mappings
        self.id_map = {i: item_id for i, item_id in enumerate(item_ids)}
        self.reverse_id_map = {item_id: i for i, item_id in enumerate(item_ids)}
        
        self.is_trained = True
        
        logger.info(f"Index built successfully! Total vectors: {self.index.ntotal}")
        
        # Log index statistics
        if hasattr(self.index, 'nlist'):
            logger.info(f"Number of clusters: {self.index.nlist}")
        if hasattr(self.index, 'nprobe'):
            logger.info(f"Search probes: {self.index.nprobe}")
    
    def search(self, 
              query_vectors: np.ndarray,
              k: int = 10,
              nprobe: Optional[int] = None) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Search for k nearest neighbors
        
        Args:
            query_vectors: Query vectors [num_queries, dimension] 
            k: Number of neighbors to return
            nprobe: Number of probes for IVF search (optional override)
            
        Returns:
            Tuple of (item_ids_list, distances_list)
        """
        
        if not self.is_trained:
            raise ValueError("Index not trained. Call build_index() first.")
        
        # Ensure correct dtype and shape
        query_vectors = np.atleast_2d(query_vectors).astype('float32')
        
        if query_vectors.shape[1] != self.dimension:
            raise ValueError(f"Query dimension {query_vectors.shape[1]} doesn't match index dimension {self.dimension}")
        
        # Normalize query vectors for inner product search
        if self.metric == "inner_product":
            faiss.normalize_L2(query_vectors)
        
        # Set nprobe if specified
        if nprobe and hasattr(self.index, 'nprobe'):
            original_nprobe = self.index.nprobe
            self.index.nprobe = nprobe
        
        # Search
        start_time = time.time()
        distances, indices = self.index.search(query_vectors, k)
        search_time = time.time() - start_time
        
        logger.debug(f"Search completed in {search_time*1000:.2f}ms for {len(query_vectors)} queries")
        
        # Restore original nprobe
        if nprobe and hasattr(self.index, 'nprobe'):
            self.index.nprobe = original_nprobe
        
        # Map indices back to item IDs
        item_ids_list = []
        distances_list = []
        
        for i in range(len(query_vectors)):
            query_item_ids = []
            query_distances = []
            
            for j in range(k):
                idx = indices[i][j]
                dist = distances[i][j]
                
                # Check if valid index
                if idx >= 0 and idx in self.id_map:
                    query_item_ids.append(self.id_map[idx])
                    query_distances.append(float(dist))
            
            item_ids_list.append(query_item_ids)
            distances_list.append(query_distances)
        
        return item_ids_list, distances_list
    
    def add_items(self, 
                 new_embeddings: np.ndarray,
                 new_item_ids: List[int]):
        """
        Add new items to existing index
        
        Args:
            new_embeddings: New embedding vectors
            new_item_ids: Corresponding new item IDs
        """
        
        if not self.is_trained:
            raise ValueError("Index not trained. Call build_index() first.")
        
        # Ensure correct dtype
        new_embeddings = new_embeddings.astype('float32')
        
        # Normalize if needed
        if self.metric == "inner_product":
            faiss.normalize_L2(new_embeddings)
        
        # Add to index
        current_size = self.index.ntotal
        self.index.add(new_embeddings)
        
        # Update ID mappings
        for i, item_id in enumerate(new_item_ids):
            faiss_id = current_size + i
            self.id_map[faiss_id] = item_id
            self.reverse_id_map[item_id] = faiss_id
        
        logger.info(f"Added {len(new_item_ids)} new items. Total: {self.index.ntotal}")
    
    def remove_items(self, item_ids: List[int], embedding_store: Optional[Dict[int, np.ndarray]] = None):
        """
        Remove items from index (creates new index without specified items)
        
        Args:
            item_ids: Item IDs to remove
            embedding_store: Optional dict mapping item_id -> embedding for reconstruction
        """
        
        logger.warning("FAISS doesn't support efficient removal. Rebuilding index...")
        
        if not embedding_store:
            logger.warning(
                "No embedding store provided. Cannot efficiently remove items. "
                "Consider rebuilding index from scratch or providing embedding_store parameter."
            )
            raise ValueError(
                "FAISS doesn't support efficient item removal without original embeddings. "
                "Please provide embedding_store parameter or rebuild from scratch."
            )
        
        # Get remaining items
        remaining_item_ids = []
        remaining_embeddings = []
        
        for item_id in self.reverse_id_map.keys():
            if item_id not in item_ids and item_id in embedding_store:
                remaining_item_ids.append(item_id)
                remaining_embeddings.append(embedding_store[item_id])
        
        if not remaining_item_ids:
            logger.warning("No items remaining after removal")
            self.index = None
            self.id_map = {}
            self.reverse_id_map = {}
            self.is_trained = False
            return
        
        # Convert to numpy array
        remaining_embeddings = np.array(remaining_embeddings)
        
        logger.info(f"Rebuilding index with {len(remaining_item_ids)} items (removed {len(item_ids)} items)")
        
        # Rebuild index
        self.build_index(remaining_embeddings, remaining_item_ids)
        
        logger.info(f"Successfully removed {len(item_ids)} items. New index size: {self.index.ntotal}")
    
    def save(self, path: str):
        """Save index and metadata to disk"""
        
        if not self.is_trained:
            raise ValueError("Cannot save untrained index")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{path}.index")
        
        # Save metadata
        metadata = {
            'dimension': self.dimension,
            'index_type': self.index_type,
            'metric': self.metric,
            'nlist': self.nlist,
            'nprobe': self.nprobe,
            'id_map': self.id_map,
            'reverse_id_map': self.reverse_id_map,
            'is_trained': self.is_trained
        }
        
        with open(f"{path}.metadata", 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Saved FAISS index to {path}")
    
    def load(self, path: str):
        """Load index and metadata from disk"""
        
        path = Path(path)
        
        if not path.with_suffix('.index').exists():
            raise FileNotFoundError(f"Index file not found: {path}.index")
        
        if not path.with_suffix('.metadata').exists():
            raise FileNotFoundError(f"Metadata file not found: {path}.metadata")
        
        # Load FAISS index
        self.index = faiss.read_index(f"{path}.index")
        
        # Load metadata
        with open(f"{path}.metadata", 'rb') as f:
            metadata = pickle.load(f)
        
        self.dimension = metadata['dimension']
        self.index_type = metadata['index_type']
        self.metric = metadata['metric']
        self.nlist = metadata['nlist']
        self.nprobe = metadata['nprobe']
        self.id_map = metadata['id_map']
        self.reverse_id_map = metadata['reverse_id_map']
        self.is_trained = metadata['is_trained']
        
        # Set search parameters
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.nprobe
        
        logger.info(f"Loaded FAISS index from {path}")
        logger.info(f"Index contains {self.index.ntotal} vectors")
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        
        stats = {
            'dimension': self.dimension,
            'index_type': self.index_type,
            'metric': self.metric,
            'is_trained': self.is_trained,
            'num_vectors': self.index.ntotal if self.index else 0,
            'memory_usage_mb': self.get_memory_usage() / (1024 * 1024)
        }
        
        if hasattr(self.index, 'nlist'):
            stats['nlist'] = self.index.nlist
        
        if hasattr(self.index, 'nprobe'):
            stats['nprobe'] = self.index.nprobe
        
        return stats
    
    def get_memory_usage(self) -> int:
        """Estimate memory usage in bytes"""
        
        if not self.index:
            return 0
        
        # Base calculation: embeddings + overhead
        base_memory = self.index.ntotal * self.dimension * 4  # float32
        
        # Add index-specific overhead
        if self.index_type == "IVF":
            # IVF overhead: centroids + posting lists
            overhead = self.nlist * self.dimension * 4
        elif self.index_type == "HNSW":
            # HNSW overhead: graph structure
            overhead = self.index.ntotal * 32 * 4  # Rough estimate
        else:
            overhead = 0
        
        return base_memory + overhead
    
    def benchmark_search(self, 
                        query_vectors: np.ndarray,
                        k_values: List[int] = [10, 50, 100],
                        nprobe_values: Optional[List[int]] = None) -> Dict:
        """
        Benchmark search performance
        
        Args:
            query_vectors: Test query vectors
            k_values: List of k values to test
            nprobe_values: List of nprobe values to test (for IVF)
            
        Returns:
            Benchmark results
        """
        
        if not self.is_trained:
            raise ValueError("Index not trained")
        
        results = {}
        
        # Default nprobe values for IVF
        if nprobe_values is None:
            if hasattr(self.index, 'nprobe'):
                nprobe_values = [1, 5, 10, 20, 50]
            else:
                nprobe_values = [None]
        
        logger.info(f"Benchmarking with {len(query_vectors)} queries...")
        
        for k in k_values:
            for nprobe in nprobe_values:
                # Time multiple runs
                times = []
                for _ in range(3):
                    start_time = time.time()
                    self.search(query_vectors, k=k, nprobe=nprobe)
                    times.append(time.time() - start_time)
                
                avg_time = np.mean(times)
                qps = len(query_vectors) / avg_time
                
                key = f"k={k}"
                if nprobe is not None:
                    key += f",nprobe={nprobe}"
                
                results[key] = {
                    'avg_time_ms': avg_time * 1000,
                    'qps': qps,
                    'latency_per_query_ms': (avg_time * 1000) / len(query_vectors)
                }
        
        return results


class CandidateGenerator:
    """High-level candidate generation using FAISS"""
    
    def __init__(self, faiss_index: FAISSIndex):
        self.index = faiss_index
        self.logger = get_logger(__name__)
    
    def get_candidates(self,
                      user_embedding: np.ndarray,
                      k: int = 100,
                      filters: Optional[Dict] = None) -> List[Dict]:
        """
        Get candidate items for a user
        
        Args:
            user_embedding: User embedding vector
            k: Number of candidates to retrieve
            filters: Additional filters (category, popularity, etc.)
            
        Returns:
            List of candidate items with metadata
        """
        
        # Search for similar items
        item_ids_list, scores_list = self.index.search(
            user_embedding.reshape(1, -1), k=k
        )
        
        candidates = []
        
        if item_ids_list and scores_list:
            item_ids = item_ids_list[0]
            scores = scores_list[0]
            
            for item_id, score in zip(item_ids, scores):
                candidate = {
                    'item_id': item_id,
                    'similarity_score': score,
                    'source': 'collaborative_filtering'
                }
                candidates.append(candidate)
        
        # Apply filters if specified
        if filters:
            candidates = self._apply_filters(candidates, filters)
        
        return candidates
    
    def _apply_filters(self, candidates: List[Dict], filters: Dict) -> List[Dict]:
        """Apply filtering logic to candidates"""
        
        # This is a placeholder - implement based on your specific requirements
        filtered = candidates
        
        if 'min_score' in filters:
            filtered = [c for c in filtered if c['similarity_score'] >= filters['min_score']]
        
        if 'exclude_items' in filters:
            exclude_set = set(filters['exclude_items'])
            filtered = [c for c in filtered if c['item_id'] not in exclude_set]
        
        return filtered
    
    def get_batch_candidates(self,
                           user_embeddings: np.ndarray,
                           k: int = 100) -> List[List[Dict]]:
        """
        Get candidates for multiple users in batch
        
        Args:
            user_embeddings: User embedding vectors [num_users, dim]
            k: Number of candidates per user
            
        Returns:
            List of candidate lists (one per user)
        """
        
        # Batch search
        item_ids_lists, scores_lists = self.index.search(user_embeddings, k=k)
        
        all_candidates = []
        
        for item_ids, scores in zip(item_ids_lists, scores_lists):
            candidates = []
            
            for item_id, score in zip(item_ids, scores):
                candidate = {
                    'item_id': item_id,
                    'similarity_score': score,
                    'source': 'collaborative_filtering'
                }
                candidates.append(candidate)
            
            all_candidates.append(candidates)
        
        return all_candidates