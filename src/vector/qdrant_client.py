"""Qdrant Vector Database Client for Anime Search

Provides high-performance vector search capabilities optimized for anime data
with advanced filtering, cross-platform ID lookups, and hybrid search.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from qdrant_client import QdrantClient as QdrantSDK
from qdrant_client.models import (
    Distance, VectorParams, CollectionInfo, PointStruct,
    Filter, FieldCondition, MatchValue, MatchAny, Range,
    SearchRequest, CountRequest, UpdateStatus
)
from fastembed import TextEmbedding
import asyncio
import hashlib
from ..config import Settings

logger = logging.getLogger(__name__)


class QdrantClient:
    """Qdrant client wrapper optimized for anime search operations."""
    
    def __init__(self, url: Optional[str] = None, collection_name: Optional[str] = None, settings: Optional[Settings] = None):
        """Initialize Qdrant client with FastEmbed and configuration.
        
        Args:
            url: Qdrant server URL (optional, uses settings if not provided)
            collection_name: Name of the anime collection (optional, uses settings if not provided)
            settings: Configuration settings instance (optional, will import default if not provided)
        """
        # Use provided settings or import default settings
        if settings is None:
            from ..config import get_settings
            settings = get_settings()
        
        self.settings = settings
        self.url = url or settings.qdrant_url
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.client = QdrantSDK(url=self.url)
        self.encoder = None
        self._vector_size = settings.qdrant_vector_size
        self._distance_metric = settings.qdrant_distance_metric
        self._fastembed_model = settings.fastembed_model
        
        # Initialize FastEmbed encoder
        self._init_encoder()
        
        # Create collection if it doesn't exist
        self._ensure_collection_exists()
    
    def _init_encoder(self):
        """Initialize FastEmbed encoder for semantic embeddings."""
        try:
            # Initialize FastEmbed with configured model
            cache_dir = getattr(self.settings, 'fastembed_cache_dir', None)
            init_kwargs = {"model_name": self._fastembed_model}
            if cache_dir:
                init_kwargs["cache_dir"] = cache_dir
                
            self.encoder = TextEmbedding(**init_kwargs)
            logger.info(f"Initialized FastEmbed encoder ({self._fastembed_model}) with vector size: {self._vector_size}")
        except Exception as e:
            logger.error(f"Failed to initialize FastEmbed encoder with model {self._fastembed_model}: {e}")
            raise
    
    def _ensure_collection_exists(self):
        """Create anime collection if it doesn't exist."""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(col.name == self.collection_name for col in collections)
            
            if not collection_exists:
                # Map distance metric string to Qdrant Distance enum
                distance_mapping = {
                    "cosine": Distance.COSINE,
                    "euclid": Distance.EUCLID,
                    "dot": Distance.DOT
                }
                distance = distance_mapping.get(self._distance_metric, Distance.COSINE)
                
                logger.info(f"Creating collection: {self.collection_name} with distance metric: {self._distance_metric}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self._vector_size,
                        distance=distance
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check if Qdrant is healthy and reachable."""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            # Simple health check by getting collections
            collections = await loop.run_in_executor(
                None, 
                lambda: self.client.get_collections()
            )
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            loop = asyncio.get_event_loop()
            
            # Get collection info
            collection_info = await loop.run_in_executor(
                None,
                lambda: self.client.get_collection(self.collection_name)
            )
            
            # Count total points
            count_result = await loop.run_in_executor(
                None,
                lambda: self.client.count(
                    collection_name=self.collection_name,
                    count_filter=None,
                    exact=True
                )
            )
            
            return {
                "collection_name": self.collection_name,
                "total_documents": count_result.count,
                "vector_size": self._vector_size,
                "distance_metric": "cosine",
                "status": collection_info.status,
                "optimizer_status": collection_info.optimizer_status,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "points_count": collection_info.points_count
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}
    
    def _create_embedding(self, text: str) -> List[float]:
        """Create semantic embedding vector from text using FastEmbed."""
        try:
            if not text or not text.strip():
                logger.warning("Empty text provided for embedding")
                # Return zero vector for empty text
                return [0.0] * self._vector_size
            
            # Use FastEmbed to generate semantic embeddings
            embeddings = list(self.encoder.embed([text.strip()]))
            if not embeddings:
                logger.warning(f"No embedding generated for text: {text[:100]}...")
                return [0.0] * self._vector_size
            
            # FastEmbed returns numpy arrays, convert to list
            embedding = embeddings[0].tolist()
            
            # Ensure correct dimensions
            if len(embedding) != self._vector_size:
                logger.warning(f"Embedding size mismatch: got {len(embedding)}, expected {self._vector_size}")
                # Pad or truncate to match expected size
                if len(embedding) < self._vector_size:
                    embedding.extend([0.0] * (self._vector_size - len(embedding)))
                else:
                    embedding = embedding[:self._vector_size]
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to create embedding for text '{text[:100]}...': {e}")
            # Return zero vector on error to prevent pipeline failure
            return [0.0] * self._vector_size
    
    def _generate_point_id(self, anime_id: str) -> str:
        """Generate unique point ID from anime ID."""
        return hashlib.md5(anime_id.encode()).hexdigest()
    
    async def add_documents(self, documents: List[Dict[str, Any]], batch_size: int = 100) -> bool:
        """Add anime documents to the collection.
        
        Args:
            documents: List of anime documents
            batch_size: Number of documents to process per batch
            
        Returns:
            True if successful, False otherwise
        """
        try:
            loop = asyncio.get_event_loop()
            total_docs = len(documents)
            
            logger.info(f"Adding {total_docs} documents in batches of {batch_size}")
            
            for i in range(0, total_docs, batch_size):
                batch = documents[i:i + batch_size]
                points = []
                
                for doc in batch:
                    try:
                        # Create embedding from embedding_text
                        embedding_text = doc.get("embedding_text", "")
                        if not embedding_text:
                            logger.warning(f"Empty embedding_text for anime_id: {doc.get('anime_id')}")
                            continue
                        
                        embedding = self._create_embedding(embedding_text)
                        point_id = self._generate_point_id(doc["anime_id"])
                        
                        # Prepare payload (all fields except embedding_text)
                        payload = {k: v for k, v in doc.items() if k != "embedding_text"}
                        
                        point = PointStruct(
                            id=point_id,
                            vector=embedding,
                            payload=payload
                        )
                        points.append(point)
                        
                    except Exception as e:
                        logger.error(f"Failed to process document {doc.get('anime_id')}: {e}")
                        continue
                
                if points:
                    # Upload batch
                    await loop.run_in_executor(
                        None,
                        lambda: self.client.upsert(
                            collection_name=self.collection_name,
                            points=points
                        )
                    )
                    
                    logger.info(f"Uploaded batch {i//batch_size + 1}/{(total_docs-1)//batch_size + 1} ({len(points)} points)")
            
            logger.info(f"Successfully added {total_docs} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False
    
    async def search(self, query: str, limit: int = 20, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Perform semantic search on anime collection.
        
        Args:
            query: Search query text
            limit: Maximum number of results
            filters: Optional filters for metadata
            
        Returns:
            List of search results with anime data and scores
        """
        try:
            # Create query embedding
            query_embedding = self._create_embedding(query)
            
            # Build filter if provided
            qdrant_filter = None
            if filters:
                qdrant_filter = self._build_filter(filters)
            
            loop = asyncio.get_event_loop()
            
            # Perform search
            search_result = await loop.run_in_executor(
                None,
                lambda: self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    query_filter=qdrant_filter,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False
                )
            )
            
            # Format results
            results = []
            for hit in search_result:
                result = dict(hit.payload)
                result["_score"] = hit.score
                result["_id"] = hit.id
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def get_similar_anime(self, anime_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find similar anime based on vector similarity.
        
        Args:
            anime_id: ID of the reference anime
            limit: Maximum number of similar anime to return
            
        Returns:
            List of similar anime with similarity scores
        """
        try:
            # Find the reference anime
            point_id = self._generate_point_id(anime_id)
            
            loop = asyncio.get_event_loop()
            
            # Get the reference point
            reference_point = await loop.run_in_executor(
                None,
                lambda: self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=[point_id],
                    with_vectors=True
                )
            )
            
            if not reference_point:
                logger.warning(f"Anime not found: {anime_id}")
                return []
            
            # Use the reference vector to find similar anime
            reference_vector = reference_point[0].vector
            
            # Search excluding the reference anime itself
            filter_out_self = Filter(
                must_not=[
                    FieldCondition(
                        key="anime_id",
                        match=MatchValue(value=anime_id)
                    )
                ]
            )
            
            search_result = await loop.run_in_executor(
                None,
                lambda: self.client.search(
                    collection_name=self.collection_name,
                    query_vector=reference_vector,
                    query_filter=filter_out_self,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False
                )
            )
            
            # Format results
            results = []
            for hit in search_result:
                result = dict(hit.payload)
                result["similarity_score"] = hit.score
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Similar anime search failed: {e}")
            return []
    
    def _build_filter(self, filters: Dict) -> Filter:
        """Build Qdrant filter from filter dictionary.
        
        Args:
            filters: Dictionary with filter conditions
            
        Returns:
            Qdrant Filter object
        """
        conditions = []
        
        for key, value in filters.items():
            if isinstance(value, dict):
                # Range filter
                if "gte" in value or "lte" in value or "gt" in value or "lt" in value:
                    conditions.append(
                        FieldCondition(
                            key=key,
                            range=Range(**value)
                        )
                    )
                # Match any filter
                elif "any" in value:
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchAny(any=value["any"])
                        )
                    )
            elif isinstance(value, list):
                # Match any from list
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchAny(any=value)
                    )
                )
            else:
                # Exact match
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
        
        return Filter(must=conditions) if conditions else None
    
    async def get_by_id(self, anime_id: str) -> Optional[Dict[str, Any]]:
        """Get anime by ID.
        
        Args:
            anime_id: The anime ID to retrieve
            
        Returns:
            Anime data dictionary or None if not found
        """
        try:
            loop = asyncio.get_event_loop()
            point_id = self._generate_point_id(anime_id)
            
            # Retrieve point by ID
            points = await loop.run_in_executor(
                None,
                lambda: self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=[point_id],
                    with_payload=True,
                    with_vectors=False
                )
            )
            
            if points:
                return dict(points[0].payload)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get anime by ID {anime_id}: {e}")
            return None
    
    async def find_similar(self, anime_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find similar anime using vector similarity.
        
        Args:
            anime_id: Reference anime ID
            limit: Number of similar anime to return
            
        Returns:
            List of similar anime with similarity scores
        """
        try:
            # First get the reference anime
            reference_anime = await self.get_by_id(anime_id)
            if not reference_anime:
                logger.warning(f"Reference anime not found: {anime_id}")
                return []
            
            # Use the title and tags for similarity search
            search_text = reference_anime.get("title", "")
            if reference_anime.get("tags"):
                search_text += " " + " ".join(reference_anime["tags"][:5])  # Limit tags
            
            # Perform similarity search
            loop = asyncio.get_event_loop()
            embedding = self._create_embedding(search_text)
            point_id = self._generate_point_id(anime_id)
            
            # Filter to exclude the reference anime itself
            filter_out_self = Filter(
                must_not=[
                    FieldCondition(
                        key="anime_id",
                        match=MatchValue(value=anime_id)
                    )
                ]
            )
            
            search_result = await loop.run_in_executor(
                None,
                lambda: self.client.search(
                    collection_name=self.collection_name,
                    query_vector=embedding,
                    query_filter=filter_out_self,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False
                )
            )
            
            # Format results
            results = []
            for hit in search_result:
                result = dict(hit.payload)
                result["similarity_score"] = hit.score
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to find similar anime for {anime_id}: {e}")
            return []
    
    async def clear_index(self) -> bool:
        """Clear all points from the collection (for fresh re-indexing)."""
        try:
            # Delete and recreate collection for clean state
            await self.delete_collection()
            await self.create_collection()
            logger.info(f"Cleared and recreated collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear index: {e}")
            return False
    
    async def delete_collection(self) -> bool:
        """Delete the anime collection (for testing/reset)."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.client.delete_collection(self.collection_name)
            )
            logger.info(f"Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False
    
    async def create_collection(self) -> bool:
        """Create the anime collection."""
        try:
            self._ensure_collection_exists()
            return True
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False