#!/usr/bin/env python3
"""
FastAPI server for the Product Recommendation Engine
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import time
import uvicorn

# Initialize the recommendation system
print("Initializing recommendation system...")

# Create synthetic data (same as demo)
np.random.seed(42)
torch.manual_seed(42)

n_users = 1000
n_items = 500
n_interactions = 5000

interactions = []
for _ in range(n_interactions):
    user_id = np.random.randint(0, n_users)
    item_id = np.random.randint(0, n_items)
    rating = np.random.uniform(1, 5)
    interactions.append({
        'user_id': user_id,
        'item_id': item_id,
        'rating': rating
    })

interactions_df = pd.DataFrame(interactions)

# Two-Tower Model (same as demo)
class SimpleTwoTower(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=64):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        self.user_layers = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
        
        self.item_layers = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
    
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        user_repr = self.user_layers(user_emb)
        item_repr = self.item_layers(item_emb)
        
        user_repr = F.normalize(user_repr, p=2, dim=1)
        item_repr = F.normalize(item_repr, p=2, dim=1)
        
        return user_repr, item_repr

# Train model
model = SimpleTwoTower(n_users, n_items)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

train_users = torch.tensor(interactions_df['user_id'].values, dtype=torch.long)
train_items = torch.tensor(interactions_df['item_id'].values, dtype=torch.long)
train_ratings = torch.tensor(interactions_df['rating'].values, dtype=torch.float32) / 5.0

model.train()
for epoch in range(20):
    optimizer.zero_grad()
    user_repr, item_repr = model(train_users, train_items)
    scores = torch.sum(user_repr * item_repr, dim=1)
    loss = F.mse_loss(scores, train_ratings)
    loss.backward()
    optimizer.step()

# Build FAISS index
model.eval()
with torch.no_grad():
    all_item_ids = torch.arange(n_items, dtype=torch.long)
    item_embeddings = model.item_layers(model.item_embedding(all_item_ids))
    item_embeddings = F.normalize(item_embeddings, p=2, dim=1).numpy()

embedding_dim = item_embeddings.shape[1]
index = faiss.IndexFlatIP(embedding_dim)
index.add(item_embeddings.astype('float32'))

# Recommendation function
def get_recommendations(user_id, num_recommendations=10):
    model.eval()
    with torch.no_grad():
        user_tensor = torch.tensor([user_id], dtype=torch.long)
        user_emb = model.user_layers(model.user_embedding(user_tensor))
        user_emb = F.normalize(user_emb, p=2, dim=1).numpy()
        
        scores, item_indices = index.search(user_emb.astype('float32'), num_recommendations)
        
        recommendations = []
        for score, item_idx in zip(scores[0], item_indices[0]):
            recommendations.append({
                'item_id': int(item_idx),
                'score': float(score)
            })
        
        return recommendations

print(f"Model trained and FAISS index built with {index.ntotal} items")

# FastAPI app
app = FastAPI(
    title="Product Recommendation Engine API",
    description="A Netflix/YouTube-style recommendation system using two-tower neural networks and FAISS",
    version="1.0.0"
)

# Pydantic models
class RecommendationRequest(BaseModel):
    user_id: int
    num_recommendations: Optional[int] = 10

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[dict]
    response_time_ms: float

@app.get("/")
async def root():
    return {
        "message": "Product Recommendation Engine API",
        "status": "running",
        "users": n_users,
        "items": n_items,
        "interactions": len(interactions_df)
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "index_size": index.ntotal,
        "users": n_users,
        "items": n_items
    }

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    start_time = time.time()
    
    # Validate user_id
    if request.user_id >= n_users or request.user_id < 0:
        raise HTTPException(status_code=400, detail=f"Invalid user_id. Must be between 0 and {n_users-1}")
    
    # Get recommendations
    recommendations = get_recommendations(request.user_id, request.num_recommendations)
    
    end_time = time.time()
    response_time_ms = (end_time - start_time) * 1000
    
    return RecommendationResponse(
        user_id=request.user_id,
        recommendations=recommendations,
        response_time_ms=response_time_ms
    )

@app.get("/user/{user_id}/profile")
async def user_profile(user_id: int):
    if user_id >= n_users or user_id < 0:
        raise HTTPException(status_code=400, detail=f"Invalid user_id. Must be between 0 and {n_users-1}")
    
    # Get user interactions
    user_history = interactions_df[interactions_df['user_id'] == user_id]
    
    return {
        "user_id": user_id,
        "interaction_count": len(user_history),
        "avg_rating": float(user_history['rating'].mean()) if len(user_history) > 0 else 0.0,
        "items_interacted": user_history['item_id'].tolist() if len(user_history) > 0 else []
    }

@app.get("/stats")
async def stats():
    return {
        "total_users": n_users,
        "total_items": n_items,
        "total_interactions": len(interactions_df),
        "avg_interactions_per_user": len(interactions_df) / n_users,
        "model_embedding_dim": embedding_dim,
        "faiss_index_size": index.ntotal
    }

@app.get("/benchmark")
async def benchmark():
    """Run a quick performance benchmark"""
    test_users = np.random.choice(n_users, 100, replace=False)
    start_time = time.time()
    
    for user_id in test_users:
        _ = get_recommendations(user_id, 10)
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_latency_ms = total_time / len(test_users) * 1000
    throughput_qps = len(test_users) / total_time
    
    return {
        "test_users": len(test_users),
        "total_time_seconds": total_time,
        "avg_latency_ms": avg_latency_ms,
        "throughput_qps": throughput_qps
    }

if __name__ == "__main__":
    print("\nStarting FastAPI server...")
    print("API will be available at: http://localhost:8000")
    print("Interactive docs at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)