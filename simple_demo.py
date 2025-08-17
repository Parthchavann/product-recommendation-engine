#!/usr/bin/env python3
"""
Simple demo script for the Product Recommendation Engine
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timedelta
import time
import sys
from pathlib import Path

print("*** Product Recommendation Engine - Demo ***")
print("=" * 50)

# Create synthetic data
print("\n1. Creating synthetic dataset...")
np.random.seed(42)
torch.manual_seed(42)

n_users = 1000
n_items = 500
n_interactions = 5000

# Generate interactions
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
print(f"Created {n_users} users, {n_items} items, {len(interactions_df)} interactions")

# Simple Two-Tower Model
print("\n2. Building Two-Tower Model...")

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

model = SimpleTwoTower(n_users, n_items)
print("Two-Tower model initialized")

# Quick training
print("\n3. Training model...")
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
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/20, Loss: {loss.item():.4f}")

print("Model training completed")

# Build FAISS index
print("\n4. Building FAISS index...")
import faiss

model.eval()
with torch.no_grad():
    all_item_ids = torch.arange(n_items, dtype=torch.long)
    all_user_ids = torch.arange(n_users, dtype=torch.long)
    
    item_embeddings = model.item_layers(model.item_embedding(all_item_ids))
    item_embeddings = F.normalize(item_embeddings, p=2, dim=1).numpy()

# Build FAISS index
embedding_dim = item_embeddings.shape[1]
index = faiss.IndexFlatIP(embedding_dim)
index.add(item_embeddings.astype('float32'))

print(f"FAISS index built with {index.ntotal} item embeddings")

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

print("\n5. Testing recommendations...")

# Test recommendations for a few users
for user_id in [10, 50, 100]:
    print(f"\nUser {user_id} recommendations:")
    start_time = time.time()
    recs = get_recommendations(user_id, 5)
    end_time = time.time()
    
    for i, rec in enumerate(recs, 1):
        print(f"  {i}. Item {rec['item_id']} (Score: {rec['score']:.3f})")
    
    print(f"  Latency: {(end_time - start_time)*1000:.1f}ms")

# Performance benchmark
print("\n6. Performance benchmark...")
test_users = np.random.choice(n_users, 100, replace=False)
start_time = time.time()

for user_id in test_users:
    _ = get_recommendations(user_id, 10)

end_time = time.time()
avg_latency = (end_time - start_time) / len(test_users) * 1000

print(f"Average latency: {avg_latency:.1f}ms")
print(f"Throughput: ~{1000/avg_latency:.0f} QPS")

print("\n" + "=" * 50)
print("DEMO COMPLETED SUCCESSFULLY!")
print("=" * 50)
print(f"- Users: {n_users:,}")
print(f"- Items: {n_items:,}")
print(f"- Interactions: {len(interactions_df):,}")
print(f"- Avg Latency: {avg_latency:.1f}ms")
print(f"- Throughput: ~{1000/avg_latency:.0f} QPS")
print("\nThe recommendation system is working!")