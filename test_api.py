#!/usr/bin/env python3
"""
Test script to demonstrate the recommendation API
"""

import requests
import json
import time

def test_api():
    base_url = "http://localhost:8000"
    
    print("Testing Product Recommendation Engine API")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. Health Check:")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"   Status: {data['status']}")
            print(f"   Users: {data['users']}")
            print(f"   Items: {data['items']}")
            print(f"   Index Size: {data['index_size']}")
        else:
            print(f"   Error: {response.status_code}")
    except Exception as e:
        print(f"   Connection failed: {e}")
        return False
    
    # Test 2: Get recommendations
    print("\n2. Get Recommendations:")
    test_users = [10, 50, 100]
    
    for user_id in test_users:
        try:
            start_time = time.time()
            response = requests.post(
                f"{base_url}/recommend",
                json={"user_id": user_id, "num_recommendations": 5}
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                print(f"   User {user_id}:")
                print(f"     Response time: {data['response_time_ms']:.1f}ms")
                print(f"     Top recommendations:")
                for i, rec in enumerate(data['recommendations'][:3], 1):
                    print(f"       {i}. Item {rec['item_id']} (Score: {rec['score']:.3f})")
            else:
                print(f"   User {user_id}: Error {response.status_code}")
        except Exception as e:
            print(f"   User {user_id}: Request failed: {e}")
    
    # Test 3: User profile
    print("\n3. User Profiles:")
    for user_id in [10, 50]:
        try:
            response = requests.get(f"{base_url}/user/{user_id}/profile")
            if response.status_code == 200:
                data = response.json()
                print(f"   User {user_id}:")
                print(f"     Interactions: {data['interaction_count']}")
                print(f"     Avg Rating: {data['avg_rating']:.2f}")
            else:
                print(f"   User {user_id}: Error {response.status_code}")
        except Exception as e:
            print(f"   User {user_id}: Request failed: {e}")
    
    # Test 4: System stats
    print("\n4. System Statistics:")
    try:
        response = requests.get(f"{base_url}/stats")
        if response.status_code == 200:
            data = response.json()
            print(f"   Total Users: {data['total_users']}")
            print(f"   Total Items: {data['total_items']}")
            print(f"   Total Interactions: {data['total_interactions']}")
            print(f"   Avg Interactions/User: {data['avg_interactions_per_user']:.1f}")
            print(f"   Embedding Dimension: {data['model_embedding_dim']}")
        else:
            print(f"   Error: {response.status_code}")
    except Exception as e:
        print(f"   Request failed: {e}")
    
    # Test 5: Performance benchmark
    print("\n5. Performance Benchmark:")
    try:
        response = requests.get(f"{base_url}/benchmark")
        if response.status_code == 200:
            data = response.json()
            print(f"   Test Users: {data['test_users']}")
            print(f"   Avg Latency: {data['avg_latency_ms']:.1f}ms")
            print(f"   Throughput: {data['throughput_qps']:.0f} QPS")
        else:
            print(f"   Error: {response.status_code}")
    except Exception as e:
        print(f"   Request failed: {e}")
    
    return True

if __name__ == "__main__":
    success = test_api()
    if success:
        print("\n" + "=" * 50)
        print("API TESTS COMPLETED")
    else:
        print("API Server not running. Please start with: python api_server.py")