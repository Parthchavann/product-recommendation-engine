import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import random

from ..utils.logger import get_logger


logger = get_logger(__name__)


class ClickstreamSimulator:
    """Simulate realistic user clickstream data for recommendation system"""
    
    def __init__(self, 
                 users_df: pd.DataFrame,
                 items_df: pd.DataFrame,
                 seed: int = 42):
        
        self.users_df = users_df
        self.items_df = items_df
        self.rng = np.random.RandomState(seed)
        
        # User behavior profiles
        self.user_profiles = self._create_user_profiles()
        
        # Item popularity distribution
        self.item_popularity = self._create_item_popularity()
    
    def _create_user_profiles(self) -> Dict[int, Dict]:
        """Create user behavior profiles"""
        
        profiles = {}
        
        for _, user in self.users_df.iterrows():
            user_id = user['user_id']
            
            # Activity level (low, medium, high)
            activity_level = self.rng.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2])
            
            # Session patterns
            if activity_level == 'low':
                avg_session_length = self.rng.normal(3, 1)
                sessions_per_day = self.rng.poisson(0.5)
            elif activity_level == 'medium':
                avg_session_length = self.rng.normal(8, 2)
                sessions_per_day = self.rng.poisson(1.5)
            else:  # high
                avg_session_length = self.rng.normal(15, 3)
                sessions_per_day = self.rng.poisson(3)
            
            # Preference diversity (how varied their interests are)
            diversity = self.rng.beta(2, 5)  # Skewed towards lower diversity
            
            # Time of day preferences
            hour_preferences = self.rng.dirichlet(np.ones(24))
            
            profiles[user_id] = {
                'activity_level': activity_level,
                'avg_session_length': max(1, avg_session_length),
                'sessions_per_day': max(0.1, sessions_per_day),
                'diversity': diversity,
                'hour_preferences': hour_preferences,
                'preferred_genres': self._get_user_genre_preferences(user_id)
            }
        
        return profiles
    
    def _get_user_genre_preferences(self, user_id: int) -> Dict[str, float]:
        """Get user's genre preferences based on demographics"""
        
        # Extract all genres
        all_genres = set()
        for genres_str in self.items_df['genres'].fillna(''):
            if genres_str and genres_str != '(no genres listed)':
                all_genres.update(genres_str.split('|'))
        
        all_genres = sorted(list(all_genres))
        
        # Create preference distribution
        # More realistic preferences based on user demographics
        preferences = {}
        
        # Base preferences (slightly random)
        base_prefs = self.rng.dirichlet(np.ones(len(all_genres)) * 0.5)
        
        for i, genre in enumerate(all_genres):
            preferences[genre] = base_prefs[i]
        
        return preferences
    
    def _create_item_popularity(self) -> Dict[int, float]:
        """Create item popularity distribution (Zipfian)"""
        
        n_items = len(self.items_df)
        ranks = np.arange(1, n_items + 1)
        
        # Zipfian distribution
        zipf_probs = 1.0 / (ranks ** 1.2)
        zipf_probs = zipf_probs / np.sum(zipf_probs)
        
        popularity = {}
        for i, item_id in enumerate(self.items_df['item_id']):
            popularity[item_id] = zipf_probs[i]
        
        return popularity
    
    def simulate_user_session(self, 
                            user_id: int,
                            timestamp: datetime,
                            context: Optional[Dict] = None) -> List[Dict]:
        """
        Simulate a single user session
        
        Args:
            user_id: User ID
            timestamp: Session start time
            context: Additional context (device, location, etc.)
            
        Returns:
            List of interaction events
        """
        
        if user_id not in self.user_profiles:
            return []
        
        profile = self.user_profiles[user_id]
        session_events = []
        
        # Session length
        session_length = max(1, int(self.rng.normal(
            profile['avg_session_length'], 
            profile['avg_session_length'] / 3
        )))
        
        current_time = timestamp
        
        for step in range(session_length):
            # Select item based on preferences and popularity
            item_id = self._select_item(user_id, profile, session_events)
            
            # Determine action type
            action = self._determine_action(user_id, item_id, step, session_length)
            
            # Create event
            event = {
                'user_id': user_id,
                'item_id': item_id,
                'action': action,
                'timestamp': current_time,
                'session_step': step,
                'context': context or {}
            }
            
            session_events.append(event)
            
            # Update timestamp (30 seconds to 5 minutes between actions)
            time_delta = timedelta(seconds=self.rng.randint(30, 300))
            current_time += time_delta
            
            # Early session termination (user gets bored)
            if action in ['dislike', 'skip'] and self.rng.random() < 0.3:
                break
        
        return session_events
    
    def _select_item(self, 
                    user_id: int, 
                    profile: Dict, 
                    session_history: List[Dict]) -> int:
        """Select item based on user preferences and session context"""
        
        # Get items not yet seen in this session
        seen_items = {event['item_id'] for event in session_history}
        available_items = [
            item_id for item_id in self.items_df['item_id'] 
            if item_id not in seen_items
        ]
        
        if not available_items:
            available_items = list(self.items_df['item_id'])
        
        # Calculate selection probabilities
        probs = []
        
        for item_id in available_items:
            item_row = self.items_df[self.items_df['item_id'] == item_id].iloc[0]
            
            # Base popularity
            pop_score = self.item_popularity[item_id]
            
            # Genre preference score
            genre_score = self._calculate_genre_score(item_row['genres'], profile)
            
            # Diversity bonus (if user likes diverse content)
            diversity_bonus = 1.0
            if session_history:
                diversity_bonus = self._calculate_diversity_bonus(
                    item_row['genres'], 
                    session_history, 
                    profile['diversity']
                )
            
            # Combine scores
            total_score = pop_score * 0.3 + genre_score * 0.5 + diversity_bonus * 0.2
            probs.append(total_score)
        
        # Normalize probabilities
        probs = np.array(probs)
        probs = probs / np.sum(probs)
        
        # Select item
        selected_idx = self.rng.choice(len(available_items), p=probs)
        return available_items[selected_idx]
    
    def _calculate_genre_score(self, genres_str: str, profile: Dict) -> float:
        """Calculate how well item genres match user preferences"""
        
        if not genres_str or genres_str == '(no genres listed)':
            return 0.1
        
        genres = genres_str.split('|')
        
        score = 0.0
        for genre in genres:
            if genre in profile['preferred_genres']:
                score += profile['preferred_genres'][genre]
        
        return score / len(genres) if genres else 0.0
    
    def _calculate_diversity_bonus(self, 
                                  item_genres: str,
                                  session_history: List[Dict],
                                  diversity_preference: float) -> float:
        """Calculate diversity bonus based on session history"""
        
        if not session_history:
            return 1.0
        
        # Get genres from session history
        session_genres = set()
        for event in session_history:
            item_row = self.items_df[self.items_df['item_id'] == event['item_id']]
            if len(item_row) > 0:
                hist_genres = item_row.iloc[0]['genres']
                if hist_genres and hist_genres != '(no genres listed)':
                    session_genres.update(hist_genres.split('|'))
        
        # Get current item genres
        if not item_genres or item_genres == '(no genres listed)':
            current_genres = set()
        else:
            current_genres = set(item_genres.split('|'))
        
        # Calculate overlap
        overlap = len(session_genres & current_genres) / max(len(current_genres), 1)
        
        # Diversity bonus: high diversity users prefer less overlap
        diversity_bonus = 1.0 - (overlap * diversity_preference)
        
        return max(0.1, diversity_bonus)
    
    def _determine_action(self, 
                         user_id: int, 
                         item_id: int, 
                         step: int, 
                         session_length: int) -> str:
        """Determine user action for an item"""
        
        profile = self.user_profiles[user_id]
        
        # Get item info
        item_row = self.items_df[self.items_df['item_id'] == item_id].iloc[0]
        
        # Calculate engagement probability based on genre preferences
        genre_score = self._calculate_genre_score(item_row['genres'], profile)
        
        # Base engagement probability
        base_engagement = 0.3 + genre_score * 0.4
        
        # Adjust based on session position
        position_factor = 1.0 - (step / session_length) * 0.3  # Fatigue over time
        
        engagement_prob = base_engagement * position_factor
        
        # Determine specific actions
        if self.rng.random() < engagement_prob:
            # Engaged actions
            action_probs = {
                'view': 0.4,
                'like': 0.3,
                'share': 0.1,
                'comment': 0.1,
                'purchase': 0.1
            }
        else:
            # Disengaged actions
            action_probs = {
                'skip': 0.6,
                'dislike': 0.2,
                'view': 0.2
            }
        
        # Select action
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        
        return self.rng.choice(actions, p=probs)
    
    def simulate_period(self, 
                       start_date: datetime,
                       end_date: datetime,
                       output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Simulate clickstream data for a time period
        
        Args:
            start_date: Start date for simulation
            end_date: End date for simulation
            output_file: File to save results (optional)
            
        Returns:
            DataFrame with simulated clickstream events
        """
        
        logger.info(f"Simulating clickstream from {start_date} to {end_date}")
        
        all_events = []
        current_date = start_date
        
        while current_date < end_date:
            # Simulate each user's activity for this day
            for user_id in self.users_df['user_id']:
                profile = self.user_profiles[user_id]
                
                # Determine number of sessions
                num_sessions = self.rng.poisson(profile['sessions_per_day'])
                
                for _ in range(num_sessions):
                    # Select session time based on preferences
                    hour = self.rng.choice(24, p=profile['hour_preferences'])
                    minute = self.rng.randint(0, 60)
                    
                    session_time = current_date.replace(
                        hour=hour, 
                        minute=minute,
                        second=self.rng.randint(0, 60)
                    )
                    
                    # Generate context
                    context = {
                        'device': self.rng.choice(['mobile', 'desktop', 'tablet'], 
                                                p=[0.6, 0.3, 0.1]),
                        'hour_of_day': hour,
                        'day_of_week': current_date.weekday(),
                        'is_weekend': current_date.weekday() >= 5
                    }
                    
                    # Simulate session
                    session_events = self.simulate_user_session(
                        user_id, session_time, context
                    )
                    
                    all_events.extend(session_events)
            
            # Move to next day
            current_date += timedelta(days=1)
            
            if len(all_events) % 10000 == 0:
                logger.info(f"Generated {len(all_events)} events...")
        
        # Convert to DataFrame
        events_df = pd.DataFrame(all_events)
        
        if len(events_df) > 0:
            # Sort by timestamp
            events_df = events_df.sort_values('timestamp').reset_index(drop=True)
            
            # Add additional features
            events_df['hour'] = events_df['timestamp'].dt.hour
            events_df['day_of_week'] = events_df['timestamp'].dt.dayofweek
            events_df['is_weekend'] = events_df['day_of_week'] >= 5
        
        logger.info(f"Generated {len(events_df)} total events")
        
        # Save to file if specified
        if output_file:
            events_df.to_csv(output_file, index=False)
            logger.info(f"Saved clickstream data to {output_file}")
        
        return events_df


def create_ctr_features(events_df: pd.DataFrame,
                       users_df: pd.DataFrame,
                       items_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features for CTR prediction from clickstream data
    
    Args:
        events_df: Clickstream events
        users_df: Users data
        items_df: Items data
        
    Returns:
        DataFrame with CTR features
    """
    
    logger.info("Creating CTR features from clickstream data")
    
    # Define positive actions (clicked/engaged)
    positive_actions = {'like', 'share', 'comment', 'purchase', 'view'}
    
    ctr_data = []
    
    # Group events by user-item pairs
    for (user_id, item_id), group in events_df.groupby(['user_id', 'item_id']):
        
        # User historical features
        user_events = events_df[events_df['user_id'] == user_id]
        user_positive = user_events[user_events['action'].isin(positive_actions)]
        
        user_features = {
            'avg_session_length': user_events.groupby(user_events['timestamp'].dt.date)['session_step'].max().mean(),
            'total_interactions': len(user_events),
            'positive_rate': len(user_positive) / len(user_events) if len(user_events) > 0 else 0,
            'unique_items': user_events['item_id'].nunique(),
            'avg_hour': user_events['hour'].mean(),
            'weekend_ratio': user_events['is_weekend'].mean(),
        }
        
        # Item features
        item_events = events_df[events_df['item_id'] == item_id]
        item_positive = item_events[item_events['action'].isin(positive_actions)]
        
        item_features = {
            'item_popularity': len(item_events),
            'item_ctr': len(item_positive) / len(item_events) if len(item_events) > 0 else 0,
            'unique_users': item_events['user_id'].nunique(),
        }
        
        # Context features (from first interaction)
        first_event = group.iloc[0]
        context_features = first_event['context']
        
        # Target: whether user engaged positively
        clicked = any(action in positive_actions for action in group['action'])
        
        # Combine features
        feature_row = {
            'user_id': user_id,
            'item_id': item_id,
            'clicked': int(clicked),
            **user_features,
            **item_features,
            **context_features
        }
        
        ctr_data.append(feature_row)
    
    ctr_df = pd.DataFrame(ctr_data)
    logger.info(f"Created CTR dataset with {len(ctr_df)} samples")
    
    return ctr_df