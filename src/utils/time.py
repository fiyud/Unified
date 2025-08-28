import time
import datetime
from typing import Optional, Dict, Any
import json
import os

class TrainingTimeTracker:
    """
    A comprehensive training time tracking utility that can be integrated 
    into training loops, evaluation loops, or any iterative process.
    
    Features:
    - Track total training time
    - Track time per epoch/episode
    - Calculate average times and estimates
    - Save/load timing data
    - Handle pause/resume functionality
    """
    
    def __init__(self, save_path: Optional[str] = None):
        self.save_path = save_path
        self.reset()
    
    def reset(self):
        """Reset all timing data"""
        self.start_time = None
        self.end_time = None
        self.total_time = 0.0
        self.epoch_times = []
        self.episode_times = []
        self.pause_time = 0.0
        self.is_paused = False
        self.pause_start = None
        self.current_epoch_start = None
        self.current_episode_start = None
        
    def start_training(self):
        """Start the overall training timer"""
        self.start_time = time.time()
        print(f"Training started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    def end_training(self):
        """End the overall training timer"""
        if self.start_time is None:
            print("Warning: Training was never started!")
            return
            
        self.end_time = time.time()
        self.total_time = self.end_time - self.start_time - self.pause_time
        
        print(f"Training ended at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total training time: {self.format_time(self.total_time)}")
        
        if self.save_path:
            self.save_timing_data()
            
    def start_epoch(self, epoch_num: int):
        """Start timing for an epoch"""
        self.current_epoch_start = time.time()
        print(f"Epoch {epoch_num} started at: {datetime.datetime.now().strftime('%H:%M:%S')}")
        
    def end_epoch(self, epoch_num: int):
        """End timing for an epoch"""
        if self.current_epoch_start is None:
            print("Warning: Epoch was never started!")
            return
            
        epoch_time = time.time() - self.current_epoch_start
        self.epoch_times.append(epoch_time)
        
        print(f"Epoch {epoch_num} completed in: {self.format_time(epoch_time)}")
        print(f"Average epoch time: {self.format_time(self.get_average_epoch_time())}")
        
        # Estimate remaining time if we know total epochs
        if hasattr(self, 'total_epochs') and self.total_epochs:
            remaining_epochs = self.total_epochs - epoch_num
            if remaining_epochs > 0:
                estimated_remaining = remaining_epochs * self.get_average_epoch_time()
                print(f"Estimated remaining time: {self.format_time(estimated_remaining)}")
                
    def start_episode(self, episode_num: int):
        """Start timing for an episode"""
        self.current_episode_start = time.time()
        
    def end_episode(self, episode_num: int, log_interval: int = 100):
        """End timing for an episode"""
        if self.current_episode_start is None:
            print("Warning: Episode was never started!")
            return
            
        episode_time = time.time() - self.current_episode_start
        self.episode_times.append(episode_time)
        
        # Log every log_interval episodes
        if episode_num % log_interval == 0:
            print(f"Episode {episode_num} completed in: {self.format_time(episode_time)}")
            print(f"Average episode time: {self.format_time(self.get_average_episode_time())}")
            
            # Estimate remaining time if we know total episodes
            if hasattr(self, 'total_episodes') and self.total_episodes:
                remaining_episodes = self.total_episodes - episode_num
                if remaining_episodes > 0:
                    estimated_remaining = remaining_episodes * self.get_average_episode_time()
                    print(f"Estimated remaining time: {self.format_time(estimated_remaining)}")
    
    def pause(self):
        """Pause the timer (useful for debugging breaks, etc.)"""
        if not self.is_paused:
            self.pause_start = time.time()
            self.is_paused = True
            print("Timer paused")
            
    def resume(self):
        """Resume the timer"""
        if self.is_paused and self.pause_start:
            self.pause_time += time.time() - self.pause_start
            self.is_paused = False
            self.pause_start = None
            print("Timer resumed")
    
    def set_total_epochs(self, total_epochs: int):
        """Set total number of epochs for time estimation"""
        self.total_epochs = total_epochs
        
    def set_total_episodes(self, total_episodes: int):
        """Set total number of episodes for time estimation"""
        self.total_episodes = total_episodes
    
    def get_current_total_time(self) -> float:
        """Get current total elapsed time"""
        if self.start_time is None:
            return 0.0
        
        current_time = time.time()
        elapsed = current_time - self.start_time - self.pause_time
        
        if self.is_paused and self.pause_start:
            elapsed -= (current_time - self.pause_start)
            
        return elapsed
    
    def get_average_epoch_time(self) -> float:
        """Get average time per epoch"""
        if not self.epoch_times:
            return 0.0
        return sum(self.epoch_times) / len(self.epoch_times)
    
    def get_average_episode_time(self) -> float:
        """Get average time per episode"""
        if not self.episode_times:
            return 0.0
        return sum(self.episode_times) / len(self.episode_times)
    
    def format_time(self, seconds: float) -> str:
        """Format time in a human-readable format"""
        if seconds < 60:
            return f"{seconds:.2f} seconds"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.1f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}h {minutes}m {secs:.1f}s"
    
    def get_timing_summary(self) -> Dict[str, Any]:
        """Get a summary of all timing data"""
        summary = {
            'total_time': self.get_current_total_time(),
            'total_time_formatted': self.format_time(self.get_current_total_time()),
            'num_epochs': len(self.epoch_times),
            'num_episodes': len(self.episode_times),
            'average_epoch_time': self.get_average_epoch_time(),
            'average_episode_time': self.get_average_episode_time(),
            'pause_time': self.pause_time,
            'start_time': self.start_time,
            'end_time': self.end_time
        }
        
        if self.epoch_times:
            summary.update({
                'fastest_epoch': min(self.epoch_times),
                'slowest_epoch': max(self.epoch_times),
                'total_epoch_time': sum(self.epoch_times)
            })
            
        if self.episode_times:
            summary.update({
                'fastest_episode': min(self.episode_times),
                'slowest_episode': max(self.episode_times),
                'total_episode_time': sum(self.episode_times)
            })
            
        return summary
    
    def print_summary(self):
        """Print a formatted summary of timing data"""
        summary = self.get_timing_summary()
        
        print("\n" + "="*50)
        print("TRAINING TIME SUMMARY")
        print("="*50)
        print(f"Total time: {summary['total_time_formatted']}")
        print(f"Pause time: {self.format_time(summary['pause_time'])}")
        
        if summary['num_epochs'] > 0:
            print(f"\nEpoch Statistics:")
            print(f"  Total epochs: {summary['num_epochs']}")
            print(f"  Average time per epoch: {self.format_time(summary['average_epoch_time'])}")
            print(f"  Fastest epoch: {self.format_time(summary['fastest_epoch'])}")
            print(f"  Slowest epoch: {self.format_time(summary['slowest_epoch'])}")
            
        if summary['num_episodes'] > 0:
            print(f"\nEpisode Statistics:")
            print(f"  Total episodes: {summary['num_episodes']}")
            print(f"  Average time per episode: {self.format_time(summary['average_episode_time'])}")
            print(f"  Fastest episode: {self.format_time(summary['fastest_episode'])}")
            print(f"  Slowest episode: {self.format_time(summary['slowest_episode'])}")
            
        print("="*50)
    
    def save_timing_data(self):
        """Save timing data to file"""
        if not self.save_path:
            return
            
        timing_data = self.get_timing_summary()
        timing_data['epoch_times'] = self.epoch_times
        timing_data['episode_times'] = self.episode_times
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        
        with open(self.save_path, 'w') as f:
            json.dump(timing_data, f, indent=2)
            
        print(f"Timing data saved to: {self.save_path}")
    
    def load_timing_data(self):
        """Load timing data from file"""
        if not self.save_path or not os.path.exists(self.save_path):
            return
            
        with open(self.save_path, 'r') as f:
            data = json.load(f)
            
        self.epoch_times = data.get('epoch_times', [])
        self.episode_times = data.get('episode_times', [])
        self.pause_time = data.get('pause_time', 0.0)
        self.start_time = data.get('start_time')
        self.end_time = data.get('end_time')
        
        print(f"Timing data loaded from: {self.save_path}")


# Usage example and integration functions
def integrate_with_main_py(timer: TrainingTimeTracker):
    """
    Example of how to integrate the timer with your main.py evaluation loop
    """
    print("""
    # Add this to the beginning of your main.py:
    from training_time_tracker import TrainingTimeTracker
    
    timer = TrainingTimeTracker(save_path="./outputs/timing_data.json")
    timer.start_training()
    timer.set_total_episodes(args.num_episodes)
    
    # In your main loop, after 'while True:':
    timer.start_episode(len(episode_success) + 1)
    
    # At the end of each episode (when done=True):
    timer.end_episode(len(episode_success), log_interval=args.log_interval)
    
    # At the very end of main():
    timer.end_training()
    timer.print_summary()
    """)