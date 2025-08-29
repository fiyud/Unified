import os
import sys
import json
import logging
import time
import yaml
from collections import deque, defaultdict
from types import SimpleNamespace
import numpy as np
import torch
import argparse
from src.envs import construct_envs
from src.agent.unigoal.enhanced_agent import EnhancedUniGoalAgent as UniGoal_Agent
from src.map.bev_mapping import BEV_Map
from src.graph.graph import Graph
import gzip
from datetime import datetime, timedelta

def get_config():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="configs/config_habitat.yaml",
                        metavar="FILE", help="path to config file", type=str)
    parser.add_argument("--goal_type", default="ins-image", type=str)
    parser.add_argument("--episode_id", default=-1, type=int, help="episode id, 0~999")
    parser.add_argument("--goal", default="", type=str)
    parser.add_argument("--real_world", action="store_true")
    
    parser.add_argument("--start_episode", default=0, type=int, 
                       help="Episode to start from (for resuming evaluation)")
    parser.add_argument("--resume_metrics", nargs=2, type=float, default=[0.0, 0.0],
                       metavar=('SR', 'SPL'),
                       help="Previous SR and SPL to continue from: --resume_metrics 0.73446 0.27046")
    parser.add_argument("--checkpoint_interval", default=10, type=int,
                       help="Save checkpoint every N episodes")
    parser.add_argument("--load_checkpoint", action="store_true",
                       help="Automatically load from checkpoint file if exists")

    args = parser.parse_args()

    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)
    args = vars(args)
    args.update(config)
        
    args = SimpleNamespace(**args)

    args.is_debugging = sys.gettrace() is not None
    if args.is_debugging:
        args.experiment_id = "debug"
    
    args.log_dir = os.path.join(args.dump_location, args.experiment_id, 'log')
    args.visualization_dir = os.path.join(args.dump_location, args.experiment_id, 'visualization')

    args.map_size = args.map_size_cm // args.map_resolution
    args.global_width, args.global_height = args.map_size, args.map_size
    args.local_width = int(args.global_width / args.global_downscaling)
    args.local_height = int(args.global_height / args.global_downscaling)

    args.device = torch.device("cuda:0" if args.cuda else "cpu")
    args.num_scenes = args.num_processes
    args.num_episodes = int(args.num_eval_episodes)

    return args


def save_checkpoint(checkpoint_path, episode_id, episode_success, episode_spl, 
                   start_time, eval_metrics_id, args):
    """Save current evaluation state to checkpoint file"""
    checkpoint = {
        'last_completed_episode': episode_id,
        'episode_success': list(episode_success),
        'episode_spl': list(episode_spl),
        'eval_metrics_id': eval_metrics_id,
        'start_time': start_time.isoformat(),
        'total_episodes_completed': len(episode_success),
        'current_sr': np.mean(episode_success) if episode_success else 0.0,
        'current_spl': np.mean(episode_spl) if episode_spl else 0.0,
        'args': {
            'goal_type': args.goal_type,
            'num_episodes': args.num_episodes,
            'start_episode': args.start_episode,
            'resume_metrics': args.resume_metrics
        }
    }
    
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    
    print(f"Checkpoint saved: Episode {episode_id}, SR: {checkpoint['current_sr']:.5f}, SPL: {checkpoint['current_spl']:.5f}")


def load_checkpoint(checkpoint_path, args):
    """Load evaluation state from checkpoint file"""
    if not os.path.exists(checkpoint_path):
        return None
    
    try:
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        
        print(f"Found checkpoint from episode {checkpoint['last_completed_episode']}")
        print(f"Previous metrics - SR: {checkpoint['current_sr']:.5f}, SPL: {checkpoint['current_spl']:.5f}")
        
        # Update args with checkpoint info
        args.start_episode = checkpoint['last_completed_episode'] + 1
        args.resume_metrics = [checkpoint['current_sr'], checkpoint['current_spl']]
        
        return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None


def calculate_combined_metrics(previous_results, new_results, start_episode):
    """Calculate combined metrics from previous and new results"""
    if start_episode == 0 or not previous_results:
        # No previous results to combine
        return new_results
    
    # Convert deques to lists
    new_success = list(new_results['success'])
    new_spl = list(new_results['spl'])
    
    # Combine with previous results
    total_success = previous_results + new_success
    total_spl = previous_results + new_spl
    
    return {
        'success': total_success,
        'spl': total_spl,
        'combined_sr': np.mean(total_success),
        'combined_spl': np.mean(total_spl),
        'new_episodes': len(new_success),
        'total_episodes': len(total_success)
    }


def format_time(seconds):
    """Format seconds into human readable time"""
    return str(timedelta(seconds=int(seconds)))


def main():
    args = get_config()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.visualization_dir, exist_ok=True)

    # Checkpoint file path
    checkpoint_path = os.path.join(args.log_dir, 'eval_checkpoint.json')
    
    # Load checkpoint if requested
    checkpoint = None
    if args.load_checkpoint:
        checkpoint = load_checkpoint(checkpoint_path, args)
    
    logging.basicConfig(
        filename=os.path.join(args.log_dir, 'eval.log'),
        level=logging.INFO)
    logging.info(f"Starting evaluation - Start episode: {args.start_episode}, Resume metrics: {args.resume_metrics}")

    eval_metrics_id = args.start_episode if checkpoint else 0
    
    # Initialize episode tracking
    episode_success = deque(maxlen=args.num_episodes)
    episode_spl = deque(maxlen=args.num_episodes)
    
    # Load previous results if resuming
    previous_success_list = []
    previous_spl_list = []
    if checkpoint:
        previous_success_list = checkpoint['episode_success']
        previous_spl_list = checkpoint['episode_spl']
        
        # Fill deques with previous results for current window
        for success, spl in zip(previous_success_list, previous_spl_list):
            episode_success.append(success)
            episode_spl.append(spl)

    finished = False
    wait_env = False
    start_time = datetime.now()

    if args.goal_type == 'text':
        with gzip.open(args.text_goal_dataset, 'rt') as f:
            text_goal_dataset = json.load(f)

    BEV_map = BEV_Map(args)
    graph = Graph(args)
    envs = construct_envs(args)
    agent = UniGoal_Agent(args, envs)

    BEV_map.init_map_and_pose()
    obs, rgbd, infos = agent.reset()

    # Skip to the desired episode if resuming
    current_episode = 0
    while current_episode < args.start_episode:
        obs, rgbd, done, infos = agent.step({'wait': True})
        if done:
            current_episode += 1
            if current_episode < args.start_episode:
                agent.reset()
        if current_episode >= args.num_episodes:
            finished = True
            break

    if finished:
        print(f"Requested start episode {args.start_episode} exceeds total episodes {args.num_episodes}")
        return

    BEV_map.mapping(rgbd, infos)

    global_goals = [args.local_width // 2, args.local_height // 2]
    goal_maps = np.zeros((args.local_width, args.local_height))
    goal_maps[global_goals[0], global_goals[1]] = 1

    agent_input = {}
    agent_input['map_pred'] = BEV_map.local_map[0, 0, :, :].cpu().numpy()
    agent_input['exp_pred'] = BEV_map.local_map[0, 1, :, :].cpu().numpy()
    agent_input['pose_pred'] = BEV_map.planner_pose_inputs[0]
    agent_input['goal'] = goal_maps
    agent_input['exp_goal'] = goal_maps * 1
    agent_input['new_goal'] = 1
    agent_input['found_goal'] = 0
    agent_input['wait'] = wait_env or finished
    agent_input['sem_map'] = BEV_map.local_map[0, 4:11, :, :].cpu().numpy()
    
    if args.visualize:
        BEV_map.local_map[0, 10, :, :] = 1e-5
        agent_input['sem_map_pred'] = BEV_map.local_map[0, 4:11, :, :].argmax(0).cpu().numpy()

    obs, rgbd, done, infos = agent.step(agent_input)

    graph.reset()
    graph.set_obj_goal(infos['goal_name'])
    if args.goal_type == 'ins-image':
        graph.set_image_goal(infos['instance_imagegoal'])
    elif args.goal_type == 'text':
        graph.set_text_goal(infos['text_goal'])

    step = 0
    episodes_completed_this_run = 0

    print(f"Starting evaluation from episode {args.start_episode}")
    if args.start_episode > 0:
        print(f"Previous results: SR={args.resume_metrics[0]:.5f}, SPL={args.resume_metrics[1]:.5f}")

    while True:
        if finished == True:
            break

        global_step = (step // args.num_local_steps) % args.num_global_steps
        local_step = step % args.num_local_steps

        if done:
            spl = infos['spl']
            success = infos['success']
            success = success if success is not None else 0.0
            eval_metrics_id += 1
            episodes_completed_this_run += 1
            
            episode_success.append(success)
            episode_spl.append(spl)
            
            current_episode_id = args.start_episode + episodes_completed_this_run - 1
            
            # Save checkpoint periodically
            if episodes_completed_this_run % args.checkpoint_interval == 0:
                save_checkpoint(checkpoint_path, current_episode_id, episode_success, 
                              episode_spl, start_time, eval_metrics_id, args)
            
            if len(episode_success) == args.num_episodes or current_episode_id >= args.num_episodes - 1:
                finished = True
                
            if args.visualize:
                video_path = os.path.join(args.visualization_dir, 'videos', 'eps_{:0>6}.mp4'.format(infos['episode_no']))
                agent.save_visualization(video_path)
                
            wait_env = True
            BEV_map.update_intrinsic_rew()
            BEV_map.init_map_and_pose_for_env()

            graph.reset()
            graph.set_obj_goal(infos['goal_name'])
            if args.goal_type == 'ins-image':
                graph.set_image_goal(infos['instance_imagegoal'])
            elif args.goal_type == 'text':
                graph.set_text_goal(infos['text_goal'])

        BEV_map.mapping(rgbd, infos)

        navigate_steps = global_step * args.num_local_steps + local_step
        graph.set_navigate_steps(navigate_steps)
        if not agent_input['wait'] and navigate_steps % 2 == 0:
            graph.set_observations(obs)
            graph.update_scenegraph()

        if local_step == args.num_local_steps - 1 or np.linalg.norm(np.array([BEV_map.local_row, BEV_map.local_col]) - np.array(global_goals)) < 10:
            if wait_env == True:
                wait_env = False
            else:
                BEV_map.update_intrinsic_rew()

            BEV_map.move_local_map()

            graph.set_full_map(BEV_map.full_map)
            graph.set_full_pose(BEV_map.full_pose)
            goal = graph.explore()
            if hasattr(graph, 'frontier_locations_16'):
                graph.frontier_locations_16[:, 0] = graph.frontier_locations_16[:, 0] - BEV_map.local_map_boundary[0, 0]
                graph.frontier_locations_16[:, 1] = graph.frontier_locations_16[:, 1] - BEV_map.local_map_boundary[0, 2]
            if isinstance(goal, list) or isinstance(goal, np.ndarray):
                goal = list(goal)
                goal[0] = goal[0] - BEV_map.local_map_boundary[0, 0]
                goal[1] = goal[1] - BEV_map.local_map_boundary[0, 2]
                if 0 <= goal[0] < args.local_width and 0 <= goal[1] < args.local_height:
                    global_goals = goal

        found_goal = False
        goal_maps = np.zeros((args.local_width, args.local_height))
        goal_maps[global_goals[0], global_goals[1]] = 1
        exp_goal_maps = goal_maps.copy()

        agent_input = {}
        agent_input['map_pred'] = BEV_map.local_map[0, 0, :, :].cpu().numpy()
        agent_input['exp_pred'] = BEV_map.local_map[0, 1, :, :].cpu().numpy()
        agent_input['pose_pred'] = BEV_map.planner_pose_inputs[0]
        agent_input['goal'] = goal_maps
        agent_input['exp_goal'] = exp_goal_maps
        agent_input['new_goal'] = local_step == args.num_local_steps - 1
        agent_input['found_goal'] = found_goal
        agent_input['wait'] = wait_env or finished
        agent_input['sem_map'] = BEV_map.local_map[0, 4:11, :, :].cpu().numpy()

        if args.visualize:
            BEV_map.local_map[0, 10, :, :] = 1e-5
            agent_input['sem_map_pred'] = BEV_map.local_map[0, 4:11, :, :].argmax(0).cpu().numpy()

        obs, rgbd, done, infos = agent.step(agent_input)

        # Enhanced logging with time estimates
        if step % args.log_interval == 0:
            current_time = datetime.now()
            elapsed_time = current_time - start_time
            
            current_episode_id = args.start_episode + episodes_completed_this_run
            total_steps_so_far = step
            
            log = " ".join([
                "num timesteps {},".format(total_steps_so_far),
                "episode_id {}".format(current_episode_id),
                "elapsed_time: {},".format(format_time(elapsed_time.total_seconds())),
            ])

            # Calculate current window metrics
            total_success = list(episode_success)
            total_spl = list(episode_spl)

            if len(total_spl) > 0:
                current_sr = np.mean(total_success)
                current_spl = np.mean(total_spl)
                log += " Average SR/SPL: {:.5f}/{:.5f},".format(current_sr, current_spl)
                
                # Calculate episode time and remaining estimate
                if episodes_completed_this_run > 0:
                    avg_episode_time = elapsed_time.total_seconds() / episodes_completed_this_run
                    remaining_episodes = args.num_episodes - current_episode_id
                    est_remaining_time = avg_episode_time * remaining_episodes
                    
                    log += " Avg episode time: {},".format(format_time(avg_episode_time))
                    log += " Est. remaining: {}".format(format_time(est_remaining_time))

            print(log)
            logging.info(log)
            
        step += 1

    # Final checkpoint save
    final_episode_id = args.start_episode + episodes_completed_this_run - 1
    save_checkpoint(checkpoint_path, final_episode_id, episode_success, 
                   episode_spl, start_time, eval_metrics_id, args)

    # Calculate final combined results
    current_success = list(episode_success)
    current_spl = list(episode_spl)
    
    # Combine with previous results if resuming
    if args.start_episode > 0:
        # Reconstruct full results
        all_success = previous_success_list + current_success[len(previous_success_list):]
        all_spl = previous_spl_list + current_spl[len(previous_spl_list):]
        
        final_sr = np.mean(all_success)
        final_spl = np.mean(all_spl)
        
        print(f"\n=== FINAL COMBINED RESULTS ===")
        print(f"Total episodes: {len(all_success)}")
        print(f"Episodes completed this run: {episodes_completed_this_run}")
        print(f"Final Combined SR/SPL: {final_sr:.5f}/{final_spl:.5f}")
        
        # Save combined results
        total = {
            'succ': all_success, 
            'spl': all_spl,
            'final_sr': final_sr,
            'final_spl': final_spl,
            'episodes_this_run': episodes_completed_this_run,
            'start_episode': args.start_episode,
            'resume_metrics': args.resume_metrics
        }
    else:
        final_sr = np.mean(current_success)
        final_spl = np.mean(current_spl)
        
        print(f"\n=== FINAL RESULTS ===")
        print(f"Total episodes: {len(current_success)}")
        print(f"Final SR/SPL: {final_sr:.5f}/{final_spl:.5f}")
        
        total = {'succ': current_success, 'spl': current_spl}

    log = "Final Average SR/SPL: {:.5f}/{:.5f}".format(final_sr, final_spl)
    print(log)
    logging.info(log)

    with open('{}/total.json'.format(args.log_dir), 'w') as f:
        json.dump(total, f, indent=2)

    print(f"Results saved to {args.log_dir}/total.json")
    print(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
