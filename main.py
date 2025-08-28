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
# from src.agent.unigoal.agent import UniGoal_Agent
from src.agent.unigoal.enhanced_agent import EnhancedUniGoalAgent as UniGoal_Agent

from src.map.bev_mapping import BEV_Map
from src.graph.graph import Graph
import gzip
import time
import datetime

def get_config():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="configs/config_habitat.yaml",
                        metavar="FILE", help="path to config file", type=str)
    parser.add_argument("--goal_type", default="ins-image", type=str)
    parser.add_argument("--episode_id", default=-1, type=int, help="episode id, 0~999")
    parser.add_argument("--goal", default="", type=str)
    parser.add_argument("--real_world", action="store_true")

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

from src.utils.time import TrainingTimeTracker
def setup_timing_tracker(args):
    timing_save_path = os.path.join(args.dump_location, args.experiment_id, "timing_data.json")
    timer = TrainingTimeTracker(save_path=timing_save_path)
    
    timer.start_training()
    timer.set_total_episodes(args.num_eval_episodes)
    
    print(f"Starting evaluation of {args.num_eval_episodes} episodes")
    print(f"Timing data will be saved to: {timing_save_path}")
    
    return timer

def main():
    args = get_config()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.visualization_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(args.log_dir, 'eval.log'),
        level=logging.INFO)
    logging.info(args)

    eval_metrics_id = 0

    episode_success = deque(maxlen=args.num_episodes)
    episode_spl = deque(maxlen=args.num_episodes)

    finished = False
    wait_env = False

    if args.goal_type == 'text':
        with gzip.open(args.text_goal_dataset, 'rt') as f:
            text_goal_dataset = json.load(f)
    elif args.goal_type == 'object':
        args.task_config = "tasks/objectnav_hm3d.yaml"  # Giữ nguyên vì construct_envs sẽ thêm "configs/" vào trước
        # Cập nhật để trỏ đúng đến file chỉ mục val.json.gz  
        args.objectnav_dataset = "data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2_locobot_multifloor/val/val.json.gz"

    BEV_map = BEV_Map(args)
    graph = Graph(args)
    envs = construct_envs(args)
    agent = UniGoal_Agent(args, envs)

    BEV_map.init_map_and_pose()
    obs, rgbd, infos = agent.reset()

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
    agent_input['sem_map'] = BEV_map.local_map[0, 4:11, :, :
                                        ].cpu().numpy()
    if args.visualize:
        BEV_map.local_map[0, 10, :, :] = 1e-5
        agent_input['sem_map_pred'] = BEV_map.local_map[0, 4:11, :, :
                                            ].argmax(0).cpu().numpy()

    obs, rgbd, done, infos = agent.step(agent_input)

    graph.reset()
    graph.set_obj_goal(infos['goal_name'])
    if args.goal_type == 'ins-image':
        graph.set_image_goal(infos['instance_imagegoal'])
    elif args.goal_type == 'text':
        graph.set_text_goal(infos['text_goal'])
    elif args.goal_type == 'object':
        graph.set_text_goal(infos['goal_name'])

    timer = setup_timing_tracker(args)
    
    step = 0
    episode_count = 0

    while True:
        if finished == True:
            break

        global_step = (step // args.num_local_steps) % args.num_global_steps
        local_step = step % args.num_local_steps

        if done:
            timer.end_episode(episode_count + 1, log_interval=args.log_interval)
            
            spl = infos['spl']
            success = infos['success']
            success = success if success is not None else 0.0
            eval_metrics_id += 1
            episode_success.append(success)
            episode_spl.append(spl)
            episode_count += 1
            
            if len(episode_success) == args.num_eval_episodes:
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
            elif args.goal_type == 'object':
                graph.set_text_goal(infos['goal_name'])
                
            if not finished:
                timer.start_episode(episode_count + 1)

        BEV_map.mapping(rgbd, infos)

        navigate_steps = global_step * args.num_local_steps + local_step
        graph.set_navigate_steps(navigate_steps)
        if not agent_input['wait'] and navigate_steps % 2 == 0:
            graph.set_observations(obs)
            graph.update_scenegraph()

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
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


        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
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
        agent_input['sem_map'] = BEV_map.local_map[0, 4:11, :, :
                                        ].cpu().numpy()

        if args.visualize:
            BEV_map.local_map[0, 10, :, :] = 1e-5
            agent_input['sem_map_pred'] = BEV_map.local_map[0, 4:11, :,
                                                :].argmax(0).cpu().numpy()

        obs, rgbd, done, infos = agent.step(agent_input)

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # log

        if step % args.log_interval == 0:
            current_time = timer.get_current_total_time()
            log = " ".join([
                "num timesteps {},".format(step),
                "episode_id {}".format(infos['episode_no']),
                "elapsed_time: {},".format(timer.format_time(current_time))
            ])

            total_success = []
            total_spl = []
            for acc in episode_success:
                total_success.append(acc)
            for spl in episode_spl:
                total_spl.append(spl)

            if len(total_spl) > 0:
                log += " Average SR/SPL:"
                log += " {:.5f}/{:.5f},".format(
                    np.mean(total_success),
                    np.mean(total_spl))
                
                avg_episode_time = timer.get_average_episode_time()
                if avg_episode_time > 0:
                    log += " Avg episode time: {},".format(timer.format_time(avg_episode_time))
                    
                    remaining_episodes = args.num_eval_episodes - len(episode_success)
                    if remaining_episodes > 0:
                        estimated_remaining = remaining_episodes * avg_episode_time
                        log += " Est. remaining: {}".format(timer.format_time(estimated_remaining))

            print(log)
            logging.info(log)
        # ------------------------------------------------------------------
        step += 1
    
    timer.end_training()
    timer.print_summary()

    total_success = []
    total_spl = []
    for acc in episode_success:
        total_success.append(acc)
    for spl in episode_spl:
        total_spl.append(spl)

    if len(total_spl) > 0:
        log = "Final Average SR/SPL:"
        log += " {:.5f}/{:.5f}".format(
            np.mean(total_success),
            np.mean(total_spl))
        print(log)
        logging.info(log)
        
    total = {'succ': total_success, 'spl': total_spl}

    results = {
        'success_rates': total_success, 
        'spl_scores': total_spl,
        'timing_summary': timer.get_timing_summary(),
        'total_episodes': len(episode_success),
        'total_time_formatted': timer.format_time(timer.get_current_total_time()),
        'average_episode_time': timer.get_average_episode_time(),
        'experiment_id': args.experiment_id,
        'completed_at': datetime.datetime.now().isoformat()
    }

    with open('{}/total.json'.format(args.log_dir), 'w') as f:
        json.dump({'succ': total_success, 'spl': total_spl}, f)
        
    with open('{}/enhanced_results.json'.format(args.log_dir), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Enhanced results saved to: {args.log_dir}/enhanced_results.json")

if __name__ == "__main__":
    main()
