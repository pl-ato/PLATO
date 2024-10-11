import sys
sys.path.append('./env/build/')
import mover
import torch
import numpy as np
from PIL import Image
import json
import os

torch.manual_seed(0)

def greedy(dataset: str, save_json: str):
    from utils.dataIO import DataIO
    
    episode_len = 200
    max_num = 20
    action_type = 6
    action_space = max_num * action_type
    
    env_test = DataIO(dataset, 1, -1)
    print(len(env_test.data))

    logs = {}
    logs['step'] = []
    logs['accumulated_reward'] = []
    logs['suc'] = []
    logs['num'] = []
    logs['action'] = []

    for idx, d in enumerate(env_test.data):
        env = mover.Env(64, 64, max_num)
        env.set_map(d['pos'][0:max_num].tolist(), d['target'][0:max_num].tolist(), d['shape'][0:max_num], d['cstate'][0:max_num].tolist(), d['tstate'][0:max_num].tolist(), d['wall'], d['num'])

        obj_num = d['num']
        reward_list = []
        path_length = episode_len
        succ = False
        action_list = []
        for i in range(episode_len):
            action = -1
            find = False
            for a in range(obj_num * action_type): # first, find an action whose reawrd is greater than 0
                t_env = env.copy()

                _item = int(a / action_type)
                _action = int(a % action_type)
                reward, done = t_env.move(_item, _action)
                if done != -1 and reward > 0:
                    find = True
                    action = a
                    break
                else:
                    continue
            
            if find == False: # if can't find an action whose reward is greater than 0, then randomly choose one
                logits = torch.ones(obj_num * action_type).float()

                for _ in range(obj_num * action_type):
                    prob = torch.nn.functional.softmax(logits, -1)
                    c = torch.distributions.Categorical(prob)
                    a_t = c.sample().numpy()

                    t_env = env.copy()
                    _item = int(a_t / action_type)
                    _action = int(a_t % action_type)
                    reward, done = t_env.move(_item, _action)

                    if done == -1:
                        logits[a_t] = torch.tensor(-float("inf"))
                        continue
                    else:
                        action = a_t.item()
                        break

            if action == -1:
                raise Exception('Can\'t find an action to execute.')
            else:
                _item = int(action / action_type)
                _action = int(action % action_type)
                reward, done = env.move(_item, _action)
                reward_list.append(reward)
                action_list.append(action)
                if done == 1:
                    path_length = i + 1
                    succ = True
                    break
        
        discount_reward = 0
        for t in range(len(reward_list)-1, -1, -1):
            discount_reward = reward_list[t] + 0.95 * discount_reward
        
        logs['step'].append(path_length)
        logs['accumulated_reward'].append(discount_reward)
        logs['suc'].append(1 if succ else 0)
        logs['num'].append(obj_num)
        logs['action'].append(action_list)

        print('Test data {} with object number {}, succ : {}, step : {}.'.format(idx, obj_num, 'True' if succ else 'False', path_length))
    
    obj_suc = [0] * 4
    obj_num = [0] * 4
    for n,s in zip(logs['num'], logs['suc']):
        idx = (n - 1) // 5
        obj_suc[idx] += s
        obj_num[idx] += 1

    logs['sr'] = sum(logs['suc']) / len(logs['suc'])
    logs['ms'] = sum(logs['step']) / len(logs['step'])
    logs['mr'] = sum(logs['accumulated_reward']) / len(logs['accumulated_reward'])
    logs['obj_sr'] = [0] * 4
    for i in range(4):
        if obj_num[i] != 0:
            logs['obj_sr'][i] = obj_suc[i] / obj_num[i]

    logs['obj_suc'] = obj_suc
    logs['obj_num'] = obj_num

    os.makedirs(os.path.dirname(save_json), exist_ok=True)
    with open(save_json, 'w') as fp:
        json.dump(logs, fp)

def visualize(dataset: str, save_json: str, index: int):
    from utils.util import restore_case
    from utils.dataIO import DataIO

    with open(save_json, 'r') as fp:
        logs = json.load(fp)
    
    max_num = 20
    env_test = DataIO(dataset, 1, -1)
    d = env_test.data[index]

    env = mover.Env(64, 64, max_num)
    env.set_map(d['pos'][0:max_num].tolist(), d['target'][0:max_num].tolist(), d['shape'][0:max_num], d['cstate'][0:max_num].tolist(), d['tstate'][0:max_num].tolist(), d['wall'], d['num'])

    save_dir = os.path.join(os.path.dirname(save_json), 'result')
    restore_case(save_dir, index, logs['action'][index], env)

if __name__ == '__main__':
    dataset = './data/validate_data_259.pkl'
    save_json = './output/greedy/greedy.json'
    index = 2 # visualize the index-th case

    # greedy(dataset, save_json)
    visualize(dataset, save_json, index)