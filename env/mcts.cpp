#include <iostream>
#include <queue>
#include <cstring>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <algorithm>
#include <thread>
#include <pthread.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "hpp/mcts.hpp"
#include "hpp/env.hpp"

using namespace std;

#define gamma 0.95

/*
 * Tree Node Class
*/
TreeNode::TreeNode(int _state_size) {
    state_size = _state_size;
    env = NULL;
    h_state = new float[state_size];
    c_state = new float[state_size];

    id = 0;
    times = 0;
    reward = 0;
    value = 0;
    faid = -1;
    action = -1;
    best = -1;
    depth = 0;
}

TreeNode::~TreeNode() {
    delete h_state;
    delete c_state;
    if (env != NULL) {
        delete env;
    }
}

void TreeNode::add_child(int action, int child) {
    childs[action] = child;
}

int TreeNode::get_child(int action) {
    if (has_child(action)) {
        return childs.at(action);
    } else {
        return -1;
    }
}

bool TreeNode::has_child(int action) {
    unordered_map<int, int>::iterator g = childs.find(action);
    if (g == childs.end()) {
        return false;
    } else {
        return true;
    }
}

void TreeNode::print_node() {
    cout<< "id:" << id << ", value:" << value << ", reward:" << reward << endl;
}

// find max value index in an array
int find_max_index(float *arr, int total) {
    float maxv = -INF;
    int index = -1;
    for (int i = 0; i < total; i++) {
        if (arr[i] > maxv) {
            maxv = arr[i];
            index = i;
        }
    } 
    return index;
}

/*
 * MCT Class
*/
MCT::MCT(
    int _n, 
    int _m, 
    int _action_type, 
    int _max_num, 
    int _object_num, 
    int _state_size
) {
    nnum = 1;
    action_type = _action_type;
    object_num = _object_num;
    max_num = _max_num;
    state_size = _state_size;
    action_size = action_type * max_num;
    
    // init root node
    root = new TreeNode(state_size);
    Env *env = new Env(_n, _m, max_num);
    root -> env = env;
    root -> id = nnum++;
    root -> faid = root -> id;
    root -> action = 0;

    nodedict[root -> id] = root;
}

MCT::~MCT() {
    delete_nodes();
    nodedict.clear();
}

void MCT::delete_nodes() {
    queue<int> q;
    
    q.push(root -> id);

    while (!q.empty()) {
        TreeNode *node = nodedict.at(q.front());
        for (unordered_map<int, int>::iterator iter = node -> childs.begin(); iter != node -> childs.end(); iter++) {
            q.push(iter -> second);
        }
        q.pop();

        delete node;
    }
}

void MCT::initialize_root(
    vector<_int_pair> &_pos, 
    vector<_int_pair> &_target, 
    vector<vector<_int_pair>> &_shape, 
    vector<int> &_cstate, 
    vector<int> &_tstate, 
    vector<_int_pair> &_wall
) {
    for (int i = 0; i < state_size; i++) {
        root -> h_state[i] = 0;
        root -> c_state[i] = 0;
    }

    root -> env -> set_map(_pos, _target, _shape, _cstate, _tstate, _wall, object_num);
}

tuple<bool, int, float, int> MCT::selection(
    int id, 
    pybind11::array_t<float>& prob, 
    float C,
    int step
) {
    pybind11::buffer_info buf = prob.request();
    float* ptr = (float*)buf.ptr;

    TreeNode *node;
    try {
        node = nodedict.at(id);
    } catch (const exception& e) {
        throw "Invalid id";
    }
    if (node -> reward > 10) { // the node has successed
        return make_tuple(true, -1, -1, 1);
    }

    float CC = C;
    // int v_a = 0;
    // for (int i = 0; i < object_num; i++) {
    //     if ((node -> env -> pos[i]).first != (node -> env -> target[i]).first) v_a += 1;
    //     if ((node -> env -> pos[i]).second != (node -> env -> target[i]).second) v_a += 1;
    //     if (node -> env -> cstate[i] != node -> env -> tstate[i]) v_a += 1;
    // }
    // float CC = 2.0 - (2.0 - C) * (v_a / (object_num * 3));

    float exp_tradeoff = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    float explore_prob = 0.1 + (0.9 - 0.1) * exp(-0.01 * step);
    if (step == -1) explore_prob = 1;
    if (step == -2) explore_prob = 0;
    if (explore_prob > exp_tradeoff) {
        float total = 0;
        for (int i = 0; i < object_num * action_type; i++) {
            ptr[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            total += ptr[i];
        }

        for (int i = 0; i < object_num * action_type; i++) {
            ptr[i] /= total;
        }
    }

    float node_times = (node -> times) + 1.0;
    // compute score for every action according to UCT
    float *score = new float[object_num * action_type];
    for (int i = 0; i < object_num * action_type; i++) {
        float n_s_a = 0;
        // float qv = 0;

        if (node -> has_child(i)) {
            int child_id = (node -> childs).at(i);
            n_s_a = (nodedict.at(child_id) -> times) + 1;
            // qv = (nodedict.at(child_id) -> reward) + 0.95 * (nodedict.at(child_id) -> value);
        } else {
            n_s_a = 0 + 1;
            // qv = 0;
        }

        // score[i] = (ptr[i] + 0.02 * qv) * (1 / log(n_s_a + 1)) + CC * sqrt(log(node_times) / n_s_a);
        score[i] = (ptr[i]) * (1 / log(n_s_a + 1)) + CC * sqrt(log(node_times) / n_s_a);
    }

    Env *env = new Env(node -> env);
    // find the action to expand
    for (int i = 0; i < object_num * action_type; i++) {
        int a = find_max_index(score, object_num * action_type);
        if (node -> has_child(a)) { // the action has been expanded
            delete[] score;
            delete env;
            return make_tuple(false, a, -1, 0);
        }

        // the action has not been expanded, then check if the action can be executed 
        int item = a / action_type;
        int action = a % action_type;

        tuple<float, int> r_d = env -> move(item, action);
        float reward = get<0>(r_d);
        int done = get<1>(r_d);
        
        if (done == -1) { // the action can't be executed, change next
            score[a] = -INF;
            continue;
        } else { // the action can be executed, just return
            delete[] score;
            delete env;
            return make_tuple(true, a, reward, done);
        }
    }

    delete[] score;
    delete env;
    // cout << "error" << endl;
    throw "Unexpected result.";
    return make_tuple(false, -1, -1, -1);
}

int MCT::expansion(int id, int action, float reward) {
    TreeNode *node = nodedict.at(id);
    TreeNode *child = new TreeNode(state_size);

    Env *env = new Env(node -> env);
    child -> env = env;
    
    int i = action / action_type;
    int a = action % action_type;
    tuple<float, int> r_d = child -> env -> move(i, a);
    int done = get<1>(r_d);
    if (done == -1) throw "error, can't expansion.";

    child -> action = action;
    child -> faid = id;
    child -> id = nnum++;
    child -> reward = reward;
    child -> depth = node -> depth + 1;
    node -> childs[action] = child -> id;
    nodedict[child -> id] = child;
    return child -> id;
}

void MCT::backpropagation(int id) {
    TreeNode *node = nodedict.at(id);
    while (node -> faid != id) {
        node -> times += 1; // update node times
        int action = node -> action;
        TreeNode *fa = nodedict.at(node -> faid);

        if (fa -> best == -1) { // if father node has no child, this child will be best
            float value = gamma * node -> value + node -> reward;
            fa -> best = action;
            fa -> value = value;
        } else {
            if (fa -> best == action) { // if last best is the child, then re-compute best value
                float max_val = -INF;
                int best = -1;

                for(unordered_map<int, int>::iterator iter = fa -> childs.begin(); iter != fa -> childs.end(); iter++) {
                    TreeNode *tn = nodedict.at(iter -> second);
                    float value = gamma * tn -> value + tn -> reward;
                    if (value > max_val) {
                        best = iter -> first;
                        max_val = value;
                    }
                }

                fa -> best = best;
                fa -> value = max_val;
            } else { // if last best is not the child, then compare with it
                float value = gamma * node -> value + node -> reward;
                if (value > fa -> value) {
                    fa -> best = action;
                    fa -> value = value;
                }
            }
        }

        id = node -> faid; // update id and node, trace back
        node = nodedict.at(id);
    }

    node -> times += 1; // times of root node need to increase
}

bool MCT::nextstep(int action) {
    if (!(root -> has_child(action))) { // unexpected
        cout << "No such son!" << endl;
        return false;
    }

    vector<int> todel;
    queue<int> nodes;
    nodes.push(root -> id);

    int child_id = (root -> childs).at(action);
    root = nodedict.at(child_id);
    root -> faid = root -> id;
    while (!nodes.empty()) { // find all node need to be deleted
        TreeNode *node = nodedict.at(nodes.front());
        todel.push_back(node -> id);
        for(unordered_map<int, int>::iterator iter = node -> childs.begin(); iter != node -> childs.end(); iter++) {
            if (iter -> second != child_id)
                nodes.push(iter -> second);
        }
        nodes.pop();
    }

    for (vector<int>::iterator it = todel.begin(); it != todel.end(); it++) {
        TreeNode *node = nodedict.at(*it);
        delete node;

        nodedict.erase(*it);
    }

    return true;
}

void MCT::print_tree() {
    queue<int> q;
    q.push(root -> id);

    while (!q.empty()) {
        int id = q.front();
        q.pop();
        TreeNode *node = nodedict.at(id);
        
        for (int i = 0; i < node -> depth; i++) {
            cout << "  ";
        }
        cout << "id:" << node -> id << " reward:" << node -> reward << " value:" << node -> value << " times:" << node -> times << " best:" << node -> best << " action:" << node -> action << " faid:" << node -> faid << endl;

        for(unordered_map<int, int>::iterator iter = node -> childs.begin(); iter != node -> childs.end(); iter++) {
            q.push(iter -> second);
        }
    }
}

string MCT::get_tree_string() {
    queue<int> q;
    q.push(root -> id);
    string res ("[");
    while (!q.empty()) {
        int id = q.front();
        q.pop();
        TreeNode *node = nodedict.at(id);
        
        res += "{";
        res += "\"id\":" + to_string(node -> id) + ", " + "\"reward\":" + to_string(node -> reward) + ", " + "\"value\":" + to_string(node -> value) + ", " + "\"times\":" + to_string(node -> times) + ", " + "\"best\":" + to_string(node -> best) + ", " + "\"action\":" + to_string(node -> action) + ", " + "\"faid\":" + to_string(node -> faid);
        res += "},";
        
        for(unordered_map<int, int>::iterator iter = node -> childs.begin(); iter != node -> childs.end(); iter++) {
            q.push(iter -> second);
        }
    }
    res.pop_back();
    res += "]";
    return res;
}

float MCT::alpha(int depth, int times) {
    // TODO f(depth, times) -> [0, 1]
    return 1 + 0 * (depth + times);
}

int MCT::get_root_id() {
    return root -> id;
}

int MCT::get_root_best() {
    return root -> best;
}

int MCT::get_node_child_id(int nodeID, int action) {
    TreeNode *node = nodedict.at(nodeID);
    return node -> get_child(action);
}

void MCT::set_node_value(int nodeID, float value) {
    TreeNode *node = nodedict.at(nodeID);
    node -> value = value;
}

float MCT::get_node_value(int nodeID) {
    TreeNode *node = nodedict.at(nodeID);
    return node -> value;
}

int MCT::get_node_action(int nodeID) {
    TreeNode *node = nodedict.at(nodeID);
    return node -> action;
}

float MCT::get_node_reward(int nodeID) {
    TreeNode *node = nodedict.at(nodeID);
    return node -> reward;
}

pybind11::array_t<float> MCT::get_node_image(int nodeID) {
    TreeNode *node = nodedict.at(nodeID);
    return node -> env -> get_image();
}

vector<int> MCT::get_node_finished_tag(int nodeID) {
    TreeNode *node = nodedict.at(nodeID);
    return node -> env -> finished;
}

vector<int> MCT::get_node_cstate(int nodeID) {
    TreeNode *node = nodedict.at(nodeID);
    return node -> env -> cstate;
}

void MCT::set_node_H(pybind11::array_t<float>& h_state_, int nodeID) {
    TreeNode *node = nodedict.at(nodeID);

    pybind11::buffer_info buf = h_state_.request();
    float* ptr = (float*)buf.ptr;
    memcpy(node -> h_state, ptr, state_size * sizeof(float));
}

pybind11::array_t<float> MCT::get_node_H(int nodeID) {
    TreeNode *node = nodedict.at(nodeID);

    pybind11::array_t<float> result = pybind11::array_t<float>(state_size);
    pybind11::buffer_info buf_result = result.request();
    float* ptr_result = (float*)buf_result.ptr;

    memcpy(ptr_result, node -> h_state, state_size * sizeof(float));
    return result;
}

void MCT::set_node_C(pybind11::array_t<float>& c_state_, int nodeID) {
    TreeNode *node = nodedict.at(nodeID);

    pybind11::buffer_info buf = c_state_.request();
    float* ptr = (float*)buf.ptr;
    memcpy(node -> c_state, ptr, state_size * sizeof(float));
}

pybind11::array_t<float> MCT::get_node_C(int nodeID) {
    TreeNode *node = nodedict.at(nodeID);

    pybind11::array_t<float> result = pybind11::array_t<float>(state_size);
    pybind11::buffer_info buf_result = result.request();
    float* ptr_result = (float*)buf_result.ptr;

    memcpy(ptr_result, node -> c_state, state_size * sizeof(float));
    return result;
}

pybind11::array_t<int> MCT::get_node_map(int nodeID) {
    TreeNode *node = nodedict.at(nodeID);
    return node -> env -> get_map();
}

vector<vector<_int_pair>> MCT::get_node_shape(int nodeID) {
    TreeNode *node = nodedict.at(nodeID);
    return node -> env -> shape;
}

vector<_int_pair> MCT::get_node_wall(int nodeID) {
    TreeNode *node = nodedict.at(nodeID);
    return node -> env -> wall;
}

int MCT::get_node_best(int nodeID) {
    TreeNode *node = nodedict.at(nodeID);
    return node -> best;
}

vector<_int_pair> MCT::get_node_pos(int nodeID) {
    TreeNode *node = nodedict.at(nodeID);
    return node -> env -> pos;
}

vector<_int_pair> MCT::get_node_target(int nodeID) {
    TreeNode *node = nodedict.at(nodeID);
    return node -> env -> target;
}

float MCT::get_node_heuristic_value(int nodeID) {
    TreeNode *node = nodedict.at(nodeID);

    int step = 0;
    float value = 0;

    int bin = node -> env -> bin;
    for (int i = 0; i < object_num; i++) {
        int cstate = node -> env -> cstate[i];
        int tstate = node -> env -> tstate[i];
        _int_pair pos = node -> env -> pos[i];
        _int_pair target = node -> env -> target[i];

        int dis = min((bin - cstate + tstate) % bin, (bin + cstate - tstate) % bin) + (fabs(pos.first - target.first) + fabs(pos.second - target.second));
        step += dis;

        if (dis > 0) {
            float t = pow(gamma, step - 1);
            value += (1 - t) / (1 - gamma);

            if (i == object_num - 1) {
                value += t * (3 + 50);
            } else {
                value += t * 3;
            }
        }
    }

    return value;
}

int MCT::get_object_num() {
    return object_num;
}


vector<tuple<int, float>> MCT::getKNodeIDAlpha(int k){
    vector<tuple<int, float>> res;
    queue<int> q;
    priority_queue<pair<int, int>> pq;

    q.push(root -> id);
    int best_action = root -> best;
    if (best_action == -1) return res;
    int best_child_id = (root -> childs).at(best_action);
    while (!q.empty()) {
        TreeNode *node = nodedict.at(q.front());
        pq.push(make_pair(node -> times, node -> id));
        for (unordered_map<int, int>::iterator iter = node -> childs.begin(); iter != node -> childs.end(); iter++) {
            if (iter -> second != best_child_id)
                q.push(iter -> second);
        }
        q.pop();
    }

    while (k-- && !pq.empty()) {
        int _id = pq.top().second;
        pq.pop();

        TreeNode *node = nodedict.at(_id);
        res.push_back(make_tuple(_id, alpha(node -> depth, node -> times)));
    }

    return res;
}

vector<float> MCT::get_node_successor_value_array(int nodeID) {
    vector<float> res;
    
    TreeNode *node = nodedict.at(nodeID);

    for (int i = 0; i < object_num * action_type; i++) {
        if (node -> has_child(i)) {
            int child_id = (node -> childs).at(i);
            // res.push_back(nodedict.at(child_id) -> value);
            TreeNode *child_node = nodedict.at(child_id);
            res.push_back(gamma * child_node -> value + child_node -> reward);
        } else {
            res.push_back(-1e9);
        }
    }

    return res;
}