#ifndef MCTS_H
#define MCTS_H

#include <iostream>
#include <queue>
#include <cstring>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <algorithm>
#include <thread>
#include <pthread.h>

#include "env.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using namespace std;

class TreeNode{
public:
    int id;                         // node id
    int times;                      // node visit times
    float reward;                   // reward from father to the node
    float value;                    // value
    int faid;                       // father id
    int action;                     // action from father to the node
    int best;
    int depth;                      // depth in MCT
    int state_size;                 // hidden state size
    unordered_map<int, int> childs; // child nodes set
    Env *env;
    float *h_state;
    float *c_state;

    TreeNode(int _state_size = 512);
    ~TreeNode();

    // add one child node to the node
    void add_child(int action, int child);

    // get the child id by action
    int get_child(int action);

    // judge child exist or not by action
    bool has_child(int action);

    // print node information
    void print_node();
};

class MCT{
public:
    TreeNode* root;
    unordered_map<int, TreeNode*> nodedict;
    int nnum;
    int action_type;
    int action_size;
    int object_num;
    int max_num;
    int state_size;

    MCT(
        int _n, 
        int _m, 
        int _action_type, 
        int _max_num, 
        int _object_num, 
        int _state_size
    );
    ~MCT();

    // delete all nodes in MCT
    void delete_nodes();

    // initialize root node, input data is required by ENV
    void initialize_root(
        vector<_int_pair> &_pos, 
        vector<_int_pair> &_target, 
        vector<vector<_int_pair>> &_shape, 
        vector<int> &_cstate, 
        vector<int> &_tstate, 
        vector<_int_pair> &_wall
    );

    // selection operation, return is <find or not, action, reward, done>
    tuple<bool, int, float, int> selection(
        int id, 
        pybind11::array_t<float>& prob, 
        float C,
        int step
    );
    
    // expansion operation, return is expanded node id
    int expansion(
        int id, 
        int action, 
        float reward
    );

    // backpropagation operation
    void backpropagation(int id);

    // next step, root node moves down
    bool nextstep(int action);

    // print MCT
    void print_tree();
    string get_tree_string();
    
    // compute alpha value
    float alpha(int depth, int times);

    // get root node id
    int get_root_id();

    // get root best action
    int get_root_best();

    // get child id from action
    int get_node_child_id(int nodeID, int action);
    
    void set_node_value(int nodeID, float value);

    float get_node_value(int nodeID);
    
    int get_node_action(int nodeID);
    
    float get_node_reward(int nodeID);

    pybind11::array_t<float> get_node_image(int nodeID);

    vector<int> get_node_finished_tag(int nodeID);
    
    vector<int> get_node_cstate(int nodeID);

    void set_node_H(pybind11::array_t<float>& h_state_, int nodeID);
    
    pybind11::array_t<float> get_node_H(int nodeID);
    
    void set_node_C(pybind11::array_t<float>& c_state_, int nodeID);
    
    pybind11::array_t<float> get_node_C(int nodeID);

    pybind11::array_t<int> get_node_map(int nodeID);

    vector<vector<_int_pair>> get_node_shape(int nodeID);

    vector<_int_pair> get_node_wall(int nodeID);

    int get_node_best(int nodeID);

    vector<_int_pair> get_node_pos(int nodeID);

    vector<_int_pair> get_node_target(int nodeID);

    float get_node_heuristic_value(int nodeID);

    int get_object_num();


    vector<tuple<int, float>> getKNodeIDAlpha(int k);

    vector<float> get_node_successor_value_array(int nodeID);
};

#endif