#ifndef ENV_H
#define ENV_H

#include <cmath>
#include <algorithm>
#include <thread>
#include <pthread.h>
#include <iostream>
#include <vector>
#include <utility>
#include <unordered_map>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using namespace std;

#define pi 3.1415926
#define EPS 1e-6
#define INF 1e9

typedef pair<int, int> _int_pair;
typedef pair<float, float> _float_pair;

class Env{
public:
    Env(int _n, int _m, int _maxn);
    Env(const Env &x);
    Env(const Env *x);
    ~Env();

    const int bin = 24;                 // rotation bin
    const int _x[4] = {-1, 1, 0, 0};
    const int _y[4] = {0, 0, -1, 1}; 

    int *map;                           // current map matrix
    int *target_map;                    // target map matrix
    int n;                              // row size
    int m;                              // column size
    int maxn;                           // max object number
    int obj_num;                        // actual object number

    vector<_int_pair> pos;              // object current position list
    vector<_int_pair> target;           // object target position list
    vector<vector<_int_pair>> shape;    // object pixel list
    vector<_int_pair> wall;             // wall position list
    vector<int> cstate;                 // object current state list
    vector<int> tstate;                 // object target state list
    vector<int> finished;               // object finish tag
    unordered_map<long, int> state_dict;// hash dict of states

    // copy function, interface for python
    Env copy();
    
    // init env
    void set_map(
        vector<_int_pair> &_pos,
        vector<_int_pair> &_target,
        vector<vector<_int_pair>> &_shape,
        vector<int> &_cstate,
        vector<int> &_tstate,
        vector<_int_pair> &_wall,
        int _obj_num
    );
    
    // get current map, an numpy array with shape <n, m>
    pybind11::array_t<int> get_map();
    
    // get target map, an numpy array with shape <n, m>
    pybind11::array_t<int> get_target_map();
        
    // rotate function
    tuple<vector<_float_pair>, _float_pair, _float_pair> rotate(
        vector<_int_pair> &points,
        float radian
    );
    
    // compute hash value of current value
    long hash();
    
    // translation action, return true if action is executable
    bool translate_step(
        int index,
        int action,
        vector<_float_pair> &points,
        _float_pair &minp,
        _float_pair &maxp
    );
    
    // rotation action, return a tuple <executable, shape, minp, maxp>
    tuple<bool, vector<_float_pair>, _float_pair, _float_pair> rotate_step(
        int index,
        int action
    );
    
    // move action, a top interface of translation and rotation
    tuple<float, int> move(
        int index, 
        int action
    );
    
    // get current env image with shape <2 * obj_num + 1, n, m>, result can be used as input of neural network
    pybind11::array_t<float> get_image();

    // get an object in specific state
    tuple<vector<_float_pair>, _float_pair, _float_pair> get_object(
        int index, 
        int state
    );

    // get current position list
    vector<_int_pair> get_pos();
    
    // get target position list
    vector<_int_pair> get_target();
    
    // get current state
    vector<int> get_cstate();
    
    // get current state
    vector<int> get_tstate();
    
    // get finish tag
    vector<int> get_finished();
    
    // get object number
    int get_obj_num();

    vector<_int_pair> get_wall();

    vector<vector<_int_pair>> get_shape();

    // print map
    void print_map();


    // vector<int> getcstate();
    // vector<int> getfinishedtag();
    // vector<_int_pair> getpos();
    // vector<_int_pair> gettarget();
    // int getobjnum();

    // tuple<vector<_float_pair>, _float_pair, _float_pair> getitem(int index, int state);

};

#endif