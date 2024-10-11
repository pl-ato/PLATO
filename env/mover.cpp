#include "hpp/mcts.hpp"
#include "hpp/env.hpp"
#include <cstdio>
#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace std;

vector<_float_pair> rotate_(
    vector<_int_pair> &points,
    float radian
) {
    float minx = 1e9;
    float miny = 1e9;
    float maxx = -1e9;
    float maxy = -1e9;
    
    int pn = points.size();
    float *opoints = new float[pn * 2];

    for (int i = 0; i < pn; i++) {
        opoints[i * 2 + 0] = points[i].first + 0.5; // actual float position, position <0, 0> is <0.5, 0.5> 
        opoints[i * 2 + 1] = points[i].second + 0.5;

        if (minx > opoints[i * 2 + 0]) minx = opoints[i * 2 + 0];
        if (miny > opoints[i * 2 + 1]) miny = opoints[i * 2 + 1];
        if (maxx < opoints[i * 2 + 0]) maxx = opoints[i * 2 + 0];
        if (maxy < opoints[i * 2 + 1]) maxy = opoints[i * 2 + 1];
    }

    // move center to origin
    for (int i = 0; i < pn; i++) {
        opoints[i * 2 + 0] -= 0.5 * (maxx + minx);
        opoints[i * 2 + 1] -= 0.5 * (maxy + minx);
    }

    // rotation matrix
    float rot[2][2] = {{cos(radian), -sin(radian)}, {sin(radian), cos(radian)}};

    vector<_float_pair> res;
    for (int i = 0; i < pn; i++) {
        float nx = rot[0][0] * opoints[i * 2 + 0] + rot[0][1] * opoints[i * 2 + 1];
        float ny = rot[1][0] * opoints[i * 2 + 0] + rot[1][1] * opoints[i * 2 + 1];
        nx += 0.5 * (maxx + minx) + EPS; // move to origin center
        ny += 0.5 * (maxy + miny) + EPS;
        res.push_back(make_pair(nx, ny));
    }

    delete opoints;
    return res;
}

vector<_int_pair> get_object_(
    int x,
    int y,
    vector<_int_pair> &shape,
    float orientation
) {
    float radian = 2 * pi * orientation / 24;
    
    vector<_float_pair> points = rotate_(shape, radian);
    vector<_int_pair> res;
    int pn = points.size();
    for (int i = 0; i < pn; i++) {
        int tx = int(x + points[i].first + EPS);
        int ty = int(y + points[i].second + EPS);
        res.push_back(make_pair(tx, ty));
    }
    return res;
}

PYBIND11_MODULE( mover, m ){
    m.doc() = "pybind11 MOVER";

    // m.def( "rotate", &rotate_ )
    //  .def( "get_object", &get_object_ );
    m.def( "get_object", &get_object_ );


    pybind11::class_<Env>(m, "Env")
        .def( pybind11::init< int, int, int >() )

        .def( "copy", &Env::copy )
        .def( "set_map", &Env::set_map )
        .def( "get_map", &Env::get_map )
        .def( "get_target_map", &Env::get_target_map )
        .def( "hash", &Env::hash )
        .def( "move", &Env::move )
        .def( "get_image", &Env::get_image )
        .def( "get_pos", &Env::get_pos )
        .def( "get_target", &Env::get_target )
        .def( "get_cstate", &Env::get_cstate )
        .def( "get_tstate", &Env::get_tstate )
        .def( "get_finished_tag", &Env::get_finished )
        .def( "get_obj_num", &Env::get_obj_num )
        .def( "get_wall", &Env::get_wall )
        .def( "get_shape", &Env::get_shape )
        .def( "print_map", &Env::print_map );
    
    
    pybind11::class_<MCT>(m, "MCT")
        .def( pybind11::init< int, int, int, int, int, int >() )
        
        .def( "initialize_root", &MCT::initialize_root )
        .def( "selection", &MCT::selection )
        .def( "expansion", &MCT::expansion )
        .def( "backpropagation", &MCT::backpropagation )
        .def( "nextstep", &MCT::nextstep )
        .def( "print_tree", &MCT::print_tree )
        .def( "get_tree_string", &MCT::get_tree_string )
        .def( "get_root_id", &MCT::get_root_id )
        .def( "get_root_best", &MCT::get_root_best )
        .def( "get_node_child_id", &MCT::get_node_child_id )
        .def( "set_node_value", &MCT::set_node_value )
        .def( "get_node_value", &MCT::get_node_value )
        .def( "get_node_action", &MCT::get_node_action )
        .def( "get_node_reward", &MCT::get_node_reward )
        .def( "get_node_image", &MCT::get_node_image )
        .def( "get_node_finished_tag", &MCT::get_node_finished_tag )
        .def( "get_node_cstate", &MCT::get_node_cstate )
        .def( "set_node_H", &MCT::set_node_H )
        .def( "get_node_H", &MCT::get_node_H )
        .def( "set_node_C", &MCT::set_node_C )
        .def( "get_node_C", &MCT::get_node_C )
        .def( "get_node_map", &MCT::get_node_map )
        .def( "get_node_shape", &MCT::get_node_shape )
        .def ("get_node_wall", &MCT::get_node_wall )
        .def( "get_node_best", &MCT::get_node_best )
        .def( "get_node_pos", &MCT::get_node_pos )
        .def( "get_node_target", &MCT::get_node_target )
        .def( "get_node_heuristic_value", &MCT::get_node_heuristic_value )
        .def( "get_object_num", &MCT::get_object_num )
        .def( "get_node_successor_value_array", &MCT::get_node_successor_value_array )
        .def( "get_knode_id_alpha_pair", &MCT::getKNodeIDAlpha );
}