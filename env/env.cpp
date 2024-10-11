#include <cmath>
#include <algorithm>
#include <thread>
#include <pthread.h>
#include <iostream>
#include <vector>
#include <utility>
#include <cstring>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "hpp/env.hpp"

using namespace std;

Env::Env(int _n, int _m, int _maxn) {
    n = _n;
    m = _m;
    maxn = _maxn;

    map = new int[n * m];
    target_map = new int[n * m];

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            map[i*m + j] = 0;
            target_map[i*m + j] = 0;
        }
    }
}

Env::Env(const Env &x) {
    n = x.n;
    m = x.m;
    maxn = x.maxn;
    obj_num = x.obj_num;

    map = new int[n * m];
    target_map = new int[n * m];
    memcpy(map, x.map, n * m * sizeof(int));
    memcpy(target_map, x.target_map, n * m * sizeof(int));

    pos = x.pos;
    target = x.target;
    shape = x.shape;
    wall = x.wall;
    cstate = x.cstate;
    tstate = x.tstate;
    finished = x.finished;
    state_dict = x.state_dict;
}

Env::Env(const Env *x) {
    n = x -> n;
    m = x -> m;
    maxn = x -> maxn;
    obj_num = x -> obj_num;

    map = new int[n * m];
    target_map = new int[n * m];
    memcpy(map, x -> map, n * m * sizeof(int));
    memcpy(target_map, x -> target_map, n * m * sizeof(int));

    pos = x -> pos;
    target = x -> target;
    shape = x -> shape;
    wall = x -> wall;
    cstate = x -> cstate;
    tstate = x -> tstate;
    finished = x -> finished;
    state_dict = x -> state_dict;
}


Env::~Env() {
    delete map;
    delete target_map;
}

Env Env::copy() {
    Env env = Env(this);
    return env;
}

void Env::set_map(
    vector<_int_pair> &_pos, 
    vector<_int_pair> &_target, 
    vector<vector<_int_pair>> &_shape, 
    vector<int> &_cstate, 
    vector<int> &_tstate, 
    vector<_int_pair> &_wall, 
    int _obj_num
) {
    pos = _pos;
    target = _target;
    shape = _shape;
    cstate = _cstate;
    tstate = _tstate;
    wall = _wall;
    obj_num = _obj_num;

    for (_int_pair p: wall) { // set wall for the environment
        int x = p.first;
        int y = p.second;
        map[x * m + y] = 1;
        target_map[x * m + y] = 1;
    }

    // conflict detection, remove shape pixel if the position is occupied
    for (int i = 0; i < obj_num; i++) {
        vector<_int_pair> tmp_list;

        _int_pair position = pos[i];
        int x = position.first;
        int y = position.second;
        int s = cstate[i];
        float radian = 2 * pi * s / bin;
        tuple<vector<_float_pair>, _float_pair, _float_pair> res = rotate(shape[i], radian);
        vector<_float_pair> points = get<0>(res);
        _float_pair minp = get<1>(res);
        _float_pair maxp = get<2>(res);

        for (unsigned j = 0; j < points.size(); j++) {
            float xx = points[j].first;
            float yy = points[j].second;
            int tx = int(x + xx + EPS);
            int ty = int(y + yy + EPS);

            if (tx < 0 || tx > n - 1 || ty < 0 || ty > m - 1) {
                tmp_list.push_back(make_pair(shape[i][j].first, shape[i][j].second));
                continue;
            }

            int idx = int(tx) * m + int(ty);

            if (map[idx] >= 1 && map[idx] != i + 2) {
                tmp_list.push_back(make_pair(shape[i][j].first, shape[i][j].second));
            } else {
                map[idx] = i + 2;
            }

            if (map[idx] == 1) map[idx] = 0;
            if (target_map[idx] == 1) target_map[idx] = 0;
        }

        position = target[i];
        x = position.first;
        y = position.second;
        s = tstate[i];
        radian = 2 * pi * s / bin;

        res = rotate(shape[i], radian);
        points = get<0>(res);
        minp = get<1>(res);
        maxp = get<2>(res);

        for (unsigned j = 0; j < points.size(); j++) {
            float xx = points[j].first;
            float yy = points[j].second;
            float tx = int(x + xx + EPS);
            float ty = int(y + yy + EPS);

            if (tx < 0 || tx > n - 1 || ty < 0 || ty > m - 1) {
                bool flag = false;
                for (unsigned k = 0; k < tmp_list.size(); k++) {
                    if (tmp_list[k].first == shape[i][j].first && tmp_list[k].second == shape[i][j].second) {
                        flag = true;
                        break;
                    }
                }
                if (!flag) tmp_list.push_back(make_pair(shape[i][j].first, shape[i][j].second));
                continue;
            }

            int idx = int(tx) * m + int(ty);
            if (target_map[idx] >= 1 && target_map[idx] != i + 2) {
                bool flag = false;
                
                for (unsigned k = 0; k < tmp_list.size(); k++) {
                    if (tmp_list[k].first == shape[i][j].first && tmp_list[k].second == shape[i][j].second) {
                        flag = true;
                        break;
                    }
                }
                if (!flag) tmp_list.push_back(make_pair(shape[i][j].first, shape[i][j].second));
            } else {
                target_map[idx] = i + 2;
            }

            if (map[idx] == 1) map[idx] = 0;
            if (target_map[idx] == 1) target_map[idx] = 0;
        }

        for (unsigned j = 0; j < tmp_list.size(); j++) {
            int idx = -1;
            for (unsigned l = 0; l < shape[i].size(); l++) {
                if (shape[i][l].first == tmp_list[j].first && shape[i][l].second == tmp_list[j].second) {
                    idx = l;
                    break;
                }
            }
            shape[i].erase(shape[i].begin() + idx);
        }
    }

    // finished
    for (int i = 0; i < maxn; i++) {
        if (i < obj_num) {
            if (pos[i].first == target[i].first && pos[i].second == target[i].second && cstate[i] == tstate[i]) {
                finished.push_back(1);
            } else {
                finished.push_back(0);
            }
        } else {
            finished.push_back(1);
        }
    }

    long hash_ = hash();
    state_dict[hash_] = 1;
}

pybind11::array_t<int> Env::get_map() {
    pybind11::array_t<int> result = pybind11::array_t<int>(n * m);
    pybind11::buffer_info buf_result = result.request();
    int *ptr_result = (int*)buf_result.ptr;
    memcpy(ptr_result, map, n * m * sizeof(int));
    result.resize({n,m});
    return result;
}

pybind11::array_t<int> Env::get_target_map() {
    pybind11::array_t<int> result = pybind11::array_t<int>(n * m);
    pybind11::buffer_info buf_result = result.request();
    int *ptr_result = (int*)buf_result.ptr;
    memcpy(ptr_result, target_map, n * m * sizeof(int));
    result.resize({n,m});
    return result;
}

tuple<vector<_float_pair>, _float_pair, _float_pair> Env::rotate(
    vector<_int_pair> &points, 
    float radian
) {
    float minx = INF;
    float miny = INF;
    float maxx = -INF;
    float maxy = -INF;
    
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

    _float_pair minp = make_pair(INF, INF);
    _float_pair maxp = make_pair(-INF, -INF);
    
    for (auto p: res) {
        if (minp.first > p.first) minp.first = p.first;
        if (minp.second > p.second) minp.second = p.second;
        if (maxp.first < p.first) maxp.first = p.first;
        if (maxp.second < p.second) maxp.second = p.second;
    }

    delete opoints;
    return make_tuple(res, minp, maxp);
}

long Env::hash() {
    long mod = 1000000007;
    long num = maxn + 1;
    long total = 0;
    for (int x = 0; x < n; x++) {
        for (int y = 0; y < m; y++) {
            total *= num;
            total += map[x * m + y];
            total %= mod;
        }
    }
    return total;
}

bool Env::translate_step(
    int index, 
    int action, 
    vector<_float_pair> &points, 
    _float_pair &minp, 
    _float_pair &maxp
) {
    int xx = pos[index].first;
    int yy = pos[index].second;

    float xmin = minp.first;
    float ymin = minp.second;
    float xmax = maxp.first;
    float ymax = maxp.second;

    int x = xx + _x[action];
    int y = yy + _y[action];

    // shape bound is in outside of map
    if (x + xmin < 0 || x + xmax >= n || y + ymin < 0 || y + ymax >= m) return false;

    for (unsigned i = 0; i < points.size(); i++) {
        // check every pixel of shape
        int v = map[min(int(x + points[i].first + EPS), n - 1) * m + min(int(y + points[i].second + EPS), m - 1)];
        // the pixel in new position is conflicting
        if (v != 0 && v != index + 2) return false;
    }

    return true;
}

tuple<bool, vector<_float_pair>, _float_pair, _float_pair> Env::rotate_step(
    int index, 
    int action // action must be 0 or 1
) {
    int step_state;
    int _cstate = cstate[index];
    if (action == 0) {
        step_state = (_cstate - 1 + bin) % bin;
    } else {
        step_state = (_cstate + 1) % bin;
    }

    tuple<vector<_float_pair>, _float_pair, _float_pair> res = get_object(index, step_state);
    vector<_float_pair> points = get<0>(res);
    _float_pair minp = get<1>(res);
    _float_pair maxp = get<2>(res);

    int xx = pos[index].first;
    int yy = pos[index].second;

    if (xx + minp.first < 0 || xx + maxp.first >= n || yy + minp.second < 0 || yy + maxp.second >= m) {
        return make_tuple(false, points, minp, maxp);
    }

    for (unsigned i = 0; i < points.size(); i++) {
        int idx = min(int(xx + points[i].first + EPS), n - 1) * m +  min(int(yy + points[i].second + EPS), m - 1);
        int v = map[idx];

        if (v != 0 && v != index + 2) {
            return make_tuple(false, points, minp, maxp);
        }
    }

    return make_tuple(true, points, minp, maxp);
}

tuple<float, int> Env::move(
    int index, 
    int action
) {
    // if (index >= maxn) return make_tuple(-500, -1);
    // if (shape[index].size() == 0) return make_tuple(-500, -1);
    if (index >= maxn) throw "Object index is greater than maxn!!!";
    if (shape[index].size() == 0) throw "Shape has no pixels!!!";
    
    float base = -1;
    
    _int_pair _pos = pos[index];
    _int_pair _target = target[index];
    int _cstate = cstate[index];
    int _tstate = tstate[index];

    tuple<vector<_float_pair>, _float_pair, _float_pair> res = get_object(index, _cstate);
    vector<_float_pair> points = get<0>(res);
    _float_pair minp = get<1>(res);
    _float_pair maxp = get<2>(res);

    if (action < 4) {
        int xx = _pos.first;
        int yy = _pos.second;

        bool step = translate_step(index, action, points, minp, maxp);

        int x = xx + _x[action];
        int y = yy + _y[action];

        if (!step) {
            return make_tuple(-500, -1);
        } else {
            // if leave the target position, reward will be -4
            if (xx == _target.first && yy == _target.second && _cstate == _tstate) base += -4;

            // clear last position
            for (unsigned i = 0; i < points.size(); i++) {
                int idx = min(int(xx + points[i].first + EPS), n - 1) * m + min(int(yy + points[i].second + EPS), m - 1);
                map[idx] = 0;
            }

            // set current position
            for (unsigned i = 0; i < points.size(); i++) {
                int idx = min(int(x + points[i].first + EPS), n - 1) * m + min(int(y + points[i].second + EPS), m - 1);
                map[idx] = index + 2;
            }

            // update position
            pos[index] = make_pair(x, y);

            long hash_ = hash();
            if (state_dict.find(hash_) == state_dict.end()) {
                state_dict[hash_] = 1; // instert the hash value to state_dict
            } else {
                int penlty = state_dict[hash_];
                state_dict[hash_] += 1; // state_dict[hash_] increase
                penlty = 1;
                base = base - 2 * penlty;
            }

            if (pos[index].first == _target.first && pos[index].second == _target.second && cstate[index] == _tstate) {
                if (finished[index] == 1) { // arrive target again
                    base += 2;
                } else {
                    finished[index] = 1; // first arrive target
                    base += 4;
                }

                // check if all objects have arrived
                bool ff = true;
                for (unsigned i = 0; i < pos.size(); i++) {
                    if (fabs(pos[i].first - target[i].first) + fabs(pos[i].second - target[i].second) > 1e-6 || cstate[i] != tstate[i]) {
                        ff = false;
                        break;
                    }
                }

                if (ff) {
                    return make_tuple(base + 50, 1);
                } else {
                    return make_tuple(base, 0);
                }
            } else {
                float mr = (fabs(x - _target.first) + fabs(y - _target.second)) - (fabs(xx - _target.first) + fabs(yy - _target.second));
                return make_tuple(base - mr + 1, 0); // reward from distance
            }
        }
    } else {
        action -= 4;

        int xx = _pos.first;
        int yy = _pos.second;

        tuple<bool, vector<_float_pair>, _float_pair, _float_pair> r_res = rotate_step(index, action);
        bool r_step = get<0>(r_res);
        vector<_float_pair> r_points = get<1>(r_res);
        // _float_pair r_minp = get<2>(r_res);
        // _float_pair r_maxp = get<3>(r_res);

        if (!r_step) {
            return make_tuple(-500, -1);
        } else {
            int step_state;
            if (action == 0) {
                step_state = (_cstate - 1 + bin) % bin;
            } else {
                step_state = (_cstate + 1) % bin;
            }

            // if leave the target position, reward will be -4
            if (xx == _target.first && yy == _target.second && _cstate == _tstate) base += -4;

            // clear last position
            for (unsigned i = 0; i < points.size(); i++) {
                int idx = min(int(xx + points[i].first + EPS), n - 1) * m + min(int(yy + points[i].second + EPS), m - 1);
                map[idx] = 0;
            }

            // set current position
            for (unsigned i = 0; i < r_points.size(); i++) {
                float tx = fmax(0, fmin(xx + r_points[i].first, n - 1));
                float ty = fmax(0, fmin(yy + r_points[i].second, m - 1));
                int idx = int(tx) * m + int(ty);
                map[idx] = index + 2;
            }

            cstate[index] = step_state;

            long hash_ = hash();
            if (state_dict.find(hash_) == state_dict.end()) {
                state_dict[hash_] = 1;
            } else {
                int penlty = state_dict[hash_];
                state_dict[hash_] += 1;
                penlty = 1;
                base = base - 2 * penlty;
            }

            if (pos[index].first == _target.first && pos[index].second == _target.second && cstate[index] == _tstate) {
                if (finished[index] == 1) {
                    base += 2;
                } else {
                    finished[index] = 1;
                    base += 4;
                }

                bool ff = true;
                for (unsigned i = 0; i < pos.size(); i++) {
                    if (fabs(pos[i].first - target[i].first) + fabs(pos[i].second - target[i].second) > 1e-6 || cstate[i] != tstate[i]) {
                        ff = false;
                        break;
                    }
                }

                if (ff) {
                    return make_tuple(base + 50, 1);
                } else {
                    return make_tuple(base, 0);
                }
            } else {
                float mr = min((bin - step_state + _tstate) % bin, (bin + step_state - _tstate) % bin) - min((bin - _cstate + _tstate) % bin, (bin + _cstate - _tstate) % bin);
                return make_tuple(base - mr + 1, 0);
            }
        }
    }
}

pybind11::array_t<float> Env::get_image() {
    pybind11::array_t<float> result = pybind11::array_t<float>((maxn * 2 + 1) * n * m);
    pybind11::buffer_info buf_result = result.request();

    float *ptr_result = (float*)buf_result.ptr;
    memset(ptr_result, 0, (maxn * 2 + 1) * n * m * sizeof(float));

    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < m; j++) {
    //         ptr_result[i * m + j] = map[i * m + j] == 1 ? 1 : 0;
    //     }
    // }

    // for (int k = 0; k < obj_num; k++) {
    //     for (int i = 0; i < n; i++) {
    //         for (int j = 0; j < m; j++) {
    //             int idx1 = ((1 + 2 * k) * n * m) + i * m + j;
    //             int idx2 = ((1 + 2 * k + 1) * n * m) + i * m + j;
    //             ptr_result[idx1] = map[i * m + j] == (k + 2) ? 1 : 0;
    //             ptr_result[idx2] = target_map[i * m + j] == (k + 2) ? 1 : 0;
    //         }
    //     }
    // }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            int cor = i * m + j;
            if (map[cor] == 0) {
                continue;
            }
            else if (map[cor] == 1) {
                ptr_result[cor] = 1; // wall image
                continue;
            } else {
                int k = map[cor] - 2; // object k current map image
                int idx = ((1 + 2 * k) * n * m) + i * m + j;
                ptr_result[idx] = 1;
            }
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            int cor = i * m + j;
            if (target_map[cor] == 0 || target_map[cor] == 1) {
                continue;
            } else {
                int k = target_map[cor] - 2; // object k target map image
                int idx = ((1 + 2 * k + 1) * n * m) + i * m + j;
                ptr_result[idx] = 1;
            }
        }
    }

    result.resize({maxn * 2 + 1, n, m}); // with shape <C, W, H> = <maxn*2+1, n, m>
    return result;
}

tuple<vector<_float_pair>, _float_pair, _float_pair> Env::get_object(int index, int state) {
    vector<_int_pair> opoints = shape[index];
    float radian = 2 * pi * state / bin;

    return rotate(opoints, radian);
}

vector<_int_pair> Env::get_pos() {
    return pos;
}

vector<_int_pair> Env::get_target() {
    return target;
}

vector<int> Env::get_cstate() {
    return cstate;
}

vector<int> Env::get_tstate() {
    return tstate;
}

vector<int> Env::get_finished() {
    return finished;
}

int Env::get_obj_num() {
    return obj_num;
}

vector<_int_pair> Env::get_wall() {
    return wall;
}

vector<vector<_int_pair>> Env::get_shape() {
    return shape;
}

void Env::print_map() {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            cout << map[i * m + j] << " ";
        }
        cout << endl;
    }
}