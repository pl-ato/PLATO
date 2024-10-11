from PIL import Image
import numpy as np
import os

colors = [[182,38,61],[236,100,66],[240,147,61],[246,198,68],[234,222,107],[181,211,109],[118,197,139],[83,183,173],[66,121,152],[61,61,104],[75,27,71],[132,30,64],[0,245,255],[255,218,185],[0,255,127],[102,205,170],[205,198,115],[46,139,87],[107,142,35],[238,180,34],[139,101,8],[205,92,92],[205,133,63],[238,121,66],[250,128,114]]

def convert_map_to_img_array(input_map):
    """ Convert map to image numpy array

    Args:
        input_map: input map
    
    Return:
        Image array
    """
    color_num = len(colors)
    margin_width = 0
    grid_width = 5
    map_size = input_map.shape
    img_size = [map_size[0]*(grid_width + margin_width) + margin_width, map_size[1]*(grid_width + margin_width) + margin_width]
    img = np.ones([*img_size, 3])
    img *= 255
    
    for i in range(map_size[0]):
        for j in range(map_size[1]):
            _id = int(input_map[i,j])
            if _id == 0:
                color = [225, 225, 225]
            elif _id == 1:
                color = [128, 128, 128]
            else:
                color = colors[_id % color_num]
            
            bx = i * (grid_width + margin_width) + margin_width
            by = j * (grid_width + margin_width) + margin_width
            for k in range(grid_width):
                for l in range(grid_width):
                    img[bx + k, by + l] = color
    
    return np.uint8(img)

def generate_sequence_gif(input_maps, path='case.gif'):
    """ Convert map sequences to gif

    Args:
        input_maps: input map sequence with shape <N, h, w>
        path: file save path
    """
    im0 = Image.fromarray(convert_map_to_img_array(input_maps[0]))
    im_others = []
    for m in input_maps[1:]:
        im_others.append(Image.fromarray(convert_map_to_img_array(m)))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    im0.save(path, save_all=True, append_images=im_others, duration=300, loop=0)

def restore_case(save_dir, idx, actions, env):
    sd = os.path.join(save_dir, 'test{:0>5d}'.format(idx))

    os.makedirs(sd, exist_ok=True)

    map_target = Image.fromarray(convert_map_to_img_array(env.get_target_map()))
    map_target.save(os.path.join(sd, 'target.png'))

    map_seq = []
    map_seq.append(env.get_map())
    for action in actions:
        i = int(action / 6)
        a = int(action % 6)
        reward, done = env.move(i, a)
        map_seq.append(env.get_map())
        if done == 1:
            break
        if done == -1:
            raise Exception('Can\'t move.')
    
    print('done:', done, 'step:', len(actions), 'map_size:', len(map_seq))
    
    generate_sequence_gif(np.array(map_seq), os.path.join(sd, 'sequence.gif'))

if __name__ == '__main__':
    # generate_sequence_mp4(None,)
    im = Image.open('./debug_img/root_map1.png')
    im = np.asarray(im)

    ims = np.array([im, im, im])
    generate_sequence_gif(ims)