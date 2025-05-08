import json
import pygame
import sys
# Baba is Y'all mechanic/rule base
# Version: 2.0
# Code based on  Milk javascript code

# Assign ascii values to images
map_key = {
    '_': "border",
    ' ': "empty",
    #'.': "empty",
    'b': "baba_obj",
    'B': "baba_word",
    '1': "is_word",
    '2': "you_word",
    '3': "win_word",
    's': "skull_obj",
    'S': "skull_word",
    'f': "flag_obj",
    'F': "flag_word",
    'o': "floor_obj",
    'O': "floor_word",
    'a': "grass_obj",
    'A': "grass_word",
    '4': "kill_word",
    'l': "lava_obj",
    'L': "lava_word",
    '5': "push_word",
    'r': "rock_obj",
    'R': "rock_word",
    '6': "stop_word",
    'w': "wall_obj",
    'W': "wall_word",
    '7': "move_word",
    '8': "hot_word",
    '9': "melt_word",
    'k': "keke_obj",
    'K': "keke_word",
    'g': "goop_obj",
    'G': "goop_word",
    '0': "sink_word",
    'v': "love_obj",
    'V': "love_word"
}

features = ["hot", "melt", "open", "shut", "move"]
featPairs = [["hot", "melt"], ["open", "shut"]]

oppDir = {
    "left": "right",
    "right": "left",
    "up": "down",
    "down": "up"
}


class PhysObj:
    def __init__(self, name, x, y):
        self.type = "phys"
        self.name = name
        self.x = x
        self.y = y
        self.is_movable = False
        self.is_stopped = False
        self.feature = ""
        self.dir = ""


class WordObj:
    def __init__(self, name, obj, x, y):
        self.type = "word"
        self.name = name
        self.x = x
        self.y = y
        self.obj = obj
        self.is_movable = True
        self.is_stopped = False


def reverse_char(val):
    
    return next((key for key, value in map_key.items() if value == val), None)


def assign_map_objs(state):
    # Reset the state
    state['sort_phys'] = {}
    state['phys'] = []
    state['words'] = []
    state['is_connectors'] = []

    # Retrieve state
    m = state['obj_map']
    phys = state['phys']
    words = state['words']
    sort_phys = state['sort_phys']
    is_connectors = state['is_connectors']

    # Check if the map has been made yet
    if len(m) == 0:
        print("ERROR: Map not initialized yet")
        return False

    for r in range(len(m)):
        for c in range(len(m[r])):
            charac = m[r][c]
            
            base_obj = map_key[charac].split("_")[0]
            # Word-based object
            if "word" in map_key[charac]:
                
                w = WordObj(base_obj, None, c, r)
                
                # Add the base object to the word representation if it exists
                if (base_obj + "_obj") in map_key.values():
                    w.obj = base_obj
                    

                # Add any "is" words
                if base_obj == "is":
                    is_connectors.append(w)

                # Replace character on obj_map
                m[r][c] = w

                words.append(w)
            # Physical-based object
            elif "obj" in map_key[charac]:
                o = PhysObj(base_obj, c, r)

                phys.append(o)

                # Replace character on obj_map
                m[r][c] = o

                # Add to the list of objects under a certain name
                if base_obj not in sort_phys:
                    sort_phys[base_obj] = [o]
                else:
                    sort_phys[base_obj].append(o)


def init_empty_map(m):
    return [[' ' for _ in range(len(row))] for row in m]


def split_map(m):
    bm = init_empty_map(m)
    om = init_empty_map(m)
    for r in range(len(m)):
        
        for c in range(len(m[r])):

            charac = m[r][c]
            parts = map_key[charac].split("_")
            if len(parts) == 1:  # background
                bm[r][c] = charac
                om[r][c] = ' '
            else:  # object
                bm[r][c] = ' '
                om[r][c] = charac
    
    return bm, om


def show_map(m):
    str = ""
    for row in m:
        for charac in row:
            if charac != ' ' and charac != '_':
                str += reverse_char(charac)
            else:
                str += '.' if charac == ' ' else '+'
        str += "\n"
    return str


def map_to_str(m):
    str = ""
    for row in m:
        for charac in row:
            str += charac if charac != ' ' else '.'
        str += "\n"
    return str.rstrip("\n")


def double_map_to_str(om, bm):
    str = ""
    for r in range(len(om)):
        for c in range(len(om[0])):
            o = om[r][c]
            b = bm[r][c]
            
            if r == 0 or c == 0 or r == len(om) - 1 or c == len(om[0]) - 1:
                str += "_"
            elif o == ' ' and b == ' ':
                str += "."
            elif o == ' ':
                if b.type =="phys":
                    temp=b.name+"_obj"
                else:
                    temp=b.name+"_word"
                str += reverse_char(temp)
            else:
                if o.type =="phys":
                    temp=o.name+"_obj"
                else:
                    temp=o.name+"_word"
                str += reverse_char(temp)
        str += "\n"
    return str.rstrip("\n")


def parse_map(ms):
    new_map = []
    rows = ms.split("\n")
    for row in rows:
        new_map.append(list(row.replace('.', ' ')))
    return new_map


def parse_map_wh(ms, w, h):
    new_map = []
    for r in range(h):
        row = ms[r * w:(r + 1) * w]
        new_map.append(list(row.replace('.', ' ')))
    return new_map


def obj_at_pos(x, y, om):
    return om[y][x]


def is_word(e):
    return isinstance(e, WordObj)


def is_phys(e):
    return isinstance(e, PhysObj)


def interpret_rules(state):
    # Reset the rules (since a word object was changed)
    state['rules'] = []
    state['rule_objs'] = []

    # Get all the parameters
    om = state['obj_map']
    bm = state['back_map']
    is_connectors = state['is_connectors']
    rules = state['rules']
    rule_objs = state['rule_objs']
    sort_phys = state['sort_phys']
    phys = state['phys']
    words = state['words']
    p = state['players']
    am = state['auto_movers']
    w = state['winnables']
    u = state['pushables']
    #s = state.get('stoppables', [])  # In case 'stoppables' is not present, default to an empty list
    k = state['killers']
    n = state['sinkers']
    o = state['overlaps']
    uo = state['unoverlaps']
    f = state['featured']

    for is_connector in is_connectors:
        # Horizontal position
        word_a = obj_at_pos(is_connector.x - 1, is_connector.y, om)
        word_b = obj_at_pos(is_connector.x + 1, is_connector.y, om)
        if is_word(word_a) and is_word(word_b):
            # Add a new rule if not already made
            r = f"{word_a.name}-{is_connector.name}-{word_b.name}"
            if r not in rules:
                rules.append(r)

            # Add as a rule object
            rule_objs.extend([word_a, is_connector, word_b])

        # Vertical position
        word_c = obj_at_pos(is_connector.x, is_connector.y - 1, om)
        word_d = obj_at_pos(is_connector.x, is_connector.y + 1, om)
        if is_word(word_c) and is_word(word_d):
            r = f"{word_c.name}-{is_connector.name}-{word_d.name}"
            if r not in rules:
                rules.append(r)

            # Add as a rule object
            rule_objs.extend([word_c, is_connector, word_d])

    # Interpret sprite changing rules
    transformation(om, bm, rules, sort_phys, phys)

    # Reset the objects
    reset_all(state)

def clear_level(state):
    props = state.keys()
    for prop in props:
        entry = state[prop]
        if isinstance(entry, list):
            state[prop] = []
        else:
            state[prop] = {}


def set_state_by_map(state, original_ascii):
    clear_level(state)
    state['orig_map'] = original_ascii
    state['back_map'], state['obj_map'] = split_map(state['orig_map'])
    assign_map_objs(state)
    interpret_rules(state)


def new_state(ascii_map):
    s = {
        'orig_map': ascii_map,
        'obj_map': [],
        'back_map': [],
        'words': [],
        'phys': [],
        'is_connectors': [],
        'sort_phys': {},
        'rules': [],
        'rule_objs': [],
        'players': [],
        'auto_movers': [],
        'winnables': [],
        'pushables': [],
        'killers': [],
        'sinkers': [],
        'featured': {},
        'overlaps': [],
        'unoverlaps': []
    }
    s['back_map'], s['obj_map'] = split_map(s['orig_map'])
    assign_map_objs(s)
    interpret_rules(s)
    return s


def show_state(state):
    return double_map_to_str(state['obj_map'], state['back_map'])


def has(arr, ss):
    for item in arr:
        if ss in item:
            return True
    return False


def reset_obj_props(phys):
    for obj in phys:
        obj.is_movable = False
        obj.is_stopped = False
        obj.feature = ""


def reset_all(state):
    reset_obj_props(state['phys'])
    set_players(state)
    set_auto_movers(state)
    set_wins(state)
    set_pushes(state)
    set_stops(state)
    set_kills(state)
    set_sinks(state)
    set_overlaps(state)
    set_features(state)


def set_players(state):
    state['players'] = []
    players = state['players']
    rules = state['rules']
    sort_phys = state['sort_phys']

    for rule in rules:
        if 'you' in rule:
            if rule.split("-")[0] not in sort_phys:
                sort_phys[rule.split("-")[0]]=[]
            players.extend(sort_phys[rule.split("-")[0]])

    for player in players:
        player.is_movable = True


def set_auto_movers(state):
    state['auto_movers'] = []
    automovers = state['auto_movers']
    rules = state['rules']
    sort_phys = state['sort_phys']

    for rule in rules:
        if 'move' in rule:
            if rule.split("-")[0] not in sort_phys:
                sort_phys[rule.split("-")[0]]=[]
            automovers.extend(sort_phys[rule.split("-")[0]])

    for automover in automovers:
        automover.is_movable = True
        if not automover.dir:
            automover.dir = "right"


def set_wins(state):
    state['winnables'] = []
    winnables = state['winnables']
    rules = state['rules']
    sort_phys = state['sort_phys']

    for rule in rules:
        if 'win' in rule:
            if rule.split("-")[0] not in sort_phys:
                sort_phys[rule.split("-")[0]]=[]
            winnables.extend(sort_phys[rule.split("-")[0]])


def set_pushes(state):
    state['pushables'] = []
    pushables = state['pushables']
    rules = state['rules']
    sort_phys = state['sort_phys']

    for rule in rules:
        if 'push' in rule:
            if rule.split("-")[0] not in sort_phys:
                sort_phys[rule.split("-")[0]]=[]
            pushables.extend(sort_phys[rule.split("-")[0]])

    for pushable in pushables:
        pushable.is_movable = True


def set_stops(state):
    state['stoppables'] = []
    stoppables = state['stoppables']
    rules = state['rules']
    sort_phys = state['sort_phys']

    for rule in rules:
        if 'stop' in rule:
            if rule.split("-")[0] not in sort_phys:
                sort_phys[rule.split("-")[0]]=[]
            stoppables.extend(sort_phys[rule.split("-")[0]])

    for stoppable in stoppables:
        stoppable.is_stopped = True


def set_kills(state):
    state['killers'] = []
    killers = state['killers']
    rules = state['rules']
    sort_phys = state['sort_phys']

    for rule in rules:
        if 'kill' in rule or 'sink' in rule:
            if rule.split("-")[0] not in sort_phys:
                sort_phys[rule.split("-")[0]]=[]
            killers.extend(sort_phys[rule.split("-")[0]])


def set_sinks(state):
    state['sinkers'] = []
    sinkers = state['sinkers']
    rules = state['rules']
    sort_phys = state['sort_phys']

    for rule in rules:
        if 'sink' in rule:
            if rule.split("-")[0] not in sort_phys:
                sort_phys[rule.split("-")[0]]=[]
            sinkers.extend(sort_phys[rule.split("-")[0]])


def set_overlaps(state):
    state['overlaps'] = []
    state['unoverlaps'] = []

    bm = state['back_map']
    om = state['obj_map']
    overlaps = state['overlaps']
    unoverlaps = state['unoverlaps']
    rules = state['rules']
    phys = state['phys']
    words = state['words']

    for p in phys:
        if not p.is_movable and not p.is_stopped:
            overlaps.append(p)
            bm[p.y][p.x] = p
            om[p.y][p.x] = ' '
        else:
            unoverlaps.append(p)
            om[p.y][p.x] = p
            bm[p.y][p.x] = ' '

    unoverlaps.extend(words)
    # words will always be in the object layer
    for w in words:
        om[w.y][w.x] = w



def overlapped(a, b):
    return a == b or (a.x == b.x and a.y == b.y)


def killed(players, killers):
    dead = []
    for player in players:
        for killer in killers:
            if overlapped(player, killer):
                dead.append([player,player])
    return dead


def drowned(phys, sinkers):
    dead = []
    for p in phys:
        for sinker in sinkers:
            if p != sinker and overlapped(p, sinker):
                dead.append([p, sinker])
    return dead


def destroy_objs(dead, state):
    bm = state['back_map']
    om = state['obj_map']
    phys = state['phys']
    sort_phys = state['sort_phys']

    for item in dead:
        p = item[0]
        o = item[1]
        print(o,p)
        if o == p:
            phys.remove(p)

            sort_phys[p.name].remove(p)
            
            om[p.y][p.x] = ' '
            bm[o.y][o.x] = ' '
        else:
            phys.remove(p)
            phys.remove(o)
        
            sort_phys[p.name].remove(p)
            sort_phys[o.name].remove(o)

            bm[o.y][o.x] = ' '
            om[p.y][p.x] = ' '

    if dead:
        reset_all(state)


def win(players, winnables):
    for player in players:
        for winnable in winnables:
            if overlapped(player, winnable):
                return True
    return False


def transformation(om, bm, rules, sort_phys, phys):
    # x-is-x takes priority and makes it immutable
    xisx = []
    for rule in rules:
        parts = rule.split("-")
        if parts[0] == parts[2] and parts[0] in sort_phys:
            xisx.append(parts[0])

    # Transform sprites (x-is-y == x becomes y)
    for rule in rules:
        parts = rule.split("-")
        # Turn all objects
        if parts[0] not in xisx and is_obj_word(parts[0]) and is_obj_word(parts[2]):
            all_objs = [obj for obj_list in sort_phys.values() for obj in obj_list if obj.name == parts[0]]
            for obj in all_objs:
                change_sprite(obj, parts[2], om, bm, phys, sort_phys)


def is_obj_word(w):
    return reverse_char(w + "_obj") is not None


def change_sprite(o, w, om, bm, phys, sort_phys):
    charac = reverse_char(w + "_obj")
    o2 = PhysObj(w, o.x, o.y)
    phys.append(o2)  # In with the new...

    # Replace object on obj_map/back_map
    if obj_at_pos(o.x, o.y, om) == o:
        om[o.y][o.x] = o2
    else:
        bm[o.y][o.x] = o2

    # Add to the list of objects under a certain name
    if w not in sort_phys:
        sort_phys[w] = [o2]
    else:
        sort_phys[w].append(o2)

    phys.remove(o)  # ...out with the old
    sort_phys[o.name].remove(o)


def set_features(state):
    state['featured'] = {}
    featured = state['featured']
    rules = state['rules']
    sort_phys = state['sort_phys']

    for rule in rules:
        parts = rule.split("-")
        if parts[2] in features and parts[0] not in features and parts[0] in sort_phys:
            if parts[2] not in featured:
                featured[parts[2]] = [parts[0]]
            else:
                featured[parts[2]].append(parts[0])

            for p in sort_phys[parts[0]]:
                p.feature = parts[2]


def bad_feats(featured, sort_phys):
    baddies = []

    for pair in featPairs:
        if pair[0] in featured and pair[1] in featured:
            a_set = []
            b_set = []

            for feature in featured[pair[0]]:
                a_set.extend(sort_phys[feature])
            for feature in featured[pair[1]]:
                b_set.extend(sort_phys[feature])

            for a in a_set:
                for b in b_set:
                    if overlapped(a, b):
                        baddies.append([a, b])

    return baddies




### game ####
# State and properties for the level/game
game_state = {
    'orig_map': [],
    'obj_map': [],
    'back_map': [],
    'words': [],
    'phys': [],
    'is_connectors': [],
    'sort_phys': {},
    'rules': [],
    'rule_objs': [],
    'players': [],
    'auto_movers': [],
    'winnables': [],
    'pushables': [],
    'killers': [],
    'sinkers': [],
    'featured': {},
    'overlaps': [],
    'unoverlaps': []
    
}

orig_map = []
back_map = []
obj_map = []
mapAllReady = False
curLevel = 0
demo = False

# Sprites
moved = False
acted = False
movedObjs = []

moveSteps = []
initRules = []
endRules = []

wonGame = False
saveLevel = False
demoGame = False

aiControl = False
curPath = []
pathIndex = 0
solving = False
unsolvable = False


# Generic functions
def in_arr(arr, e):
    if len(arr) == 0:
        return False
    return e in arr


# Keyboard functions
def next_move(next_dir, state):
    moved_objects = []
    moved = False
    if next_dir != "space":
        move_players(next_dir, moved_objects, state)

    move_auto_movers(moved_objects, state)

    for obj in moved_objects:
        if obj.type == "word":
            interpret_rules(state)

    won_game = win(state['players'], state['winnables'])
    return state, won_game


def can_move(e, dir, om, bm, moved_objs, p, u, phys, sort_phys):
    if e in moved_objs:
        return False

    if e == " ":
        return False
    if not e.is_movable:
        return False

    o = ' '

    if dir == "up":
        if e.y - 1 < 0:
            return False
        if bm[e.y - 1][e.x] == '_':
            return False
        o = om[e.y - 1][e.x]
    elif dir == "down":
        if e.y + 1 >= len(bm):
            return False
        if bm[e.y + 1][e.x] == '_':
            return False
        o = om[e.y + 1][e.x]
    elif dir == "left":
        if e.x - 1 < 0:
            return False
        if bm[e.y][e.x - 1] == '_':
            return False
        o = om[e.y][e.x - 1]
    elif dir == "right":
        if e.x + 1 >= len(bm[0]):
            return False
        if bm[e.y][e.x + 1] == '_':
            return False
        o = om[e.y][e.x + 1]

    if o == ' ':
        return True
    if o.is_stopped:
        return False
    if o.is_movable:
        if o in u:
            return move_obj(o, dir, om, bm, moved_objs, p, u, phys, sort_phys)
        elif o in p and e not in p:
            return True
        elif o.type == "phys" and (len(u) == 0 or o not in u):
            return False
        elif (e.is_movable or o.is_movable) and (e.type == "phys" and o.type == "phys"):
            return True
        elif e.name == o.name and e in p and is_phys(o) and is_phys(e):
            return move_obj_merge(o, dir, om, bm, moved_objs, p, u, phys, sort_phys)
        else:
            return move_obj(o, dir, om, bm, moved_objs, p, u, phys, sort_phys)
    return True


def move_obj(o, dir, om, bm, moved_objs, p, u, phys, sort_phys):
    if can_move(o, dir, om, bm, moved_objs, p, u, phys, sort_phys):
        om[o.y][o.x] = ' '
        if dir == "up":
            o.y -= 1
        elif dir == "down":
            o.y += 1
        elif dir == "left":
            o.x -= 1
        elif dir == "right":
            o.x += 1
        om[o.y][o.x] = o
        moved_objs.append(o)
        o.dir = dir
        return True
    else:
        return False


def move_obj_merge(o, dir, om, bm, moved_objs, p, u, phys, sort_phys):
    if can_move(o, dir, om, bm, moved_objs, p, u, phys, sort_phys):
        om[o.y][o.x] = ' '
        if dir == "up":
            o.y -= 1
        elif dir == "down":
            o.y += 1
        elif dir == "left":
            o.x -= 1
        elif dir == "right":
            o.x += 1
        om[o.y][o.x] = o
        moved_objs.append(o)
        o.dir = dir
        return True
    else:
        om[o.y][o.x] = ' '
        return True


def move_players(dir, mo, state):
    om = state['obj_map']
    bm = state['back_map']
    players = state['players']
    pushs = state['pushables']
    phys = state['phys']
    sort_phys = state['sort_phys']
    killers = state['killers']
    sinkers = state['sinkers']
    featured = state['featured']
    
    for cur_player in players:
        move_obj(cur_player, dir, om, bm, mo, players, pushs, phys, sort_phys)
    
    destroy_objs(killed(players, killers), state)
    destroy_objs(drowned(sinkers, phys), state)
    destroy_objs(bad_feats(featured, sort_phys), state)


def move_auto_movers(mo, state):
    automovers = state['auto_movers']
    om = state['obj_map']
    bm = state['back_map']
    players = state['players']
    pushs = state['pushables']
    phys = state['phys']
    sort_phys = state['sort_phys']
    killers = state['killers']
    sinkers = state['sinkers']
    featured = state['featured']

    for cur_auto in automovers:
        m = move_obj(cur_auto, cur_auto.dir, om, bm, mo, players, pushs, phys, sort_phys)
        if not m:
            cur_auto.dir = oppDir[cur_auto.dir]

    destroy_objs(killed(players, killers), state)
    destroy_objs(drowned(sinkers, phys), state)
    destroy_objs(bad_feats(featured, sort_phys), state)

            

# Game loop functions
def make_level(map):
    clear_level(game_state)
    global demo
    demo = False
    game_state['orig_map'] = map
    set_level()


def get_cur_rules():
    ruleset = []
    for r in game_state['rules']:
        ruleset.append(r)
    return ruleset


def set_level():
    bm, om = split_map(game_state['orig_map'])
    game_state['back_map'] = bm
    game_state['obj_map'] = om
    assign_map_objs(game_state)
    interpret_rules(game_state)
    end_rules = []
    init_rules = get_cur_rules()
    ai_control = False
    moved = False
    unsolvable = False
    save_level = False
    move_steps = []
    won_game = False
    already_solved = False


def translate_sol(sol_str):
    solution = []
    for p in sol_str:
        if p == "U" or p == "u":
            solution.append("up")
        elif p == "D" or p == "d":
            solution.append("down")
        elif p == "L"or p == "l":
            solution.append("left")
        elif p == "R" or p == "r":
            solution.append("right")
        elif p == "S" or p == "s":
            solution.append("space")
    return solution


    

# Exporting functions
def minimize_solution(solution):
    mini_sol = []
    for s in solution:
        step = s[0] if s != "" else "space"
        mini_sol.append(step.lower())
    return "".join(mini_sol)


def level_to_json(lvl=None, ID=0, name="", author="Baba"):
    if lvl is None:
        lvl = game_state["orig_map"]

    json_level = {
        'id': ID,
        'name': name,
        'author': author,
        'ascii': map_to_str(lvl),
        'solution': minimize_solution(moveSteps)
    }

    return json.dumps(json_level)

# IMPORT LEVEL SETS FROM THE JSON DIRECTORY
def import_level_sets():
    json_dir = 'json_levels/'
    json_files = [file for file in os.listdir(json_dir) if not file.startswith('.')]
    return json_files

import os
# PARSE THE JSON FORMAT OF THE LEVELS
def import_levels(lvlset_json):
    path = 'json_levels/' + lvlset_json + ".json"
    if not os.path.exists(path):
        return None  # no level set found
    
    with open(path, 'r') as file:
        lvlset = json.load(file)
    return lvlset['levels']

# IMPORT THE SET OF LEVELS BY NAME
def get_level_set(name):
    return import_levels(name)

# IMPORT THE LEVEL BY ITS ID NUMBER
def get_level_obj(ls, _id):
    if isinstance(ls, str):
        ls = get_level_set(ls)
    
    if ls is None:
        return None
    
    for lvl in ls:
        if lvl['id'] == _id:
            return lvl
    return None

def load_images():
    images = {}
    for key, value in map_key.items():
        
        path = value+".png"
        images[key] = pygame.image.load(os.path.join('img', path)).convert()
    return images

def draw_game_map(screen, images, game_map_ascii):
    x, y = 0, 0
    tile_size = 50  # Adjust this according to your sprite size
    for line in game_map_ascii.split('\n'):
        for char in line:
            if char in images:
                screen.blit(images[char], (x, y))
            x += tile_size
        x = 0
        y += tile_size



if __name__ == "__main__":
    level = import_levels("demo_LEVELS")
    ascii_level = level[5]['ascii']
    make_level(parse_map(ascii_level))

    state = game_state
    #sol = translate_sol(level[6]['solution'])
    sol = translate_sol("dddruuddrrruuddrlulllu")
    #"""
    pygame.init()
    screen = pygame.display.set_mode((1200, 1000))  # Adjust the window size as needed
    pygame.display.set_caption("Game Map")
    images = load_images()

    running = True
    move_index = 0
    
    while running and (move_index < len(sol)):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("here")
                running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key  == pygame.K_w:
                    move= "up"
                elif event.key == pygame.K_s:
                    move= "down"
                elif event.key ==pygame.K_a:
                    move= "left"
                elif event.key == pygame.K_d:
                    move= "right"
                elif event.key == pygame.K_SPACE:
                    move= "space"
                
                    
                
            # Execute the move and update the game state
                state, won = next_move(move, state)
                if won:
                    print('correct')
                    running=False
            ascii = show_state(state)
            draw_game_map(screen, images, ascii)
            print(show_state(state))
            print(game_state['rules'])
                
            pygame.display.flip()
        # Get the next move
        #move=sol[move_index]
        #state, won = next_move(move, state)
        

        screen.fill((0, 0, 0))

        # Redraw the game map with the updated state
        """"
        ascii = show_state(state)
        draw_game_map(screen, images, ascii)
        print(show_state(state))
        print(game_state['rules'])

        pygame.display.flip()

        # Increment move index
        #move_index += 1
        
        if won:
            print('correct')
            running=False
        """

        # Delay between moves (optional)
        #pygame.time.delay(400)  # Adjust the delay time as needed

        # Winning solution reached
        
            #pygame.quit()

    pygame.quit()

    """
    for move in sol:
        # Iterate over game state
        state,won = next_move(move, state)
        print(show_state(state))
        print(game_state['rules'])

    # Winning solution reached
    if won:
        print('correct')
    """
    

