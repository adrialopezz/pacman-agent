# team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).

import random
import util
from heapq import heappush, heappop
import time
from math import log, sqrt
from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point

"""
Enhanced Adaptive Agent with Particle Filtering, A*, and MCTS
- Particle filtering: 200 particles/opponent for ghost tracking
- A* pathfinding: Safety-aware costs with caching
- MCTS: 40 simulations with UCB1 for tactical defense
- Dynamic role switching and anti-stuck mechanisms
"""

def create_team(first_index, second_index, is_red, first='AdaptiveAgent', second='AdaptiveAgent', num_training=0):
    """Create team with particle filtering, A*, and MCTS agents."""
    return [eval(first)(first_index), eval(second)(second_index)]


class ReflexCaptureAgent(CaptureAgent):
    """Base class with shared utilities: A* pathfinding, map analysis, safety checks."""

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.boundary_positions = []
        self.path_cache = {}

    def register_initial_state(self, game_state):
        """Initialize agent: compute boundary positions."""
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        self.boundary_positions = self.get_boundary_positions(game_state)

    def get_boundary_positions(self, game_state):
        """Get non-wall positions on our side of boundary."""
        layout = game_state.data.layout
        boundary_x = layout.width // 2 - (0 if self.red else 1)
        return [(boundary_x, y) for y in range(layout.height) if not game_state.has_wall(boundary_x, y)]

    def get_closest_boundary(self, position, game_state):
        """Find closest boundary position for retreating."""
        return min(self.boundary_positions, key=lambda x: self.get_maze_distance(position, x)) if self.boundary_positions else position

    def get_neighbors(self, game_state, pos):
        """Return non-wall 4-neighbors."""
        x, y = pos
        return [(int(x+dx), int(y+dy)) for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)] 
                if not game_state.has_wall(int(x+dx), int(y+dy))]

    def is_tunnel_cell(self, game_state, pos):
        """Check if position is chokepoint (≤2 neighbors)."""
        return len(self.get_neighbors(game_state, pos)) <= 2

    def closest_enemy_ghost_distance(self, game_state, pos):
        """Distance to closest active ghost, None if none visible."""
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [e for e in enemies if not e.is_pacman and e.get_position() and e.scared_timer <= 0]
        return min([self.get_maze_distance(pos, g.get_position()) for g in ghosts]) if ghosts else None

    def astar_search_improved(self, game_state, start, goal, avoid_ghosts=True, ghost_penalty=50, max_nodes=500):
        """A* with parent pointers, safety costs, and caching."""
        ghost_positions = tuple(sorted(self.get_active_ghost_positions(game_state, avoid_ghosts)))
        cache_key = (start, goal, avoid_ghosts, ghost_positions)
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        capsule_set = set(self.get_capsules(game_state))
        tunnel_set = {(x, y) for x in range(game_state.data.layout.width) 
                      for y in range(game_state.data.layout.height)
                      if not game_state.has_wall(x, y) and self.is_tunnel_cell(game_state, (x, y))}
        
        def step_cost(pos):
            cost = 1 + (5 if pos in tunnel_set else 0) - (10 if pos in capsule_set else 0)
            for gx, gy in ghost_positions:
                if abs(pos[0] - gx) + abs(pos[1] - gy) <= 4:
                    return cost + ghost_penalty
            return max(1, cost)
        
        frontier = [(abs(start[0] - goal[0]) + abs(start[1] - goal[1]), 0, start)]
        g_costs, parents, visited = {start: 0}, {start: None}, set()
        nodes_expanded = 0
        
        while frontier and nodes_expanded < max_nodes:
            f_cost, g_cost, current = heappop(frontier)
            
            if current == goal:
                path = []
                while current:
                    path.append(current)
                    current = parents[current]
                result = (list(reversed(path)), g_cost)
                self.path_cache[cache_key] = result
                return result
            
            if current in visited:
                continue
            visited.add(current)
            nodes_expanded += 1
            
            for neighbor in self.get_neighbors(game_state, current):
                if neighbor in visited:
                    continue
                new_g = g_cost + step_cost(neighbor)
                if neighbor not in g_costs or new_g < g_costs[neighbor]:
                    g_costs[neighbor] = new_g
                    parents[neighbor] = current
                    heappush(frontier, (new_g + abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1]), new_g, neighbor))
        
        return (None, float('inf'))
    
    def get_active_ghost_positions(self, game_state, avoid_ghosts=True):
        """Get active (non-scared) ghost positions."""
        if not avoid_ghosts:
            return []
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        return [e.get_position() for e in enemies if not e.is_pacman and e.get_position() and e.scared_timer <= 0]

    def manhattan_distance(self, pos1, pos2):
        """Manhattan distance."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_strategic_positions(self, game_state):
        """Get chokepoints and key boundary positions for defense."""
        strategic = [pos for pos in self.boundary_positions if len(self.get_neighbors(game_state, pos)) <= 2]
        if self.boundary_positions:
            n = len(self.boundary_positions)
            strategic.extend([self.boundary_positions[n//4], self.boundary_positions[n//2], self.boundary_positions[3*n//4]])
        seen = set()
        unique = [pos for pos in strategic if not (pos in seen or seen.add(pos))]
        return unique[:5] if unique else self.boundary_positions[:5]

    def get_safe_action_towards(self, game_state, target, use_astar=True):
        """Get action toward target using A* or greedy fallback."""
        my_pos = game_state.get_agent_state(self.index).get_position()
        actions = game_state.get_legal_actions(self.index)
        if not actions:
            return Directions.STOP
            
        if use_astar:
            path, _ = self.astar_search_improved(game_state, my_pos, target)
            if path and len(path) > 1:
                dx, dy = path[1][0] - my_pos[0], path[1][1] - my_pos[1]
                action_map = {(1,0): Directions.EAST, (-1,0): Directions.WEST, 
                             (0,1): Directions.NORTH, (0,-1): Directions.SOUTH}
                target_action = action_map.get((dx, dy))
                if target_action in actions:
                    return target_action
        
        return min(actions, key=lambda a: self.get_maze_distance(
            self.get_successor(game_state, a).get_agent_state(self.index).get_position(), target))

    def choose_action(self, game_state):
        """Base action selection (override in subclasses)."""
        actions = game_state.get_legal_actions(self.index)
        if not actions:
            return Directions.STOP
        values = [self.evaluate(game_state, a) for a in actions]
        return random.choice([a for a, v in zip(actions, values) if v == max(values)])

    def get_successor(self, game_state, action):
        """Get next state, handling partial grid positions."""
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        return successor if pos == nearest_point(pos) else successor.generate_successor(self.index, action)

    def evaluate(self, game_state, action):
        """Linear combination of features and weights."""
        return self.get_features(game_state, action) * self.get_weights(game_state, action)

    def get_features(self, game_state, action):
        """Base features (override in subclasses)."""
        features = util.Counter()
        features['successor_score'] = self.get_score(self.get_successor(game_state, action))
        return features

    def get_weights(self, game_state, action):
        """Base weights (override in subclasses)."""
        return {'successor_score': 1.0}


class AdaptiveAgent(ReflexCaptureAgent):
    """Enhanced agent with particle filtering, MCTS, A*, anti-stuck, and dynamic role switching."""
    
    team_roles, team_targets = {}, {}
    
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.patrol_index, self.patrol_positions = 0, []
        self.stuck_counter, self.last_positions, self.role_switch_cooldown = 0, [], 0
        self.particles, self.num_particles, self.belief_cache = {}, 200, {}
        self._legal_positions_cache = None
        self.mcts_simulation_budget, self.mcts_exploration_constant, self.mcts_enabled = 40, 1.41, True

    def register_initial_state(self, game_state):
        """Initialize team coordination, patrol, particle filter."""
        super().register_initial_state(game_state)
        team_indices = self.get_team(game_state)
        AdaptiveAgent.team_roles = {idx: 'offense' for idx in team_indices}
        AdaptiveAgent.team_targets = {idx: [] for idx in team_indices}
        self.patrol_positions = self.get_strategic_positions(game_state)
        self.init_particle_filter(game_state)
        self._legal_positions_cache = self.get_legal_positions(game_state)

    def should_play_offense(self, game_state):
        """Smart role switching with cooldown and priority logic."""
        my_state = game_state.get_agent_state(self.index)
        
        if self.role_switch_cooldown > 0:
            self.role_switch_cooldown -= 1
            return AdaptiveAgent.team_roles.get(self.index, 'offense') == 'offense'
        
        team_indices = self.get_team(game_state)
        teammate_idx = [i for i in team_indices if i != self.index][0]
        teammate_role = AdaptiveAgent.team_roles.get(teammate_idx, 'offense')
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [e for e in enemies if e.is_pacman and e.get_position()]
        ghosts = [e for e in enemies if not e.is_pacman and e.get_position()]
        
        if my_state.num_carrying > 0 or any(g.scared_timer > 5 for g in ghosts):
            return True
        if len(invaders) >= 2 and teammate_role != 'defense':
            self.role_switch_cooldown = 3
            return False
        if len(invaders) == 1 and teammate_role == 'offense':
            my_pos = game_state.get_agent_state(self.index).get_position()
            teammate_pos = game_state.get_agent_state(teammate_idx).get_position()
            if my_pos and teammate_pos:
                inv_pos = invaders[0].get_position()
                if self.get_maze_distance(my_pos, inv_pos) < self.get_maze_distance(teammate_pos, inv_pos):
                    self.role_switch_cooldown = 2
                    return False
        
        if self.get_score(game_state) < -5:
            return True
        return teammate_role != 'offense' or len(invaders) > 0

    def is_stuck(self, game_state):
        """Detect stuck only when defending without active threats."""
        my_pos = game_state.get_agent_state(self.index).get_position()
        if not my_pos:
            return False
        
        my_state = game_state.get_agent_state(self.index)
        if my_state.is_pacman:
            self.stuck_counter = 0
            return False
        
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [e for e in enemies if e.is_pacman and e.get_position()]
        
        if any(self.get_maze_distance(my_pos, inv.get_position()) <= 4 for inv in invaders):
            self.stuck_counter = 0
            return False
        
        self.last_positions.append(my_pos)
        if len(self.last_positions) > 10:
            self.last_positions.pop(0)
        
        if len(self.last_positions) >= 8:
            x_coords, y_coords = [p[0] for p in self.last_positions], [p[1] for p in self.last_positions]
            if max(x_coords) - min(x_coords) <= 2 and max(y_coords) - min(y_coords) <= 2:
                self.stuck_counter += 1
                return self.stuck_counter >= 4
        
        self.stuck_counter = 0
        return False

    def get_unstuck_action(self, game_state):
        """Force exploration to break stuck patterns."""
        actions = game_state.get_legal_actions(self.index)
        if not actions:
            return Directions.STOP
        
        my_pos = game_state.get_agent_state(self.index).get_position()
        scores = {}
        
        for action in actions:
            successor = self.get_successor(game_state, action)
            next_pos = successor.get_agent_state(self.index).get_position()
            score = 5 - sum(10 - i for i, old_pos in enumerate(self.last_positions) if next_pos == old_pos)
            
            if not self.should_play_offense(game_state) and self.patrol_positions:
                closest_strategic = min(self.patrol_positions, key=lambda p: self.get_maze_distance(next_pos, p))
                score += max(0, 6 - self.get_maze_distance(next_pos, closest_strategic))
            
            scores[action] = score
        
        return max(scores, key=scores.get)

    def get_features(self, game_state, action):
        """Extract features for both offensive and defensive play."""
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        
        playing_offense = self.should_play_offense(game_state)
        AdaptiveAgent.team_roles[self.index] = 'offense' if playing_offense else 'defense'
        
        features['successor_score'] = self.get_score(successor)
        
        # Offensive features
        food_list = self.get_food(successor).as_list()
        if food_list:
            features['distance_to_food'] = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['food_count'] = -len(food_list)
        
        features['carrying'] = my_state.num_carrying
        if my_state.num_carrying > 0:
            features['distance_to_home'] = self.get_maze_distance(my_pos, self.get_closest_boundary(my_pos, successor))
        
        capsules = self.get_capsules(successor)
        if capsules:
            features['distance_to_capsule'] = min([self.get_maze_distance(my_pos, cap) for cap in capsules])
        
        # Defensive features
        features['on_defense'] = 0 if my_state.is_pacman else 1
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [e for e in enemies if e.is_pacman and e.get_position()]
        features['num_invaders'] = len(invaders)
        
        if invaders:
            features['invader_distance'] = min([self.get_maze_distance(my_pos, inv.get_position()) for inv in invaders])
        elif not playing_offense and self.patrol_positions:
            patrol_target = self.patrol_positions[self.patrol_index % len(self.patrol_positions)]
            features['patrol_distance'] = self.get_maze_distance(my_pos, patrol_target)
        
        # Safety features
        ghosts = [e for e in enemies if not e.is_pacman and e.get_position()]
        if ghosts:
            scared_ghosts = [g for g in ghosts if g.scared_timer > 0]
            active_ghosts = [g for g in ghosts if g.scared_timer <= 0]
            
            if active_ghosts:
                active_dists = [self.get_maze_distance(my_pos, g.get_position()) for g in active_ghosts]
                features['ghost_distance'] = min(active_dists)
                if min(active_dists) <= 2 and my_state.is_pacman:
                    features['in_danger'] = 1
            
            if scared_ghosts:
                features['scared_ghost_distance'] = min([self.get_maze_distance(my_pos, g.get_position()) for g in scared_ghosts])
        
        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1
        
        return features

    def get_weights(self, game_state, action):
        """Role-based dynamic weights."""
        my_state = game_state.get_agent_state(self.index)
        playing_offense = self.should_play_offense(game_state)
        
        if playing_offense:
            weights = {
                'successor_score': 10, 'food_count': 100, 'distance_to_food': -3,
                'carrying': 0, 'distance_to_home': 0, 'distance_to_capsule': -2,
                'on_defense': 0, 'num_invaders': -20, 'invader_distance': 0, 'patrol_distance': 0,
                'ghost_distance': 4, 'scared_ghost_distance': -6, 'in_danger': -3000,
                'stop': -100, 'reverse': -10
            }
            
            my_pos = game_state.get_agent_state(self.index).get_position()
            if my_pos:
                close_ghost = self.closest_enemy_ghost_distance(game_state, my_pos)
                should_retreat = ((close_ghost and close_ghost <= 4 and my_state.num_carrying >= 3) or my_state.num_carrying >= 6)
                if should_retreat:
                    weights.update({'distance_to_home': -15, 'distance_to_food': -0.5, 'carrying': 5})
        else:
            weights = {
                'successor_score': 5, 'food_count': 0, 'distance_to_food': 0,
                'carrying': 0, 'distance_to_home': 0, 'distance_to_capsule': 0,
                'on_defense': 200, 'num_invaders': -2000, 'invader_distance': -15,
                'patrol_distance': -2, 'ghost_distance': 0, 'scared_ghost_distance': 0,
                'in_danger': 0, 'stop': -150, 'reverse': -5
            }
        
        return weights

    def choose_action(self, game_state):
        """Enhanced action selection with particle filter, MCTS, and anti-stuck."""
        start_time = time.time()
        
        # Particle filter update
        self.observe_opponents(game_state)
        self.elapse_time_particles(game_state)
        
        # Retreat check
        if self.should_retreat_from_ghosts(game_state):
            my_pos = game_state.get_agent_state(self.index).get_position()
            if my_pos:
                return self.get_safe_action_towards(game_state, self.get_closest_boundary(my_pos, game_state))
        
        # MCTS for defense
        playing_defense = not self.should_play_offense(game_state)
        if self.mcts_enabled and playing_defense and (time.time() - start_time) < 0.5:
            enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
            invaders = [e for e in enemies if e.is_pacman and e.get_position()]
            
            if invaders:
                remaining_time = 0.85 - (time.time() - start_time)
                action = self.mcts_search(game_state, time_limit=remaining_time)
                if action != Directions.STOP:
                    return action
        
        # Reflex action selection
        if self.role_switch_cooldown > 0:
            self.role_switch_cooldown -= 1
        
        if self.is_stuck(game_state):
            return self.get_unstuck_action(game_state)
        
        actions = game_state.get_legal_actions(self.index)
        if not actions:
            return Directions.STOP
        
        values = [self.evaluate(game_state, action) for action in actions]
        best_actions = [action for action, value in zip(actions, values) if value == max(values)]
        
        food_left = len(self.get_food(game_state).as_list())
        if food_left <= 2:
            my_pos = game_state.get_agent_state(self.index).get_position()
            if my_pos:
                return self.get_safe_action_towards(game_state, self.get_closest_boundary(my_pos, game_state))
        
        action = random.choice(best_actions) if best_actions else Directions.STOP
        
        # Patrol update
        if not self.should_play_offense(game_state) and self.patrol_positions:
            my_pos = game_state.get_agent_state(self.index).get_position()
            if my_pos:
                patrol_target = self.patrol_positions[self.patrol_index % len(self.patrol_positions)]
                if self.get_maze_distance(my_pos, patrol_target) <= 2:
                    self.patrol_index = (self.patrol_index + 1) % len(self.patrol_positions)
                    
                    enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
                    invaders = [e for e in enemies if e.is_pacman and e.get_position()]
                    
                    if not invaders and self.patrol_index % 3 == 0:
                        food_list = self.get_food(game_state).as_list()
                        if food_list:
                            nearby_food = [f for f in food_list if self.get_maze_distance(my_pos, f) <= 12]
                            if nearby_food:
                                closest_food = min(nearby_food, key=lambda f: self.get_maze_distance(my_pos, f))
                                return self.get_safe_action_towards(game_state, closest_food)
        
        return action

    # Particle filtering methods
    def get_legal_positions(self, game_state):
        """Get all non-wall positions (cached)."""
        if self._legal_positions_cache:
            return self._legal_positions_cache
        return [(x, y) for x in range(game_state.data.layout.width)
                for y in range(game_state.data.layout.height)
                if not game_state.has_wall(x, y)]

    def init_particle_filter(self, game_state):
        """Initialize particles for each opponent."""
        legal = self.get_legal_positions(game_state)
        for opponent in self.get_opponents(game_state):
            self.particles[opponent] = random.choices(legal, k=self.num_particles)
            self.belief_cache[opponent] = None

    def observe_opponents(self, game_state):
        """Update particles based on noisy distances."""
        noisy_distances = game_state.get_agent_distances()
        my_pos = game_state.get_agent_state(self.index).get_position()
        
        for opponent in self.get_opponents(game_state):
            opponent_pos = game_state.get_agent_state(opponent).get_position()
            
            if opponent_pos:
                self.particles[opponent] = [opponent_pos] * self.num_particles
                self.belief_cache[opponent] = None
                continue
            
            noisy_dist = noisy_distances[opponent]
            if noisy_dist is None:
                continue
            
            weights = [game_state.get_distance_prob(self.manhattan_distance(my_pos, p), noisy_dist)
                      for p in self.particles[opponent]]
            
            if sum(weights) > 0:
                self.particles[opponent] = random.choices(self.particles[opponent], weights=weights, k=self.num_particles)
                self.belief_cache[opponent] = None
            else:
                legal = self.get_legal_positions(game_state)
                self.particles[opponent] = random.choices(legal, k=self.num_particles)

    def elapse_time_particles(self, game_state):
        """Predict particle movement (random walk)."""
        for opponent in self.get_opponents(game_state):
            new_particles = []
            for particle in self.particles[opponent]:
                neighbors = self.get_neighbors(game_state, particle)
                new_particles.append(random.choice(neighbors) if neighbors else particle)
            self.particles[opponent] = new_particles
            self.belief_cache[opponent] = None

    def get_most_likely_ghost_position(self, opponent_index):
        """Get most likely position for opponent."""
        beliefs = util.Counter()
        for particle in self.particles[opponent_index]:
            beliefs[particle] += 1
        return beliefs.arg_max() if beliefs else None

    def should_retreat_from_ghosts(self, game_state):
        """Decide if retreat needed based on tracked ghosts."""
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        
        if not my_state.is_pacman or my_state.num_carrying == 0:
            return False
        
        for opponent in self.get_opponents(game_state):
            opponent_state = game_state.get_agent_state(opponent)
            
            if opponent_state.scared_timer > 0:
                continue
            
            ghost_pos = opponent_state.get_position() if opponent_state.get_position() else self.get_most_likely_ghost_position(opponent)
            
            if ghost_pos and self.get_maze_distance(my_pos, ghost_pos) <= 5:
                return True
        
        return False

    # MCTS methods
    def mcts_search(self, game_state, time_limit=0.8):
        """Run MCTS search for defensive decisions."""
        start_time = time.time()
        root = MCTSNode(game_state, agent_index=self.index)
        
        simulations = 0
        while simulations < self.mcts_simulation_budget and (time.time() - start_time) < time_limit:
            node = root
            while node.is_fully_expanded() and node.children:
                node = node.best_child(self.mcts_exploration_constant)
            
            if not node.game_state.is_over() and not node.is_fully_expanded():
                node = node.expand()
            
            reward = self.mcts_simulate(node.game_state)
            node.update(reward)
            simulations += 1
        
        if root.children:
            return max(root.children.values(), key=lambda c: c.visits).action
        
        legal = game_state.get_legal_actions(self.index)
        return random.choice(legal) if legal else Directions.STOP

    def mcts_simulate(self, game_state, max_depth=6):
        """Fast greedy rollout."""
        state, depth = game_state, 0
        
        while depth < max_depth and not state.is_over():
            actions = state.get_legal_actions(self.index)
            if not actions:
                break
            action = max(actions, key=lambda a: self.evaluate(state, a))
            state = state.generate_successor(self.index, action)
            depth += 1
        
        return self.evaluate(state, Directions.STOP)

class MCTSNode:
    """MCTS Node with dictionary-based children for O(1) lookup."""
    
    def __init__(self, game_state, parent=None, action=None, agent_index=None):
        self.game_state = game_state
        self.parent = parent
        self.action = action
        self.agent_index = agent_index
        self.visits = 0
        self.value = 0.0
        self.untried_actions = None
        self.children = {}

    def get_untried_actions(self):
        """Lazy initialize untried actions."""
        if self.untried_actions is None:
            actions = self.game_state.get_legal_actions(self.agent_index)
            self.untried_actions = [a for a in actions if a != Directions.STOP]
        return self.untried_actions

    def is_fully_expanded(self):
        """Check if all actions tried."""
        return len(self.get_untried_actions()) == 0

    def best_child(self, exploration_weight=1.41):
        """UCB1 selection."""
        def ucb_value(child):
            if child.visits == 0:
                return float('inf')
            exploit = child.value / child.visits
            explore = exploration_weight * sqrt(log(self.visits) / child.visits)
            return exploit + explore
        
        return max(self.children.values(), key=ucb_value)

    def expand(self):
        """Expand node by trying untried action."""
        untried = self.get_untried_actions()
        if not untried:
            return self
        
        action = random.choice(untried)
        self.untried_actions.remove(action)
        
        next_state = self.game_state.generate_successor(self.agent_index, action)
        child_node = MCTSNode(next_state, parent=self, action=action, agent_index=self.agent_index)
        self.children[action] = child_node
        return child_node

    def update(self, reward):
        """Backpropagate reward."""
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.update(reward)

