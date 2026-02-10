import zlib
import time
from dataclasses import asdict
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from clashroyalebuildabot.bot import Bot
from clashroyalebuildabot.constants import DISPLAY_HEIGHT
from clashroyalebuildabot.constants import DISPLAY_WIDTH
from clashroyalebuildabot.constants import N_HEIGHT_TILES
from clashroyalebuildabot.constants import N_WIDE_TILES
from clashroyalebuildabot.constants import TILE_HEIGHT
from clashroyalebuildabot.constants import TILE_INIT_X
from clashroyalebuildabot.constants import TILE_INIT_Y
from clashroyalebuildabot.constants import TILE_WIDTH
from clashroyalebuildabot.namespaces import Screens, Cards, State
from clashroyalebuildabot.namespaces.units import UnitDetection, Position


class ClashRoyaleEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, bot: Bot, play_action_delay=1.0):
        super().__init__()
        self.bot = bot
        self.play_action_delay = play_action_delay
        
        # Action Space: [Card_Index (0-3), Tile_X (0-17), Tile_Y (0-31)]
        self.action_space = spaces.MultiDiscrete([4, 18, 32])

        # Limits
        self.max_units = 32
        self.n_cards = 4
        
        # Observation Space
        # - scalars: 5 features
        # - allies: (MAX_UNITS, 11) -> [UnitID, TX, TY, Conf, BX, BY, BW, BH, Cat, Tgt, Trans]
        # - enemies: (MAX_UNITS, 11) 
        # - hand: (4, 3) -> [CardID, Cost, IsReady]
        self.observation_space = spaces.Dict({
            "scalars": spaces.Box(low=0, high=10000, shape=(5,), dtype=np.float32),
            "allies": spaces.Box(low=0, high=1000000, shape=(self.max_units, 11), dtype=np.float32),
            "enemies": spaces.Box(low=0, high=1000000, shape=(self.max_units, 11), dtype=np.float32),
            "hand": spaces.Box(low=-1, high=100000000, shape=(self.n_cards, 3), dtype=np.float32),
            "screen": spaces.Discrete(len(Screens.__dict__))  # Approximate
        })
        
        self.last_state = None
        self.last_time = time.time()
        
        self.unit_cost_map = self._build_unit_cost_map()
        self.frame_stack_n = 3

    def _build_unit_cost_map(self):
        cost_map = {}
        # Iterate over all cards defined in the Cards namespace
        for card_name, card in asdict(Cards).items():
            # Apply heuristic: If swarms, we might want lower cost per unit.
            # But we don't know count. Just use Card Cost for now.
            # Or assume min cost across all cards spawning this unit.
            for unit in card.units:
                if unit.name not in cost_map:
                    cost_map[unit.name] = card.cost
                else:
                    # Keep the lower cost (conservative estimation for swarms appearing in multiple cards)
                    if card.cost < cost_map[unit.name]:
                        cost_map[unit.name] = card.cost
        
        # Manually adjust known swarms if necessary
        # e.g. 'skeleton': 1 (Skeletons) vs 3 (Skarmy). Min is 1. Reasonable.
        # 'bat': 2 (Bats)
        # 'minion': 3 (Minions) vs 5 (Horde). Min is 3.
        return cost_map

    def _get_unit_cost_val(self, unit_name):
        return self.unit_cost_map.get(unit_name, 3.0) # Default to 3 average if unknown

    def _hash_name(self, name):
        """Stable hash for unit names to integer IDs (0-999)."""
        return zlib.crc32(name.encode()) % 1000

    def _set_stable_state(self):
        """Captures multiple frames and merges detections to reduce noise."""
        states = []
        for _ in range(self.frame_stack_n):
            self.bot.set_state()
            states.append(self.bot.state)
            time.sleep(0.05)
            
        # Merge Units (Spatial Clustering)
        all_allies = [s.allies for s in states]
        all_enemies = [s.enemies for s in states]
        
        merged_allies = self._merge_units(all_allies)
        merged_enemies = self._merge_units(all_enemies)
        
        # Merge Numbers (Median)
        # Using last state as base
        base_s = states[-1]
        
        # We can implement sophisticated median logic for numbers if needed,
        # but for now we just take the last one or maybe average?
        # Let's take the state with max elixir + HP to avoid '0' glitches?
        # Actually median is best for cleaning noise.
        # But Numbers object structure is deep.
        # Simplification: Just use last state's numbers/cards/screen, 
        # but replace units with stable ones.
        
        # Construct new stable state
        # We need to preserve the type of State
        stable_state = State(
            allies=merged_allies,
            enemies=merged_enemies,
            numbers=base_s.numbers,
            cards=base_s.cards,
            ready=base_s.ready,
            screen=base_s.screen
        )
        
        self.bot.state = stable_state

    def _merge_units(self, unit_lists):
        """
        Merge lists of units from multiple frames.
        Returns a single list of UnitDetections.
        """
        flat_units = []
        for ul in unit_lists:
            flat_units.extend(ul)
            
        # Group by unit name
        by_name = {}
        for u in flat_units:
            if u.unit.name not in by_name: by_name[u.unit.name] = []
            by_name[u.unit.name].append(u)
            
        merged = []
        for name, units in by_name.items():
            # Cluster units of same name
            processed = [False] * len(units)
            for i in range(len(units)):
                if processed[i]: continue
                
                cluster = [units[i]]
                processed[i] = True
                
                # Greedy clustering by distance
                base_x, base_y = units[i].position.tile_x, units[i].position.tile_y
                
                for j in range(i+1, len(units)):
                    if processed[j]: continue
                    ux, uy = units[j].position.tile_x, units[j].position.tile_y
                    # Distance in tiles
                    dist = ((base_x - ux)**2 + (base_y - uy)**2) ** 0.5
                    if dist < 2.5: # Threshold: 2.5 tiles radius
                        cluster.append(units[j])
                        processed[j] = True
                
                # Filter: Must appear in majority of frames
                # e.g. if 3 frames, need >= 2 detections
                threshold = (self.frame_stack_n // 2) + 1
                if len(cluster) >= threshold:
                    # Average the cluster
                    avg_x = int(np.mean([u.position.tile_x for u in cluster]))
                    avg_y = int(np.mean([u.position.tile_y for u in cluster]))
                    avg_conf = float(np.mean([u.position.conf for u in cluster]))
                    
                    # Create new Position (bbox is just taken from first)
                    p0 = cluster[0].position
                    new_pos = Position(bbox=p0.bbox, conf=avg_conf, tile_x=avg_x, tile_y=avg_y)
                    new_ud = UnitDetection(unit=cluster[0].unit, position=new_pos)
                    merged.append(new_ud)
                    
        return merged

    def _process_units(self, units_list):
        """Convert list of UnitDetection to fixed-size numpy array."""
        # Features per unit: 
        # 0: UnitID (Hash 0-999)
        # 1: Tile X
        # 2: Tile Y
        # 3: Confidence
        # 4: BBox X
        # 5: BBox Y
        # 6: BBox W
        # 7: BBox H
        # 8: Category (0: None, 1: Troop, 2: Building)
        # 9: Target (0: None, 1: Ground, 2: Buildings, 3: Air, 4: All)
        # 10: Transport (0: None, 1: Ground, 2: Air)
        
        feature_dim = 11
        arr = np.zeros((self.max_units, feature_dim), dtype=np.float32)
        
        for i, u in enumerate(units_list[:self.max_units]):
            # UnitDetection: unit (Unit), position (Position)
            uid = self._hash_name(u.unit.name)
            
            # Position
            tx = u.position.tile_x
            ty = u.position.tile_y
            conf = u.position.conf
            bbox = u.position.bbox if u.position.bbox else (0, 0, 0, 0)
            bx, by, bw, bh = bbox
            
            # Attributes
            cat = 0
            if u.unit.category == 'troop': cat = 1
            elif u.unit.category == 'building': cat = 2
            
            tgt = 0
            if u.unit.target == 'ground': tgt = 1
            elif u.unit.target == 'buildings': tgt = 2
            elif u.unit.target == 'air': tgt = 3
            elif u.unit.target == 'all': tgt = 4
            
            trans = 0
            if u.unit.transport == 'ground': trans = 1
            elif u.unit.transport == 'air': trans = 2
            
            arr[i] = [uid, tx, ty, conf, bx, by, bw, bh, cat, tgt, trans]
            
        return arr

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Ensure we are in game
        if not self.bot.state:
            self.bot.set_state()
            
        while self.bot.state.screen != Screens.IN_GAME:
            if self.bot.state.screen == Screens.LOBBY:
                self.bot.emulator.click(*self.bot.state.screen.click_xy)
                time.sleep(2)
            elif self.bot.state.screen == Screens.END_OF_GAME:
                self.bot.emulator.click(*self.bot.state.screen.click_xy)
                time.sleep(2)
            else:
                # Wait or random click interaction
                time.sleep(1)
            
            self.bot.set_state()
        
        # Stabilize state before returning
        self._set_stable_state()
        self.last_state = self.bot.state

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        card_idx, tile_x, tile_y = action
        
        # Check validity (elixir)
        valid_action = False
        if card_idx < len(self.bot.state.cards):
            card = self.bot.state.cards[card_idx]
            if card_idx in self.bot.state.ready:
                 current_elixir = self.bot.state.numbers.elixir.number
                 if current_elixir >= card.cost:
                     valid_action = True

        if valid_action:
            card_centre = self.bot._get_card_centre(card_idx)
            tile_centre = self.bot._get_tile_centre(tile_x, tile_y)
            self.bot.emulator.click(*card_centre)
            self.bot.emulator.click(*tile_centre)
        
        # Wait for action to have effect
        time.sleep(self.play_action_delay)
        
        # Update State
        self._set_stable_state()
        new_state = self.bot.state
        
        # Calculate Reward
        reward = self._calculate_reward(self.last_state, new_state)
        if not valid_action:
            reward -= 0.1 # Small penalty for invalid action
            
        # Termination
        terminated = new_state.screen == Screens.END_OF_GAME
        truncated = False # Or time limit
        
        self.last_state = new_state
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        # Build observation from self.bot.state
        s = self.bot.state
        nums = s.numbers
        
        # Scalars
        scalars = np.array([
            nums.elixir.number,
            nums.left_ally_princess_hp.number,
            nums.right_ally_princess_hp.number,
            nums.left_enemy_princess_hp.number,
            nums.right_enemy_princess_hp.number
        ], dtype=np.float32)

        # Units
        allies = self._process_units(s.allies)
        enemies = self._process_units(s.enemies)
        
        # Hand / Cards
        # state.cards is a Tuple of 4 Cards.
        # We produce a (4, 3) array: [CardID, Cost, IsReady]
        hand = np.zeros((self.n_cards, 3), dtype=np.float32)
        for i in range(min(len(s.cards), self.n_cards)):
            card = s.cards[i]
            is_ready = 1.0 if i in s.ready else 0.0
            # Use card.id_ if available, else -1 or hash name
            cid = float(card.id_) if hasattr(card, 'id_') else float(self._hash_name(card.name))
            hand[i] = [cid, float(card.cost), is_ready]

        # Screen (simple hash or enum value if possible)
        # s.screen is a Screen object or enum. 
        # Since Screen is likely an enum or class instance comparison
        # We'll just map known screens to ints if possible or return 0
        screen_val = 0 # Default
        # Note: Screens is a namespace with string constants? Or Enum?
        # Checked screens.py earlier? Not deep detail. Assuming it's standard comparison.
        
        return {
            "scalars": scalars,
            "allies": allies,
            "enemies": enemies,
            "hand": hand,
            "screen": 0 # Placeholder if not enumerating screens
        }


    def _get_info(self):
        return {}

    def _calculate_reward(self, old_state, new_state):
        reward = 0
        
        # Scaling factor for HP (to keep rewards in reasonable range, e.g., -1 to 1 per significant event)
        HP_SCALE = 0.01
        
        # HP Differences
        d_ally_left = (new_state.numbers.left_ally_princess_hp.number - old_state.numbers.left_ally_princess_hp.number) * HP_SCALE
        d_ally_right = (new_state.numbers.right_ally_princess_hp.number - old_state.numbers.right_ally_princess_hp.number) * HP_SCALE
        d_enemy_left = (new_state.numbers.left_enemy_princess_hp.number - old_state.numbers.left_enemy_princess_hp.number) * HP_SCALE
        d_enemy_right = (new_state.numbers.right_enemy_princess_hp.number - old_state.numbers.right_enemy_princess_hp.number) * HP_SCALE
        
        # Filter out massive jumps (detection errors or game end reset)
        if new_state.screen == Screens.IN_GAME and old_state.screen == Screens.IN_GAME:
             # 1. Damage Rewards
             reward += (d_ally_left + d_ally_right) * 1.0 # Penalize Ally Damage (negative diff)
             reward -= (d_enemy_left + d_enemy_right) * 1.5 # Reward Enemy Damage (negative diff -> positive)
             
             # 2. Elixir Leak Punishment
             # If elixir is at max (10), we are wasting resources. 
             # Only punish if it stays at 10.
             if new_state.numbers.elixir.number >= 10:
                 reward -= 0.1
                 
             # 3. Elimination Reward & Spawn Penalty
             # "Reward for eliminate rival troops": When enemy count decreases.
             n_enemies_old = len(old_state.enemies)
             n_enemies_new = len(new_state.enemies)
             d_enemies = n_enemies_new - n_enemies_old
             reward -= d_enemies * 0.5 
             
             # 4. Troop Advantage (Dense Reward, Weighted by Elixir Cost)
             # "The troop allies should be pondered by its cost of elixir"
             # We compare the total elixir value of allies vs enemies on board.
             
             val_allies = sum(self._get_unit_cost_val(u.unit.name) for u in new_state.allies)
             val_enemies = sum(self._get_unit_cost_val(u.unit.name) for u in new_state.enemies)
             
             elixir_advantage = val_allies - val_enemies
             
             # Scale factor: 0.02.
             # Example: +5 advantage (Knight vs Nothing) -> +0.1 reward per step.
             reward += elixir_advantage * 0.02

        return reward
