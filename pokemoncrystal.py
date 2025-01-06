from gymnasium import Env,spaces
from pyboy import PyBoy
from pyboy.utils import WindowEvent
import numpy as np
from skimage.transform import downscale_local_mean

class crystalenv(Env):

    def __init__(self):
        self.pyboy = PyBoy('Pokémon Crystal Version.gbc')
        self.step_threshold = 2048 * 80
        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START
        ]

        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START
        ]   

        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.obs_dims = (72,80,3)
        self.observation_space = spaces.Dict({
            # red experiments considers 3 most recent 'seen' screens
            "screens":spaces.Box(low = 0, high = 255, shape = self.obs_dims,dtype=np.uint8), 
            # red experiments uses 8 fourier terms calculated from the sum of party levels the observation space for pokemon levels
            "party_level":spaces.Box(low=-1,high=1, shape = (8,))
            ,"pokemon_health":spaces.Box(low=0,high=1),
            "johto_gym_badges":spaces.MultiBinary(8)
            #,"map_explore": spaces.Box(low=0,high=255)
            #"box_space":spaces.Box(low=0,high=100,shape=(12*20,))
            #badge space?
            #event flag space?
        })


    def reset(self,seed=None):
        self.seed = seed
        with open('post_pokedex.state','rb') as f:
            self.pyboy.load_state(f)

        self.step_count = 0
        self.recent_screens = np.zeros(self.obs_dims,dtype=np.uint8)

        self.seen_coords = {}
        return self.get_observation(), {}
    
    def step(self,action):
        self.run_action_on_emulator(action)
        
        current_reward = self.update_reward()
        obs = self.get_observation()
        self.step_count += 1

        self.update_seen_coords()
        truncated = self.step_count >= self.step_threshold
        
        return obs,current_reward,False,truncated, {}
    
    def get_observation(self):
        screen = self.get_screen()
        self.update_recent_screens(screen)
        level_sum = self.get_level_sum()
        level_sum = self.fourier_transform(level_sum)

        observation = {
            "screens": self.recent_screens,
            "party_level":np.array(level_sum,np.float32)
            "party_level":np.array(level_sum,np.float32),
            "pokemon_health":np.array([self.get_health()]),
            #Convert 2 digit hex to 8 bit value (gym badges are stored as 8 binary switches)
            "johto_gym_badges":np.array(self.get_gym_badges(),dtype=np.int8)

        }

        return observation
    
    def update_reward(self):
        #TODO add other rewards
        scores = {
            "level" : self.get_level_sum()
            "level" : self.get_level_sum(),
            "health": self.get_health(),
            "coord_explore":0.1 * self.get_coord_reward(),
            "gym_badges":10 * sum(self.get_gym_badges())
        }
        return sum([val for _,val in scores.items()])

    def close(self):
        self.pyboy.stop()
    
    def get_gym_badges(self):
        return [int(x) for x in f"{self.read_mem(0xD857):08b}"]

    #TODO create health observation space and introduce it into reward calculation
    def get_health(self):
        cur_array = [self.read_mem(x) for x in [0xDD02,0xDD32,0xDD62,0xDD92,0xDDC2]]

        total_health = [max(self.read_mem(x),1) for x in [0xDD04,0xDD34,0xDD64,0xDD94,0xDDC4,0xDDF4]]
        
        cur_array = [self.read_mem(x) + self.read_mem(x-1) * 256 for x in [0xDD02,0xDD32,0xDD62,0xDD92,0xDDC2]]
        total_health = [max(self.read_mem(x) + self.read_mem(x-1) * 256,1) for x in [0xDD04,0xDD34,0xDD64,0xDD94,0xDDC4,0xDDF4]]
        # Return health proportion as single value
        return sum(cur_array) / sum(total_health)

    def get_level_sum(self):
        min_poke_level = 2
        starter_additional_levels = 5
        poke_levels = max(sum([max(self.read_mem(x)-min_poke_level,0) for x in [0xDCFE,0xDD2E,0xDD5E,0xDD8E,0xDDBE,0xDDEE]])-starter_additional_levels,0)
        return poke_levels
    
    def run_action_on_emulator(self,action):
        self.pyboy.send_input(self.valid_actions[action])
        # Tick emulation 8 times with button pressed and 7 with it released
        # Update screen on 8th 'release' tick
        action_ticks = 8
        self.pyboy.tick(action_ticks,False)
        self.pyboy.send_input(self.release_actions[action])
        
        self.pyboy.tick(action_ticks-1, False)
        self.pyboy.tick(1)

   
    def update_seen_coords(self):
        Map, X, Y = [self.read_mem(x) for x in [0xDCB6, 0xDCB7, 0xDCB8]]
        key = f"{Map}, {X}, {Y}" 
        if key in self.seen_coords:
            self.seen_coords[key] += 1
        else:
            self.seen_coords[key] = 1

    def get_coord_reward(self):
        return len(self.seen_coords)

    def fourier_transform(self,value):
        return np.sin(value * 2 ** np.arange(8))
    
    def get_screen(self):
        pixels = self.pyboy.screen.ndarray[:,:,0:1] # Just copies first color channel (alpha channel?)
        #down sample screen by 1/2, maybe look into using tilemaps here instead of copying the entire screen
        
        #TODO maybe look into other resizing options for optimatization?
        # Attempted to resize image and obs. space by 1/8 but got a tensor error
        pixels = downscale_local_mean(pixels,(2,2,1)).astype(np.uint8)
        return pixels
    
    def update_recent_screens(self,screen):
        # np.roll basically wraps around an ndarray, in this case we want to shift it by 1 on the third axis (move all screens around)
        self.recent_screens = np.roll(self.recent_screens,1,axis=2)
        # then we set the lowest element on the "stack" to be equal to the current screen
        self.recent_screens[:,:,0] = screen[:,:,0]
    
    def read_mem(self,addr):
        return self.pyboy.memory[addr]