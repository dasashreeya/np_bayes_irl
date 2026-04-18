"""
objectworld.py  |  Owner: P1  |  Built: Day 1
10x10 grid with colored objects. Defines states, actions, transitions, feature map phi(s).
Reward: R(s,a) = w . phi(s)
"""
import jax.numpy as jnp 
import numpy as np 

COLORS = ['red','blue','green',' yellow','purple','orange','white','black']
N_COLORS= len(COLORS)

class ObjectWorld:
    def __init__(self, size =10 , n_objects=5 , seed=42):
        self.size = size 
        self.n_states= size * size
        self.n_actions = 4
        self.actions = {0:'up', 1:'down',2:'left', 3:'right'}
        rng = np.random.RandomState(seed)
        # Placing Objects 
        self.objects = []
        positions = rng.choice(size*size , n_objects , replace=False)
        for pos in positions:
            row,col=pos//size , pos%size
            color = rng.randint(0,N_COLORS)
            self.objects.append((row,col,color))
    
    def state_to_pos(self,s):
        return s//self.size , s% self.size
    
    def pos_to_state(self,row,col):
        return row * self.size + col
    
    def transitions(self):
        T=np.zeros((self.n_states,self.n_actions,self.n_states))
        for s in range(self.n_states):
            row,col=self.state_to_pos(s)
            # up=0 down=1 left=2 right=3
            moves = [(-1,0),(1,0),(0,-1),(0,1)]
            for a,(dr,dc) in enumerate(moves):
                nr=max(0,min( self.size-1, row +dr))
                nc=max(0,min( self.size-1,col+dc))
                ns=self.pos_to_state(nr,nc)
                T[s,a,ns]=1.0
        return jnp.array(T)

    def features(self):
       n_features = N_COLORS * 2
       phi = np.zeros((self.n_states, n_features))
       for s in range(self.n_states):
           row, col = self.state_to_pos(s)
           dists = []
           for (or_, oc, color) in self.objects:
               # Manhattan distance
               d = abs(row - or_) + abs(col - oc)  
               dists.append((d, color))
           dists.sort(key=lambda x: x[0])
           if len(dists) >= 1:
               phi[s, dists[0][1]] = 1.0       
           if len(dists) >= 2:
               phi[s, N_COLORS + dists[1][1]] = 1.0  
       return jnp.array(phi)

    def reward_from_weights(self, w, phi=None):
       
        if phi is None: phi = self.features()
        R = phi @ w  
        # Expand to (n_states, n_actions) 
        return jnp.tile(R[:, None], (1, self.n_actions))