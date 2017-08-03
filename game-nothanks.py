import sys
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from collections import defaultdict




def draw_card(np_random, deck):
    idx = np_random.randint(0, len(deck))
    return deck.pop(idx),  deck

# Function to sum up the hand of each player
def sum_hand(hand):
    # Sort the cards in descending order first
    points = 0
    hand.sort()
    for i in range(len(hand)):        
        if i == 0 :
            points = hand[i]            
        else:
            if (hand[i-1] + 1) != hand[i] :
                points += hand[i]
    
    return points

def make_epsilon_greedy_policy(Q, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            #sys.stdout.flush()

        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = []
        state = env.reset()
        for t in range(100):
            #print("\n\tStep {}".format(t), end ="")
            probs = policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        # Find all (state, action) pairs we've visited in this episode
        # We convert each state to a tuple so that we can use it as a dict key
        sa_in_episode = set([(tuple(x[0]), x[1]) for x in episode])
        for state, action in sa_in_episode:
            sa_pair = (state, action)
            # Find the first occurance of the (state, action) pair in the episode
            first_occurence_idx = next(i for i,x in enumerate(episode)
                                       if x[0] == state and x[1] == action)
            # Sum up all rewards since the first occurance
            G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:])])
            # Calculate average return for this state over all sampled episodes
            returns_sum[sa_pair] += G
            returns_count[sa_pair] += 1.0
            Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]
        
        # The policy is improved implicitly by changing the Q dictionar
    
    return Q, policy            

class CardgameEnv(gym.Env):
   #The player can have 2 actions - either pick the card up, or pass the card by relinquishing coins
   #until the player runs out of coins    
    def __init__(self):
        
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((
                spaces.Discrete(55), #Space representing the possible points for the player
                spaces.Discrete(7), #Space representing the possible coins for the player
                spaces.Discrete(55), #Space representing the possible points for the dealer
                spaces.Discrete(7) #Space representing the possible coins for the dealer
                ))
        self._seed(seed = 12345) 
        self._reset()
        self.nA = 2
        
        
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def _step(self, action):        
        assert self.action_space.contains(action)          
        
        if action or self.playercoin == 0 : # Pick up the card if action determines it or if no coins left
            prevpoints = self.playercoin - sum(self.playerhand)
            self.playerhand.append(self.card)
            self.playercoin += self.coin
            self.coin = 0
            self.card, self.deck = draw_card(self.np_random, self.deck)
            currentpoints = self.playercoin - sum(self.playerhand)            
            reward = currentpoints - prevpoints
            
        else: # Use coin to avoid taking the card
            prevpoints = self.playercoin - sum(self.playerhand)
            self.playercoin -= 1
            self.coin += 1
            currentpoints = self.playercoin - sum(self.playerhand)            
            reward = currentpoints - prevpoints
            
            # Since current player did not pick the card, the dealer needs to act next
            # The dealer uses a random policy to interact with the game
            dealeraction = self.np_random.choice([0,1])
            if dealeraction or self.dealercoin == 0 : # Pick up the card if action determines it or if no coins left
                self.dealerhand.append(self.card)
                self.dealercoin += self.coin
                self.coin = 0
                self.card, self.deck = draw_card(self.np_random, self.deck)                
            else: # dealer uses coins to avoid taking the card
                self.dealercoin -= 1
                self.coin += 1
        
        if len(self.deck) == 0:
            done = True
        else:
            done = False
            
        return self._get_obs(), reward, done, {}    
        
            
        
    def _get_obs(self):
        return (sum_hand(self.playerhand), self.playercoin, sum_hand(self.dealerhand), self.dealercoin)
        
    def _reset(self):
        self.deck = list( range(1, 11) )
        self.card, self.deck = draw_card(self.np_random, self.deck)
        self.coin = 0
        self.playerhand = []
        self.dealerhand = []
        self.playercoin = 3
        self.dealercoin = 3
        return self._get_obs()

# Create a deck of cards

   
env = CardgameEnv()
Q, policy = mc_control_epsilon_greedy(env, num_episodes=100000, epsilon=0.1)


V = defaultdict(float)
for state, actions in Q.items():
    action_value = np.max(actions)
    V[state] = action_value


