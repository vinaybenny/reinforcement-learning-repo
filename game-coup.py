from mesa import Agent, Model
from mesa.time import BaseScheduler
import numpy as np
from collections import defaultdict

def assignCard(possiblecards, count):
    hand = []
    for i in range(count):
        idx = np.random.randint(0, len(cards))
        hand.append(possiblecards[idx])
        possiblecards.pop(idx)        
    return hand, possiblecards


class Player(Agent):
    """ An agent with fixed initial wealth."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.hand, model.possiblecards = assignCard(model.possiblecards, 2)        
        self.wealth = 2        
        self.knowledge_dict = defaultdict(list)        
        print("Player " + str(self.unique_id) +"'s cards are: " + str(self.hand) )

    def step(self):  
        # Logic for normal action or card-based action
        
        
        
        if self.wealth >= 7 and np.random.random_sample() > 0.01:
            self.conduct_coup()
        elif ( self.wealth >= 3 and 'Assassin' in self.hand ):
            self.assassinate_other()
        elif ( 'Duke' in self.hand ):
            self.increase_wealth(3)
            self.model.setKnowledgedicts(self, 'Duke')
        else:
            self.increase_wealth(1)
    
    def assassinate_other(self):
        chosen_enemy = np.random.choice( list(set(self.model.schedule.agents) - set([self])) )
        self.wealth-=3
        print("Player " + str(self.unique_id) + " spends 3 wealth for assassination attempt on Player " + str(chosen_enemy.unique_id) )
        if ('Contessa' in chosen_enemy.hand):
            print("Player " + str(chosen_enemy.unique_id) + " uses Contessa to block assassination")
            print("Assassination attempt from Player " + str(self.unique_id) + " on Player " + str(chosen_enemy.unique_id) + " failed!!")
        else:
            print("Assassination attempt from Player " + str(self.unique_id) + " on Player " + str(chosen_enemy.unique_id) + " succeeded!!")
            chosen_enemy.lose_card()
            
        self.model.setKnowledgedicts(self, 'Assassin')
        chosen_enemy.model.setKnowledgedicts(chosen_enemy, 'Contessa' )
    
    def conduct_coup(self):
        chosen_enemy = np.random.choice( list(set(self.model.schedule.agents) - set([self])) )
        self.wealth-=7
        print("Player " + str(self.unique_id) + " spends 7 wealth for coup on Player" + str(chosen_enemy.unique_id))
        chosen_enemy.lose_card()
    
    def increase_wealth(self, amount):
        self.wealth+=amount
        print('Player ' + str(self.unique_id) + ' increases wealth by ' + str(amount) + ' to ' + str(self.wealth))
    
    def lose_card(self):
        card = np.random.choice(self.hand)
        self.hand.remove(card)
        print('Player ' + str(self.unique_id) + " loses " + card +" card")
        if len(self.hand) == 0:
            self.declare_loss();

    def declare_loss(self):
        print('Player ' + str(self.unique_id) + " loses game!")
        self.model.schedule.agents.remove(self)
        if len(model.schedule.agents) == 1:
            self.declare_winner()
    
    def declare_winner(self):
        print('Player ' + str(self.model.schedule.agents[0].unique_id) + ' wins!!')
        self.model.schedule.agents.remove(self.model.schedule.agents[0])

class CoupModel(Model):
    """A model with some number of agents."""    
    def __init__(self, N, cardsList):
        self.possiblecards = cardsList
        self.num_players = N        
        self.schedule = BaseScheduler(self)
        # Create agents
        for i in range(self.num_players):
            p = Player(i, self)
            self.schedule.add(p)
    
    def setKnowledgedicts(self, player, knowledge):        
        for otherplayer in list( set(self.schedule.agents) - set([player]) ):
            otherplayer.knowledge_dict[player.unique_id].append(knowledge)
    
    def step(self):
        self.schedule.step()
            
if __name__ == "__main__":
    cards=['Assassin', 'Contessa', 'Duke', 'Assassin', 'Contessa', 'Duke']
    
    model = CoupModel(3, cards)
    
    while len(model.schedule.agents) > 0 :
        model.step()