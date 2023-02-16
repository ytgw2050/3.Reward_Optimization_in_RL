import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import networkx as nx





class Person:
    def __init__(self, person_id , total_capital, holding_stock_num, env):
        self.person_id = person_id
        self.name = 'node'
        self.color = 'red'
        self.size = 500

        self.total_capital = total_capital
        self.investable_capital = total_capital
        self.holding_stock_num = holding_stock_num
        self.env = env
        # self.agent = DDPGAgent(env,num_actions=3, input_window_size=10)
        
        self.input_nodes = []
        self.output_nodes = []
        
        self.access = 0
        self.denied = 0
        self.alpha  = 1
        
        
        self.criterion = [0 , 0] # 기준치 변경
        self.reputation = 0.5 # 신뢰도
        self.total_node_num = len(self.input_nodes) + len(self.output_nodes) # 총 연결수
        self.total_node_rate = 0 # 총 생산수 받은것분의 준것
        
        self.try_experience = 1
        self.experience_data = []
    
    

         
    def get_reputation(self): # 사회에서 인정받은 정도 신뢰성
        return self.reputation
        
        
    def update_reputation(self, success,another):
        if success:
            self.access += 1# * another.total_capitals
        else:
            self.denied += 1 #* another.total_capitals # 신뢰도 가진 자본으로 측정 단순히 성공 실패가 아닌 어떤사람한테 성공했는지 어떤사람한테 실패했는지 파악 다른 반대값 들어가야함 별거없는 애한테 무시당했으면 타격이 더큼 
            
        
        if (self.access + self.denied) == 0:
            self.reputation = 0
        else:
            self.reputation = self.access / (self.access + self.denied)
        
        
    def make_random_criterion(self):# 각자의 기준 생성
        x1 = random.random()
        x2 = random.random()
        
        min_ = min(x1,x2)
        max_ = max(x1,x2)
        
        self.criterion = [min_ , max_]
        
        return min_ , max_



    def make_real_invest(self): #모델 주문
        state = self.env.get_past_prices(self.agent.input_window_size)
        action, order_num, order_price_ratio = self.agent.choose_action(state)
        order_type = ['buy', 'sell', 'hold'][action]
        current_price = self.env.get_price()
        order_price = current_price + current_price * order_price_ratio
        getattr(self, order_type)(order_num, order_price, invest_able_for_market=None)

        reward = self.total_capital 

        next_state = self.env.get_past_prices(self.agent.input_window_size)
        action_target, _, _ = self.agent.choose_action(next_state)
        action_target = tf.keras.utils.to_categorical(action_target, num_classes=self.agent.num_actions)
        target = reward + self.agent.discount_factor * self.agent.critic.predict([np.expand_dims(next_state, axis=0), np.array([action_target]), np.array([order_num]), np.array([order_price_ratio])],verbose = 0)
        self.agent.critic.fit([np.expand_dims(state, axis=0), np.array([action_target]), np.array([order_num]), np.array([order_price_ratio])], np.array([target]))#, verbose=0
        
        
    def make_random_invest(self): # 랜덤 주문
        order_type = random.choice(['buy', 'sell', 'hold'])
        order_num = random.randint(1, 10)
        order_price = self.env.get_price() + max(random.randint(-10, 10),-self.env.get_price()+1) # 값이 0보다작아져서 추천해주면 변경최소선
        getattr(self,order_type)(order_num, order_price, invest_able_for_market=None)
        
        
    def can_invest(self, order_num, order_price, order_type): # 거래 가능한지
        if order_type == 'buy':     
            return (self.investable_capital // order_price) >= order_num
        elif order_type == 'sell':
            return (self.holding_stock_num) >= order_num
        else:
            return False

        
    def buy_first(self, order_num): # 초기 주식 판매
        self.investable_capital -= order_num * self.env.get_price()
        self.holding_stock_num += order_num
        self.total_capital = self.investable_capital + self.holding_stock_num * self.env.get_price()


    def buy(self, order_num, order_price, invest_able_for_market=None):# None 값 거래주문만 하면 /Ture 이면 실제 자본 업데이트 거래 성사 / False 거래 성사 실패
        if invest_able_for_market is None:
            if self.can_invest(order_num, order_price, 'buy'): # invest_able_for_market 적으면 주문성사된것대로 자본업데이트 하기 
                self.env.add_invest(self.person_id, 'buy', order_num, order_price)
            else:# 거래 성사 실패시
                self.env.add_invest(self.person_id, 'hold', order_num, order_price)
        else: # invest_able_for_market 안적고 buy 만 하면 주문 들어가도록 하기 
            if invest_able_for_market:
                self.investable_capital -= order_num * order_price
                self.holding_stock_num += order_num
                self.total_capital = self.investable_capital + self.holding_stock_num * self.env.get_price()
            else:
                self.total_capital = self.investable_capital + self.holding_stock_num * self.env.get_price()

            
    def sell(self, order_num, order_price, invest_able_for_market=None):
        if invest_able_for_market is None:
            if self.can_invest(order_num, order_price, 'sell'):
                self.env.add_invest(self.person_id, 'sell', order_num, order_price)
            else:
                self.env.add_invest(self.person_id,'hold', order_num, order_price)
        else:
            if invest_able_for_market:
                self.investable_capital += order_num * order_price
                self.holding_stock_num -= order_num
                self.total_capital = self.investable_capital + self.holding_stock_num * self.env.get_price()
            else:
                self.total_capital = self.investable_capital + self.holding_stock_num * self.env.get_price()

            
    def hold(self, order_num, order_price, invest_able_for_market=None):
        if invest_able_for_market is None:
            self.env.add_invest(self.person_id, 'hold', order_num, order_price)
        else:
            # 주문이 성사되었을 때
            self.investable_capital += order_num * order_price
            self.holding_stock_num -= order_num
            self.total_capital = self.investable_capital + self.holding_stock_num * self.env.get_price()

