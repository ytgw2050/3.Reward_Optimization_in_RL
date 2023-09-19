import random
import numpy as np

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

from keras.optimizers import Adam

class DQN:
    def __init__(self, person_id,gamma=0.99, epsilon=0.1, batch_size=64, model_type = None):
        random.seed(None)
        np.random.seed(None)
        if model_type == 'lite':
            self.model = self.build_dqn_lite()
        elif model_type == 'hard':
            self.model = self.build_dqn_hard()
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.memory = []  
        self.model_type = model_type
        
        
    def build_dqn_lite(self):
        input_layer = Input(shape=(37,))

        hidden_layer = Dense(32, activation='relu',kernel_initializer='random_uniform', bias_initializer='random_uniform')(input_layer)


        output_layer_1 = Dense(3, activation='linear',kernel_initializer='random_uniform', bias_initializer='random_uniform')(hidden_layer)
        output_layer_2 = Dense(10, activation='linear',kernel_initializer='random_uniform', bias_initializer='random_uniform')(hidden_layer)
        output_layer_3 = Dense(9, activation='linear',kernel_initializer='random_uniform', bias_initializer='random_uniform')(hidden_layer) # For [0,1,2]

        model = Model(inputs=input_layer, outputs = [output_layer_1,output_layer_2,output_layer_3])
        model.compile(loss='mse', optimizer=Adam())
        return model
        
    def build_dqn_hard(self):

        input_layer = Input(shape=(37,))

        hidden_layer = Dense(256, activation='relu',kernel_initializer='random_uniform', bias_initializer='random_uniform')(input_layer)
        hidden_layer = Dense(128, activation='relu',kernel_initializer='random_uniform', bias_initializer='random_uniform')(hidden_layer)
        hidden_layer = Dense(32, activation='relu',kernel_initializer='random_uniform', bias_initializer='random_uniform')(hidden_layer)
        hidden_layer = Dense(12, activation='relu',kernel_initializer='random_uniform', bias_initializer='random_uniform')(hidden_layer)


        output_layer_1 = Dense(3, activation='linear',kernel_initializer='random_uniform', bias_initializer='random_uniform')(hidden_layer)
        output_layer_2 = Dense(10, activation='linear',kernel_initializer='random_uniform', bias_initializer='random_uniform')(hidden_layer)
        output_layer_3 = Dense(9, activation='linear',kernel_initializer='random_uniform', bias_initializer='random_uniform')(hidden_layer) 

        model = Model(inputs=input_layer, outputs = [output_layer_1,output_layer_2,output_layer_3])
        model.compile(loss='mse', optimizer=Adam())
        return model

    
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return [self.env.action_space[i].sample() for i in range(2)]
        else:
            q_values = self.model.predict(state)
            return [np.argmax(q_values[i]) for i in range(2)]
      

    def train(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))
        
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        for state, action, next_state, reward, done in batch:
            q_update = reward
            if not done:
                q_update = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            
            q_values = self.model.predict(state)
            q_values[action] = q_update

            self.model.fit(state, q_values, verbose=0)


class Person: # 개인의 주체
    def __init__(self, person_id , total_capital, holding_stock_num, env,model_type = None):
        
        # 네트워크속에서의 노드 정보
        self.person_id = person_id # 개인의 번호
        
        
        self.try_experience = 1     # 개인의 현재 경험
        self.experience_data = []   # 개인의 경험 데이터

        
        
        # 주식시장내에서의 정보
        self.total_capital = total_capital       # 개인의 자본
        self.investable_capital = total_capital    # 개인의 투자가능 자본
        self.holding_stock_num = holding_stock_num  # 개인의 holding 수
        self.env = env                              # 환경
        if model_type == 'lite':
            self.agent = DQN(env,person_id,model_type = 'lite') # 에이전트 정의
        elif model_type == 'hard':
            self.agent = DQN(env,person_id,model_type = 'hard') # 에이전트 정의 

            
    def can_invest(self, order_num, order_price, order_type): # 거래 가능한지 파악
        if order_type == 'buy':     
            return (self.investable_capital // order_price) >= order_num
        elif order_type == 'sell':
            return (self.holding_stock_num) >= order_num
        else:
            return False #hold 일때
        
        return True

    def buy_first(self, order_num): # 초기 주식 구매
        self.investable_capital -= order_num * self.env.get_price() # 거래가능자본
        self.holding_stock_num += order_num
        self.total_capital = self.investable_capital + self.holding_stock_num * self.env.get_price() # 총보유자본

    def buy(self, order_num, order_price, invest_able_for_market=None):# None 값 : 거래주문 거래소에 넣을떄 /Ture : 실제 거래소에서 거래 성사되었을때 / False 실제 거래소에서 거래 실패했을때
        if invest_able_for_market is None:
            if self.can_invest(order_num, order_price, 'buy'): # invest_able_for_market 적으면 주문성사된것대로 자본업데이트 하기 
                self.env.add_invest(self.person_id, 'buy', order_num, order_price) #거래소에 거래 넣어보기 
            else:# 개인자본 문제로 거래 성사 실패시
                self.env.add_invest(self.person_id, 'hold', order_num, order_price)#거래소에 거래 넣기전에 자본 문제로 실패
        else: # 실제 거래소에서 거래 성립했을떄
            if invest_able_for_market:#거래소에 거래 성공
                self.investable_capital -= order_num * order_price
                self.holding_stock_num += order_num
                self.total_capital = self.investable_capital + self.holding_stock_num * self.env.get_price()
            else:#거래소에 거래 실패
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
            
            
            
    def sum_1(self,probabilities):
        abs_probabilities = [abs(p) for p in probabilities]
        total = sum(abs_probabilities)
        normalized_probabilities = [abs(p)/total for p in abs_probabilities]
        
        return normalized_probabilities
            
            
    def real_invest(self,state): #모델 주문
        Q_values = self.agent.model.predict(state, verbose=0)

        action_1 = np.random.choice([0,1,2], p=self.sum_1(Q_values[0][0]))
        action_2 = np.random.choice([0,1,2,3,4,5,6,7,8,9], p=self.sum_1(Q_values[1][0]))
        action_3 = np.random.choice([0,1,2,3,4,5,6,7,8], p=self.sum_1(Q_values[2][0]))
    
        order_type = ['buy','sell','hold'][action_1]
        order_num = [1,2,3,4,5,6,7,8,9,10][action_2]
        order_price = self.env.stock_price + [-4,-3,-2,-1,0,1,2,3,4][action_3]

        if order_type == 'hold':
            order_num = 0
            order_price = 0


        getattr(self,order_type)(order_num, order_price, invest_able_for_market=None)
        return [action_1,action_2,action_3]
    

    def real_agent_invest(self,state): #모델 주문
        Q_values = self.agent.model.predict(state, verbose=0)
        action_1 = np.argmax(Q_values[0])
        action_2 = np.argmax(Q_values[1])
        action_3 = np.argmax(Q_values[2])

        order_type = ['buy','sell','hold'][action_1]
        order_num = [1,2,3,4,5,6,7,8,9,10][action_2]
        order_price = self.env.stock_price + [-4,-3,-2,-1,0,1,2,3,4][action_3]
        
        order_type = 'sell'
        order_num = 10
        order_price = self.env.stock_price 

#         if order_type == 'hold':
#             order_num = 0
#             order_price = 0
                        

        getattr(self,order_type)(order_num, order_price, invest_able_for_market=None)
        return [action_1,action_2,action_3]

    def random_invest(self): # 랜덤 주문
        Q_values = self.agent.model.predict(state, verbose=0)
        action_1 = np.random.choice([0,1,2])
        action_2 = np.random.choice([0,1,2,3,4,5,6,7,8,9])
        action_3 = np.random.choice([0,1,2,3,4,5,6,7,8])
        
        order_type = ['buy', 'sell', 'hold'][action_1]
        order_num = [1,2,3,4,5,6,7,8,9,10][action_2]
        order_price = self.env.get_price() + [-4,-3,-2,-1,0,1,2,3,4][action_3] # 값이 0보다작아져서 추천해주면 변경 최소선
        getattr(self,order_type)(order_num, order_price, invest_able_for_market=None) # 거래소에 거래 넣기 
        return Q_values,[action_1,action_2,action_3]
        
    def reset(self,total_capital):
        # 네트워크속에서의 노드 정보
#         self.person_id = person_id # 개인의 번호
        
        
        self.try_experience = 1     # 개인의 현재 경험
        self.experience_data = []   # 개인의 경험 데이터


        
        # 주식시장내에서의 정보
        self.total_capital = total_capital       # 개인의 자본
        self.investable_capital = total_capital    # 개인의 투자가능 자본
        self.holding_stock_num = 0  # 개인의 holding 수
        