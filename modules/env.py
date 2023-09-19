import random
import numpy as np
from tqdm import tqdm
import copy
from modules.person import Person

def calculate_ratios_invest(data):
    p = []
    total_count = len(data)
    type_ = ['buy','sell','hold']

    for i in range(3):
        if total_count == 0:
            p.append(0)
        else:
            p.append(round(data.count(type_[i]) / total_count,2))
    return p


def calculate_ratios_num(data):
    p = []
    total_count = len(data)
    for i in range(10):
        if total_count == 0:
            p.append(0)
        else:
            p.append(round(data.count(i) / total_count,2))

    return p

def calculate_ratios_price(data,mean_price):
    p = []
    total_count = len(data)
    for i in range(10):
        if total_count == 0:
            p.append(0)
        else:
            p.append(round(data.count(mean_price-5+i) / total_count,2))

    return p

def moving_average(lst):
    window=100
    ret = []
    for i in range(len(lst)):
        if i < window:
            ret.append(sum(lst[:i+1])/(i+1))
        else:
            ret.append(sum(lst[i-window+1:i+1])/window)
    return ret


import copy
class Environment:  # 환경 구성
    def __init__(self,person_num,agent_num,step):
        self.steps = step            #스텝수
        self.person_num = person_num #사람수
        self.agent_num = agent_num   #모델수
        self.total_person_num = self.person_num + self.agent_num # 총 인원
        self.initial_price = 10000  # 시작수
        self.person_dict = {}       # 사람 리스트
        self.information_all = {}   # 총 정보 
        self.information_individual = {}  # 개인의 정보 리스트 썼다 지웠다
        self.step = 1  # 현재 스텝수

        self.stock_price = 100 # 현재 주식 가격
        self.invest_list = [] # 주식 거래 정보
        self.stock_price_list = [] # 주식 가격 리스트
        self.total_capital_list = []
        self.holding_stock_num = []
        self.agent_num = agent_num
        self.total_invest = []
        
        
    def reset(self):
        self.person_num = self.person_num #사람수
        self.agent_num = self.agent_num   #모델수
        self.total_person_num = self.person_num + self.agent_num # 총 인원
        self.initial_price = 10000  # 시작수
        self.information_all = {}   # 총 정보 
        self.information_individual = {}  # 개인의 정보 리스트 썼다 지웠다
        self.step = 1  # 현재 스텝수

        self.stock_price = 100 # 현재 주식 가격
        self.invest_list = [] # 주식 거래 정보
        self.stock_price_list = [] # 주식 가격 리스트
        self.total_capital_list = []
        self.holding_stock_num = []
        self.total_invest = []
        
        

        total_capital = 100000
        holding_stock_num = 0
        for person in list(self.person_dict.values()):
            person.reset(total_capital)
        
        input_layer = np.array([[0]*37])#np.expand_dims(np.random.rand(37), 0)


        self.state = input_layer
        return self.state

    def create_person(self,target_model_id): # 사람 생성
        holding_stock_num = 0
        total_capital = 100000
        for i in range(self.person_num):
#             if i == target_model_id:
#                 model_type = 'hard'
#             else:
            model_type = 'lite'
            person_id = i
            person = Person(person_id , total_capital, holding_stock_num, self, model_type) # person_id 1부터 생성
            self.add_person(person)

    def save_person_information(self):
        t_c = []
        h_s_n = []
        for i in self.person_dict:
            t_c.append(self.person_dict[i].total_capital)
            h_s_n.append(self.person_dict[i].holding_stock_num)
        self.total_capital_list.append(t_c)
        self.holding_stock_num.append(h_s_n)

        
    def get_price(self):
        return self.stock_price
        
    def set_add_price(self, price):
        self.stock_price_list.append(price)
                
    def add_person(self, person): # 새로운 사람 생성
        self.person_dict[person.person_id] = person
        
    def set_price(self, price):
        self.stock_price = price
        
    def add_invest(self, person_id,order_type, order_num, order_price):
        invest = {
            "person_id": person_id,
            "order_type": order_type,
            "order_num": order_num,
            "order_price": order_price,
        }
        self.invest_list.append(invest)
    
        
    def get_past(self,n):
        return self.invest_list[n:]
    
    def get_past_prices(self, window_size):
        if len(self.stock_price_list[-window_size-1:-1]) < 10:
            return np.array([100 for i in range(10)])
        return np.array(self.stock_price_list[-window_size-1:-1])

    def invest(self):
        investable_price = []
        self.invest_list = sorted(self.invest_list, key=lambda x: x['order_price'], reverse=True) # 높은것 먼저 순서대로

        for me in self.invest_list:
            before_len = len(self.invest_list)
            for other in self.invest_list:
                if me['order_type'] in ['buy','sell'] and other['order_type'] in ['buy','sell'] and me['order_type'] != other['order_type'] and me['order_price'] == other['order_price'] :
                    min_order_num = min(me['order_num'],other['order_num'])
                    getattr(self.person_dict[me['person_id']],me['order_type'])(min_order_num,  me['order_price'], invest_able_for_market=True) # 거래 성사시
                    getattr(self.person_dict[other['person_id']],other['order_type'])(min_order_num,  me['order_price'], invest_able_for_market=True) # 거래 성사시
                    self.invest_list.remove(me)
                    self.invest_list.remove(other)
                    investable_price.append(me['order_price'])
                    break
            after_len = len(self.invest_list)

            if before_len != after_len:
                getattr(self.person_dict[me['person_id']],me['order_type'])(min_order_num,  me['order_price'], invest_able_for_market=False) # 거래 불가


        if len(investable_price) == 0:
            price = self.get_price()
        else:
            price = float(sum(investable_price)/len(investable_price))

        
        type_invest = []
        type_price = []
        type_num = []

        for bv in self.invest_list:
            type_invest.append(list(bv.values())[1])
            type_num.append(list(bv.values())[2])
            type_price.append(list(bv.values())[3])



        self.total_invest.append(self.invest_list)


        tt = []
        for ip in list(self.person_dict.values()):
            tt.append(ip.total_capital)
            
        if len(self.stock_price_list[-10:]) < 10:
            gkd = copy.deepcopy(self.stock_price_list[-10:])
            

            gkd = [100 for i in range(10-len(self.stock_price_list[-10:]))] + gkd
            
        else:
            gkd= self.stock_price_list[-10:]
            
            
        per = list(self.person_dict.values())[0]
        
            
        # 지금 현재 자기 순위 , 사람들 사고팔고 정도 , 사람들 개수 정도 , 사람들 단위 정도
        nested_list = [[per.total_capital] , [per.investable_capital] ,[per.holding_stock_num], gkd ,[tt[0]],calculate_ratios_invest(type_invest),calculate_ratios_num(type_num),calculate_ratios_price(type_price,self.stock_price)]
        
        
        
        
        
        
        self.state = np.array([[item for sublist in nested_list for item in sublist]])

        self.set_add_price(price)
        self.set_price(price)
        
#
        def rank_of_first_element(lst):
            first_element = lst[0]
            sorted_list = sorted(lst, reverse=False)  # Sort in descending order
            rank = sorted_list.index(first_element) + 1  # Adding 1 to start the rank from 1 instead of 0
            return rank - 5

        
        
        self.invest_list = []
        
        result = rank_of_first_element(tt)
        
#         if  result < 0:
#             result_r  = - (result)**2
#         else:
#             result_r  =  (result)**2


        total_reward = []
        for i in range(len(list(self.person_dict.values()))):
#             total_reward.append(result)
            total_reward.append(list(self.person_dict.values())[i].total_capital)#-100000 + result*200)
            
        return self.state ,total_reward#result # result*100 + # targetid 만 reward 변화
        
        
  
    def invest_start(self, first_start=None):# 스텝수 뺐음
        self.create_person()
        
        if first_start:
            for person in self.person_dict.values():
                person.buy_first(random.randint(10,20))
        
        for i in tqdm(range(self.steps)):
            for person,t in zip(self.person_dict.values(),range(len(self.person_dict.values()))): # 주문 생성
#                 if t == 0:
#                     getattr(person,order_type)(order_num, order_price, invest_able_for_market=None)
                person.random_invest() # 거래소에 주문 추가 
        
            self.total_invest.append(self.invest_list)
            self.invest() # 주문 거래소에서 시행
        
            self.save_person_information()
            
            
          
    def record_data(self,data):
        self.dataes.append(data)
            
            
    def update_Q(self,last_step_reward):
        model = self.person_dict[self.target_id].agent.model
        
        for data in self.dataes:
            Q_values_1 = data[2][0]
            Q_values_2 = data[2][1]
            Q_values_3 = data[2][2]

            Q_values_next = model.predict(data[3], verbose=0)

            Q_values_next_1 = Q_values_next[0]
            Q_values_next_2 = Q_values_next[1]
            Q_values_next_3 = Q_values_next[2]
            

            Q_values_1[0][data[1][0]] = last_step_reward + 0.95 * np.max(Q_values_next_1[0]) # gamma = 0.95
            Q_values_2[0][data[1][1]] = last_step_reward + 0.95 * np.max(Q_values_next_2[0]) # gamma = 0.95
            Q_values_3[0][data[1][2]] = last_step_reward + 0.95 * np.max(Q_values_next_3[0]) # gamma = 0.95

            self.target_data.append([data[0],[Q_values_1,Q_values_2,Q_values_3]])
        self.dataes = []
        
    def update_Q_last(self,last_step_reward):
        model = self.person_dict[self.target_id].agent.model
        
        for data in self.dataes:
            Q_values_1 = data[2][0]
            Q_values_2 = data[2][1]
            Q_values_3 = data[2][2]

            Q_values_next = model.predict(data[3], verbose=0)

            Q_values_next_1 = Q_values_next[0]
            Q_values_next_2 = Q_values_next[1]
            Q_values_next_3 = Q_values_next[2]
            

            Q_values_next_1[0][data[1][0]] = last_step_reward
            Q_values_next_2[0][data[1][1]] = last_step_reward 
            Q_values_next_3[0][data[1][2]] = last_step_reward 

            self.target_data.append([data[0],[Q_values_1,Q_values_2,Q_values_3]])
        self.dataes = []
            
    def train(self):
        if len(self.target_data) >64:
            model = self.person_dict[self.target_id].agent.model
            for batch in range(64):
                target_df = self.target_data[random.randint(0,len(self.target_data)-1)]
                model.fit(target_df[0],target_df[1], epochs=1, verbose=0)

            while len(self.target_data) >1000:
                del self.target_data[0]