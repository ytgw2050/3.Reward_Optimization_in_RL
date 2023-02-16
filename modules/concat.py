import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx




class Environment:
    def __init__(self, agent_num):
        self.stock_price = 100
        self.invest_list = []
        self.price_history = [100]
        self.buy_history = [0]
        self.sell_history = [0]
        self.stock_price_list = []
        self.person_dict = {}
        self.total_capital_list = []
        self.holding_stock_num = []
        self.agent_num = agent_num
        
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
                
    def add_person(self, person):
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
        
    def plot_graph(self):
        stock_simulator = StockSimulator()
        stock_simulator.stock_price_list = self.stock_price_list
        stock_simulator.total_capital_list = self.total_capital_list
        stock_simulator.holding_stock_num = self.holding_stock_num
        stock_simulator.plot_graph(self.agent_num)
        
    def get_past(self,n):
        return self.invest_list[n:]
    
    def get_past_prices(self, window_size):
        if len(self.stock_price_list[-window_size-1:-1]) < 10:
            return np.array([100 for i in range(10)])
        return np.array(self.stock_price_list[-window_size-1:-1])

    def reset(self):
        self.stock_price = self.stock_price
        self.invest_list = []
        self.price_history = [self.stock_price]
        self.buy_history = [0]
        self.sell_history = [0]
        self.stock_price_list = []
        self.total_capital_list = []
        self.holding_stock_num = []
        
    def invest(self):
        investable_price = []
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
        self.set_add_price(price)
        self.set_price(price)
        self.invest_list = []
        

class Company:
    def __init__(self,person_num,agent_num,step):
        self.steps = step
        self.person_num = person_num
        self.agent_num = agent_num
        self.total_person_num = self.person_num + self.agent_num
        self.initial_price = 10000
        self.env = Environment(self.agent_num)

        self.person_dict = {}
        self.information_all = {}
        self.information_individual = {}
        self.step = 1


    def create_person(self): # 사람 생성
        holding_stock_num = 0
        total_capital = 100000
        for i in range(self.person_num):
            person_id = i + 1
            person = Person(person_id , total_capital, holding_stock_num, self.env) # person_id 1부터 생성
            self.env.add_person(person)
            self.person_dict[person_id] = person
        
        
    def add_person(self, person): # 새로운 사람 생성
        self.person_dict[person.person_id] = person
        
        
    def connect_persons(self,connect_num): # 새로운 사람들간의 연결 생성
        for i in range(connect_num):
            persons_num = len(self.person_dict)
            s = np.random.randint(1, persons_num)
            e = np.random.randint(1, persons_num)
            self.person_dict[s].output_nodes.append(self.person_dict[e].person_id)
            self.person_dict[e].input_nodes.append(self.person_dict[s].person_id)
        
        
    def new_connect(self,find_num): # 탐색을 통한 새로운 구조 창출 및 기준치에 맞는 구조 교체
        
        for person in self.person_dict.values(): # 나중에 추후 새로운 기준 가져올때 experience 저장하기 -> 새롭게  
            person_criterion_output_x1 , person_criterion_output_x2 = person.make_random_criterion()
            person.criterion = person_criterion_output_x1 , person_criterion_output_x2 # 업데이트
            person_reputation = person.get_reputation()
            
            for i in range(find_num): # 새롭게 연결할것 find 하기 output 만 있으면 됨 직접 주는것만 
                another_input = self.person_dict[np.random.randint(1, len(self.person_dict))]
                another_criterion_input_x1 , another_criterion_input_x2 = another_input.make_random_criterion() # 새롭게 기준을 만든다는것보다 가져온다고 생각할 필요성 있음
                another_reputation = another_input.get_reputation()

                if person_criterion_output_x1 < another_reputation < person_criterion_output_x2: # 다른사람이 나의 기준에 충족 -> 내가 다른사람에게 정보를 줌
                    if another_criterion_input_x1 < person_reputation < another_criterion_input_x2:#다른 사람이 연결을 허락했을떄
                        person.output_nodes.append(another_input.person_id)
                        another_input.input_nodes.append(person.person_id)
                        another_input.update_reputation(True,another_input)
                    else: #다른 사람이 연결을 불허했을떄
                        person.update_reputation(False,another_input)
                    

            for another_input_index in person.output_nodes: # 내가 주고 있는 연결지점중에 끊을것들 찾기
                another_input = self.person_dict[another_input_index]
                another_criterion_input_x1 , another_criterion_input_x2 = another_input.make_random_criterion()
                if not another_criterion_input_x1 < person_reputation < another_criterion_input_x2  : # 내가 다른사람의 기준에 충족  -> 다른사람이 나에게 정보를 줌
                    person.output_nodes.remove(another_input_index)
                    another_input.input_nodes.remove(person.person_id)
                    person.update_reputation(False,person) # 준사람 신뢰도 감소 (나)
            
            
            for another_output_index in person.input_nodes: # 내가 받고 있는 연결지점중에 끊을것들 찾기
                another_output = self.person_dict[another_output_index]
                another_criterion_output_x1 , another_criterion_output_x2 = another_output.make_random_criterion()
                if not another_criterion_output_x1 < person_reputation < another_criterion_output_x2  : # 내가 다른사람의 기준에 충족  -> 다른사람이 나에게 정보를 줌
                    person.input_nodes.remove(another_output_index)
                    another_output.output_nodes.remove(person.person_id)
                    another_output.update_reputation(False,person) # 준사람 신뢰도 감소 (상대)

                    
                    
            
            out_node_num = len(person.output_nodes) # 단순 연결노드수에 따른 색변화 
            in_node_num = len(person.input_nodes)
            person.total_node_num = out_node_num + in_node_num


            if len(person.output_nodes) == 0:  # 생산성 업데이트
                person.total_node_rate = 0
            else:
                person.total_node_rate = len(person.input_nodes)/len(person.output_nodes)
                
                
                
#             t = person.reputation  # 단순 신뢰도 기준 
                
#             if person.total_node_rate > 1: # 전체 생산성 기준
#                 t = 1
#             else:
#                 t = person.total_node_rate
                
                
            if person.total_node_num > 10: # 전체 연결개수 기준
                t = 1
            else:
                t = person.total_node_num*0.1
                
            person.color = (0,t,0) # 기준에 따른 색변화
            person.size = t*3000 # 기준에 따른 사이즈 변화
            if t <0.4:            # 기준에 따른 투과도 변화
                person.alpha = 0.4
            else:
                person.alpha = t
                

            
        self.save_information()  
        self.step += 1
        
                
            
    def get_useable_experience(self,person_idx): # 서로 공유하고 각 사람마다 사용가능한 데이터 들고오는것
        useable_experience = []
        person = self.person_dict[person_idx]
        for other in self.person_dict.values():
            if other.person_id in person.input_nodes:
                useable_experience.append(other.experience_data) 
        return useable_experience
        

            
    def save_information(self): # 각자 정보 신뢰성 정보 저장 및 경험 공유 저장 -> 경험저장할떄에는 person.node_output 처럼만 저장해두고 나중에 경험으로 학습시킬때 person_idx를 바탕으로 새로운 경험 리스트 만들기
        for person in self.person_dict.values():
            self.information_individual[person.person_id] = [person.criterion, person.reputation,person.total_node_num , person.total_node_rate,person.input_nodes,person.output_nodes]
            person.experience_data.append(person.try_experience) # 추후 경험 사람 마다 개개인의 가져오기 수정하기 ->>>
            
        self.information_all[self.step] = self.information_individual
        self.information_individual = {}
            
    def draw_person_graph(self):
        person_colors = []
        person_sizes = []
        person_alpha = []
        G = nx.Graph()
        
        for person in self.person_dict.values():
            G.add_node(person.person_id)
        
        edges = []
        for person in self.person_dict.values():
            for next_person_idx in person.output_nodes:
                edges.append((person.person_id, next_person_idx))
            person_colors.append(person.color) # 색 저장
            person_sizes.append(person.size) # 크기 저장
            person_alpha.append(person.alpha) # 크기 저장
        
        G.add_edges_from(edges)        
        plt.figure(figsize=(15, 15))
        pos = nx.fruchterman_reingold_layout(G,seed=777,k = 0.09)
        nx.draw_networkx(G, pos=pos, with_labels=False, node_color=person_colors, node_size=person_sizes, font_size=10, font_weight='bold',alpha = 0.9)#person_alpha)
        plt.show()
        
        
    def draw_person_information_graph(self):
        values_1 = [[] for _ in range(5)]
        values_2 = [[] for _ in range(5)]
        
        for keys, values  in self.information_all.items():
            for key, value in values.items():
                value_1 = [value[0][0],value[0][1],value[1],value[2],value[3]]

                for i in range(5):
                    values_1[i].append(value_1[i])
                
            for i in range(5):
                values_2[i].append(values_1[i])

            values_1 = [[] for _ in range(5)]
            
            
        for i in range(5):
            plt.figure(figsize = (10,5))
            plt.plot(values_2[i])
            

      
            
    def start_connect(self,connect_num,try_num,find_num):
        
        self.create_person()
        self.connect_persons(connect_num)
        # person_count = self.draw_person_graph()
        # self.save_information(person_count)
        for i in range(try_num-1):
            self.new_connect(find_num)
            self.draw_person_graph()
            self.save_information()
            
        self.draw_person_information_graph()
        
        
        
    def invest_start(self,connect_num,find_num , first_start=None):
        self.create_person()
        self.connect_persons(connect_num)
            
        for i in tqdm(range(self.steps)):
            for person,t in zip(self.person_dict.values(),range(len(self.person_dict.values()))):
                
                if i == 0 and first_start:
                    person.buy_first(random.randint(0,5))

                if t < self.agent_num:
                    person.make_real_invest()
                    person.agent.train(num_episodes = 1)
                    
                else:
                    person.make_random_invest()
                
            self.env.invest()
            self.env.save_person_information()   
            self.new_connect(find_num)
            if i % 50 == 0:
                self.draw_person_graph()
                self.save_information()   
        self.draw_person_information_graph()  

    def show(self):
        self.env.plot_graph()
        
    def reset(self):
        self.env.reset()
        