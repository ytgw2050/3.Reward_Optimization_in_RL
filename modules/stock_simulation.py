import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm




class StockSimulator:
    def plot_graph(self,agent_num):
        plt.rcParams['figure.figsize'] = [20,30]
        plt.rcParams.update({'font.size': 14})
        
        progress_bar = tqdm(total=1)

        fig, axs = plt.subplots(6, gridspec_kw={'height_ratios': [1,0.1, 1, 0.4,0.5, 1]})

        axs[0].plot(range(len(self.stock_price_list)), self.stock_price_list, color='red', linewidth= 2)
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Stock Price')
        axs[0].set_title('Stock Price vs. Time')
        progress_bar.update(0.2)

        
        
        vcv = np.array(self.total_capital_list).transpose()
        
        for i in range(len(vcv)):
            if i < agent_num :
                axs[2].plot(range(len(vcv[i])), vcv[i], color = 'r', linewidth= 5.0)
            else:
                axs[2].plot(range(len(vcv[i])), vcv[i], color = 'b', linewidth=0.2)
            axs[2].set_xlabel('Time')
            axs[2].set_ylabel('Total Capital')
            axs[2].set_title('Total Capital vs. Time')
            
        progress_bar.update(0.2)

        axs[2].plot(range(len(self.total_capital_list)), self.total_capital_list, color = 'b', linewidth=0.2)
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('Total Capital')
        axs[2].set_title('Total Capital vs. Time')
        progress_bar.update(0.2)
        
        

        
        capital_range = [float(max(b)) - float(min(b)) for b in self.total_capital_list]
        progress_bar.update(0.2)
        

    
        axs[3].bar(range(len(capital_range)), capital_range, color='gray', linewidth=1.0, edgecolor='blue')
        axs[3].set_xlabel('Time')
        axs[3].set_ylabel('Capital Range')
        axs[3].set_title('Capital Range vs. Time')
        progress_bar.update(0.2)
        

        dd = smooth_data(capital_range)
        axs[4].bar(range(len(dd)), dd, color='gray', linewidth=0.3, edgecolor='red')
        axs[4].set_xlabel('Time')
        axs[4].set_ylabel('Capital Range')
        axs[4].set_title('Capital Range vs. Time')
        progress_bar.update(0.2)
        
        
        
        vcc = np.array(self.holding_stock_num).transpose()
        
        for i in range(len(vcc)):
            if i < agent_num :
                axs[5].plot(range(len(vcc[i])), vcc[i], color = 'r', linewidth= 2)
            else:
                axs[5].plot(range(len(vcc[i])), vcc[i], color = 'b', linewidth= 0.2)

        axs[5].set_xlabel('Time')
        axs[5].set_ylabel('Holding Stock Number')
        axs[5].set_title('Holding Stock Number vs. Time')



        for ax in axs:
            ax.grid(True)

        fig.subplots_adjust(hspace=0.5)

        plt.show()

