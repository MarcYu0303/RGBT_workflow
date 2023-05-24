import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn import svm
from scipy.signal import savgol_filter
from mpl_toolkits.mplot3d import Axes3D
import cv2
import pandas as pd
import os

class RGBT():
    def __init__(self, time_series=None, file_name=None):
        """
        time_series type: .npy 对于一个像素点颜色数值变化的时间序列
        file_name type: text  读取的序列名字
        table type: dataframe 查找表
        """
        self.time_series = time_series
        self.file_name = file_name
        self.output_temperature_series = None
        self.output_hue_series = None
        self.table = None
        self.interpolated_table = None
        self.plots_output_dir = './output/plots/'
        self.data_output_dir = './output/output_data/'
        self.final_temperature = 0
        self.RBG_or_BGR = 'RGB'
        
        if not os.path.exists(self.plots_output_dir):
            os.makedirs(self.plots_output_dir)
        if not os.path.exists(self.data_output_dir):
            os.makedirs(self.data_output_dir)
        
        self.flag_plot_show = 0
        
        
    def load_time_series(self, dir):
        self.time_series = np.load(f'{dir}')
    
    def load_table(self, dir):
        table = pd.read_csv(f'{dir}')
        self.table = table[['R', 'G', 'B', 'T']]
        
    def cut_table(self, operation, _temperature):
        """
        用于裁减表，一些情况下我们希望将表拆分让高温和低温不相互干扰
        :param operation:
        :param _temperature:
        :return:
        """
        if operation == 'smaller':
            self.table = self.table[self.table['T'] < _temperature]
        elif operation == 'greater':
            self.table = self.table[self.table['T'] > _temperature]
        elif operation == 'equal':
            self.table = self.table[self.table['T'] == _temperature]
    
    def cut_time_series(self, max_frame, min_frame=0):
        self.time_series = self.time_series[min_frame:max_frame+1]
    
    def smooth_time_series(self, window_length=49, poly_order=3):
        bs, gs, rs = list(), list(), list()
        for i in self.time_series:
            bs.append(i[0])
            gs.append(i[1])
            rs.append(i[2])
        bs_smooth = savgol_filter(bs, window_length, poly_order)
        gs_smooth = savgol_filter(gs, window_length, poly_order)
        rs_smooth = savgol_filter(rs, window_length, poly_order)
        output = list()
        for b, g, r in zip(bs_smooth, gs_smooth, rs_smooth):
            output.append([b, g, r])
        self.time_series = np.array(output)
        print('Time-series is smoothed.')
        
    def plot_2d_time_series(self):
        bs, gs, rs = list(), list(), list()
        xs = np.arange(1, len(self.time_series) + 1)
        for i in self.time_series:
            bs.append(i[0])
            gs.append(i[1])
            rs.append(i[2])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xs, bs, color='b')
        ax.plot(xs, gs, color='g')
        ax.plot(xs, rs, color='r')

        plt.xlabel('Frame')
        plt.ylabel('RGB value')
        plt.title(f'RGB-time graph {self.file_name}')
        plt.savefig(f'{self.plots_output_dir}{self.file_name}_2Dplot.png')
        plt.show()
        print('plot_2d_time_series: Done')
        
    def plot_3d_time_series(self):
        """
        3d绘图窗口化才可以转换视角，请执行以下操作
        1. settings->Tools->Python Scientific->取消掉Show plots in toolwindow选项
        2. plt.show(block=True)
        :param save_direct: 保存地址
        :param flag_save: 是否保存，初始为否
        """
        xs, ys, zs = list(), list(), list()  # xs, ys, gs -> bs, gs, rs
        count, frame = 0, 0  # count the length of minimum plot unit

        fig = plt.figure()
        plt.ion()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('B')
        ax.set_ylabel('G')
        ax.set_zlabel('R')

        for i in self.time_series:
            xs.append(i[0])
            ys.append(i[1])
            zs.append(i[2])
            count += 1
            if count == 50:
                ax.plot(xs, ys, zs, color=(zs[0] / 255, ys[0] / 255, xs[0] / 255))
                xs, ys, zs, count = list(), list(), list(), 0
                xs.append(i[0])
                ys.append(i[1])
                zs.append(i[2])
            if frame % 50 == 0:
                ax.text(i[0], i[1], i[2], frame, color='r')
            frame += 1
        ax.plot(xs, ys, zs, color=(zs[0] / 255, ys[0] / 255, xs[0] / 255))

        plt.title(f'RGB-time graph (3d) {self.file_name}')
        plt.savefig(f'{self.plots_output_dir}{self.file_name}_3Dplot.png')
        plt.show()
        plt.clf()
        print('plot_3d_time_series: Done')
    
    def _plot_temperature_series(self, temperature_input):
        """
        此方法用于绘制温度变化曲线
        :param temperature_input: 温度数据
        :return:
        """
        xs = np.arange(1, len(temperature_input) + 1)
        plt.plot(xs, temperature_input)
        plt.xlabel('Frame')
        plt.ylabel('Temperature')
        plt.title(f'{self.file_name}_temperature_mapping')
        # plt.ylim(15, 25)
        # plt.xlim(0, 350)
        plt.savefig(f'{self.plots_output_dir}{self.file_name}_temperature_mapping.png')
        plt.clf()
      
    def _save_rgbT_csv(self):
        output = []
        for bgr, T in zip(self.time_series, self.output_temperature_series):
            output.append([bgr[2], bgr[1], bgr[0], T])
        output_df = pd.DataFrame(output, columns=['R', 'G', 'B', 'T'])
        output_df.to_csv(f'{self.data_output_dir}{self.file_name}.csv')
    
    def hue_time_plot(self):
        hue_time = []
        for bgr in self.time_series:
            hue_time.append(self.bgr_to_hue(bgr))
            
        xs = np.arange(1, len(hue_time) + 1)
        plt.plot(xs, hue_time)
        plt.xlabel('Frame')
        plt.ylabel('Hue')
        plt.title(f'{self.file_name}_hue_time')
        plt.savefig(f'{self.plots_output_dir}{self.file_name}_hue_time.png')
        plt.clf()
    
    def hue_to_temperature(self, initial_temprature=24):
        temp_time, hue_time = [], []
        for bgr in self.time_series:
            hue = self.bgr_to_hue(bgr)
            hue_time.append(hue)
        hue_max, hue_min = float(max(hue_time)), float(min(hue_time))
        
        for hue in hue_time:
            if self.RBG_or_BGR == 'RGB':
                if initial_temprature > self.final_temperature:
                    temp = initial_temprature - (hue - hue_min) * np.abs(self.final_temperature - initial_temprature) / (hue_max - hue_min)
                else:
                    temp = initial_temprature + (hue - hue_min) * np.abs(self.final_temperature - initial_temprature) / (hue_max - hue_min)
            elif self.RBG_or_BGR == 'BGR':
                if initial_temprature > self.final_temperature:
                    temp = initial_temprature - (hue_max - hue) * np.abs(self.final_temperature - initial_temprature) / (hue_max - hue_min)
                else:
                    temp = initial_temprature + (hue - hue_min) * np.abs(self.final_temperature - initial_temprature) / (hue_max - hue_min)
            temp_time.append(temp)
        
        self._plot_temperature_series(temp_time)
        self.output_temperature_series = temp_time
        self.output_hue_series = hue_time
    

    def bgr_to_hue(self, bgr):
        bgr = bgr.reshape(1, 1, 3).astype(np.uint8)
        if self.RBG_or_BGR == 'BGR':
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        elif self.RBG_or_BGR == 'RGB':
            hsv = cv2.cvtColor(bgr, cv2.COLOR_RGB2HSV)
        return hsv[0, 0, 0]
    
    def save(self):
        self._save_rgbT_csv()
        return self.output_temperature_series, self.output_hue_series
    


 
def main() -> None:
    file_name = 'Al-IR-16_block2'
    time_seires_path = f'./YR/point_data/{file_name}.npy'
    table_path = './YR/RGBT_rc_record_20230223.csv'
    rgbt = RGBT()
    
    rgbt.load_time_series(time_seires_path)
    rgbt.load_table(table_path)
    rgbt.file_name = file_name
    
    # cut table and input time_series
    rgbt.cut_table('smaller', 31)
    rgbt.cut_time_series(max_frame=200, min_frame=0)
    # rgbt.cut_time_series(max_frame=300, min_frame=0)
    
    
    # visiualization
    rgbt.smooth_time_series()
    
    rgbt.hue_time_plot()
    rgbt.hue_to_temperature()

# if __name__ == '__main__':
#     main()