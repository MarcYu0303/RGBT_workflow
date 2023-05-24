from myclass import RGBT


def main() -> None:
    input_folder = f'./pointdata/low_Temperature_TimeSeries_20230524'
    file_name = 23
    
    time_seires_path = f'{input_folder}/{file_name}.npy'
    rgbt = RGBT(file_name=file_name)
    rgbt.load_time_series(time_seires_path)
    rgbt.final_temperature = file_name
    
    rgbt.cut_time_series(max_frame=200, min_frame=0)
    
    # visiualization
    rgbt.smooth_time_series()
    
    rgbt.RBG_or_BGR = 'BGR'
    rgbt.hue_time_plot()
    rgbt.hue_to_temperature()

def traverse():
    input_folder = f'./pointdata/low_Temperature_TimeSeries_20230524'
    for i in range(17, 31):
        file_name = i
        
        time_seires_path = f'{input_folder}/{file_name}.npy'
        rgbt = RGBT(file_name=file_name)
        rgbt.load_time_series(time_seires_path)
        rgbt.final_temperature = file_name

        rgbt.cut_time_series(max_frame=200, min_frame=0)

        # visiualization
        rgbt.smooth_time_series()

        rgbt.RBG_or_BGR = 'BGR'
        rgbt.hue_time_plot()
        rgbt.hue_to_temperature()
        

if __name__ == '__main__':
    main()