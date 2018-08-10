# coding: utf-8
"""generate features"""
import datetime
import copy
import numpy as np
import pandas as pd

class FeatureServer:
    def __init__(self, city,
                 path_city_day, path_weather_forecast, 
                 begin_train_day, end_train_day,
                 begin_val_day, end_val_day,
                 begin_infer_day):
        self.begin_train_day = begin_train_day
        self.end_train_day = end_train_day
        self.begin_val_day = begin_val_day
        self.end_val_day = end_val_day
        self.begin_infer_day = begin_infer_day
        self.path_city_day = path_city_day
        self.path_weather_forecast = path_weather_forecast
        self.city = city
        
        # load whole data
        whole = pd.read_csv(path_city_day, encoding='gbk')
        
        whole.loc[whole.stat_date.isin(['2017-12-30', '2017-12-31',
                                        '2018-01-01']), 'fromyuandan'] = 0
        

        self.whole_data = whole.loc[whole.city_id.isin(city)]
        replace = self.whole_data[pd.to_datetime(self.whole_data.stat_date) == pd.to_datetime(begin_infer_day)].copy()
        self.whole_data = self.whole_data.loc[pd.to_datetime(self.whole_data.stat_date) <= pd.to_datetime(begin_infer_day)]
        '''
        self.whole_data = self.whole_data.loc[pd.to_datetime(self.whole_data.stat_date) <= pd.to_datetime(end_val_day)]

        # replace infer_day's weather feature with weather_forecast
        weather_forecast = pd.read_csv(path_weather_forecast, sep=',')
        weather_forecast = weather_forecast.loc[weather_forecast.city_id.isin(city)]
        weather_forecast = weather_forecast[pd.to_datetime(weather_forecast.stat_date) == pd.to_datetime(begin_infer_day)]

        # replace = self.whole_data[self.whole_data['stat_date'] == end_val_day].copy()
        # fix a specific day
        # replace = self.whole_data[pd.to_datetime(self.whole_data.stat_date) == pd.to_datetime(end_val_day)].copy()
        # replace.stat_date = self.begin_infer_day

        a = set(replace.columns.tolist())
        b = set(weather_forecast.columns.tolist())
                
        weather_forecast = weather_forecast[list(a & b)]
        replace = replace[list(a.difference(b)) + ['city_id', 'stat_date']]        
        replace = pd.merge(replace, weather_forecast, on=['city_id', 'stat_date'])
        replace = replace[self.whole_data.columns.tolist()]
        
        #print (weather_forecast.hour_skycon_15_RAIN, weather_forecast.hour_skycon_12_PARTLY, weather_forecast.hour_skycon_8_CLOUDY, weather_forecast.hour_skycon_16_CLEAR)
        #print (replace.hour_skycon_15_RAIN, replace.hour_skycon_12_PARTLY, replace.hour_skycon_8_CLOUDY, replace.hour_skycon_16_CLEAR)
        """
        print ('+++++++++++ weather_forecast +++++++++')
        print (weather_forecast)
        print ('++++++++++ replace +++++++++++++')
        print (replace)
        """

        self.whole_data = self.whole_data.append(replace, ignore_index=True)
        '''

    # delete holiday, before holiday 1, after holiday 1
    def _exclude_festival(self):
        data = self.whole_data
        spring_begin = '2018-02-07'
        spring_end = '2018-02-28'
        festival = ['toyuanxiao', 'tozhongqiu', 'toyuandan',
                    'toguoqing', 'tolaodong', 'toshengdan',
                    'tochunjie', 'toqingming', 'toduanwu',
                    'qingren', 'fromyuanxiao',
                    'fromzhongqiu', 'fromyuandan', 'fromguoqing',
                    'fromlaodong', 'fromshengdan', 'fromchunjie',
                    'fromqingming', 'fromduanwu']
        data = data.copy()
        data = data.loc[(data.stat_date >= spring_end) |
                        (data.stat_date <= spring_begin)]
        not_festival_index = ~(data[festival].isin([0, 1]).any(axis=1))
        data = data.loc[not_festival_index]
        return data
    
    def _gen_features(self):
        data = self._exclude_festival()

        data.stat_date = pd.to_datetime(data.stat_date)
        data = data.sort_values(['city_id', 'stat_date'])
        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)
        
        
        # school hanshujia
        #school = ['shujia', 'hanjia']
        # passengers
        """
        passengers = ['user_level_high', 'user_level_weixi', 'user_level_fazhan',
                      'user_level_qianli', 'user_level_low', 'user_level_liushi',
                      'user_level_new']
        """
        # general weather feature
        data['temperature_low'] = data.groupby('city_id').temperature.apply(lambda x: 1 * (x < x.quantile(0.1)))
        data['temperature_high'] = data.groupby('city_id').temperature.apply(lambda x: 1 * (x > x.quantile(0.9)))
        data['temperature_square'] = np.square(data.temperature)

        """
        day_weather = ['rain_hour_cnt', 'temperature',
                       'temperature_square', 'wind_speed', 'temperature_low',
                       'temperature_high']
        """
        day_weather = ['temperature',
               'temperature_square', 'wind_speed', 'temperature_low',
               'temperature_high']
        # rain_intensity in a whole day
        rain = []
        hour_weather_intens = ['intensity_0' + str(i) for i in range(0, 10)] + \
                              ['intensity_' + str(i) for i in range(10, 24)]

        rain_var = ['quarter_rainintens_morning', 'quarter_rainintens_noon',
                    'quarter_rainintens_afternoon', 'quarter_rainintens_night']
        
        data[rain_var[0]] = data[hour_weather_intens[6:11]].sum(axis=1)
        data[rain_var[1]] = data[hour_weather_intens[11:14]].sum(axis=1)
        data[rain_var[2]] = data[hour_weather_intens[14:17]].sum(axis=1)
        data[rain_var[3]] = data[hour_weather_intens[17:23]].sum(axis=1)
        
        discretize_num = 3
        discretize_rate = 1.0 / discretize_num
        discretize_rain_var = []
        rain_threshold = 0.05


        def _gen_value(x, quan_ratio):
            return x[x >= rain_threshold].quantile(quan_ratio)


        for item in rain_var:

            data[item + 'no_rain'] = data.groupby('city_id')[item].apply(lambda x: 1 * (
                        (x < rain_threshold) ))

            for times in range(discretize_num):
                if times != (discretize_num - 1):
                    data[item + str(times)] = data.groupby('city_id')[item].apply(lambda x: 1 * (
                        (x < _gen_value(x, discretize_rate*(times+1))) & ( x >= _gen_value(x, discretize_rate*times))))
                else:
                    data[item + str(times)] = data.groupby('city_id')[item].apply(lambda x: 1 * (
                        (x <= _gen_value(x, discretize_rate*(times+1))) & ( x>=_gen_value(x, discretize_rate*times))))
                discretize_rain_var.append(item + str(times))
            discretize_rain_var.append(item + 'no_rain')

        '''
        hour_threshold = 0.25
        discretize_num = 3
        discretize_rate = 1.0 / discretize_num

        def _gen_hour(x, quan_ratio):
            return x[x >= hour_threshold].quantile(quan_ratio)


        item = 'rain_hour_cnt'
        data['rain_hour_cnt_no_rain'] = data.groupby('city_id')[item].apply(lambda x: 1 * (
                        (x < hour_threshold) ))

        for times in range(discretize_num):

            if times != (discretize_num - 1):
                data['rain_hour_cnt'  + str(times)] = data.groupby('city_id')[item].apply(lambda x: 1 * (
                    (x < _gen_hour(x, discretize_rate*(times+1))) & ( x >= _gen_hour(x, discretize_rate*times))))
            else:
                data['rain_hour_cnt'  + str(times)] = data.groupby('city_id')[item].apply(lambda x: 1 * (
                    (x <= _gen_hour(x, discretize_rate*(times+1))) & ( x>=_gen_hour(x, discretize_rate*times))))
            discretize_rain_var.append('rain_hour_cnt' + str(times))
        discretize_rain_var.append('rain_hour_cnt_no_rain')
        '''
        rain.extend(discretize_rain_var)
        
        #rain = []

        weather_cond = ['clear_morning', 'clear_noon',  'clear_afternoon', 'clear_night',
                        'partly_morning', 'partly_noon', 'partly_afternoon', 'partly_night',
                        'cloudy_morning', 'cloudy_noon', 'cloudy_afternoon', 'cloudy_night',
                        'rain_morning', 'rain_noon', 'rain_afternoon', 'rain_night']
        clear_cols = ['hour_skycon_' + str(i) + '_CLEAR' for i in range(24)]
        partly_cols = ['hour_skycon_' + str(i) + '_PARTLY' for i in range(24)]
        cloudy_clos = ['hour_skycon_' + str(i) + '_CLOUDY' for i in range(24)]
        rain_cols = ['hour_skycon_' + str(i) + '_RAIN' for i in range(24)]


        data['clear_morning'] = data[clear_cols[6:11]].sum(axis=1)
        data['clear_noon'] = data[clear_cols[11:14]].sum(axis=1)
        data['clear_afternoon'] = data[clear_cols[14:17]].sum(axis=1)
        data['clear_night'] = data[clear_cols[17:23]].sum(axis=1)

        data['partly_morning'] = data[partly_cols[6:11]].sum(axis=1)
        data['partly_noon'] = data[partly_cols[11:14]].sum(axis=1)
        data['partly_afternoon'] = data[partly_cols[14:17]].sum(axis=1)
        data['partly_night'] = data[partly_cols[17:23]].sum(axis=1)

        data['cloudy_morning'] = data[cloudy_clos[6:11]].sum(axis=1)
        data['cloudy_noon'] = data[cloudy_clos[11:14]].sum(axis=1)
        data['cloudy_afternoon'] = data[cloudy_clos[14:17]].sum(axis=1)
        data['cloudy_night'] = data[cloudy_clos[17:23]].sum(axis=1)

        data['rain_morning'] = data[rain_cols[6:11]].sum(axis=1)
        data['rain_noon'] = data[rain_cols[11:14]].sum(axis=1)
        data['rain_afternoon'] = data[rain_cols[14:17]].sum(axis=1)
        data['rain_night'] = data[rain_cols[17:23]].sum(axis=1)


        # price, subsidy
        data['per_subsidy_b'] = np.maximum(data['total_subsidy_b'] * 1.0 / data['total_finish_order_cnt'], 0)
        data['per_subsidy_b_square'] = np.square(data['per_subsidy_b'])
        data['per_subsidy_c'] = np.maximum(data['total_subsidy_c'] * 1.0 / data['total_finish_order_cnt'], 0)
        data['per_subsidy_c_square'] = np.square(data['per_subsidy_c'])


        # add B, C transformation
        data['per_subsidy_b_sqrt'] = np.sqrt(data['per_subsidy_b'])
        data['per_subsidy_c_sqrt'] = np.sqrt(data['per_subsidy_c'])
        data['logistic_b'] = 1.0 / (1.0 + np.exp(-data['per_subsidy_b']))
        data['logistic_c'] = 1.0 / (1.0 + np.exp(-data['per_subsidy_c']))
        data['sum_bc'] = data['per_subsidy_b'] + data['per_subsidy_c']
        #data['ratio_bc'] = np.maximum(data['per_subsidy_b'], 1e-2) / np.maximum(data['per_subsidy_c'], 1e-2)
        #data['ratio_cb'] = np.maximum(data['per_subsidy_c'], 1e-2) / np.maximum(data['per_subsidy_b'], 1e-2)



        #data['kuaiche_price_square'] = np.square(data.kuaiche_price)
        #subsidy = ['per_subsidy_b', 'per_subsidy_c', 'kuaiche_price_square',
        #           'per_subsidy_b_square', 'per_subsidy_c_square', 'kuaiche_price']
        
        """
        subsidy = ['per_subsidy_b', 'per_subsidy_c',
                   'per_subsidy_b_square', 'per_subsidy_c_square']
        """
        

        
        subsidy = ['per_subsidy_b', 'per_subsidy_c',
                   'per_subsidy_b_square', 'per_subsidy_c_square',
                   'per_subsidy_b_sqrt', 'per_subsidy_c_sqrt',
                   'logistic_b', 'logistic_c',
                   'sum_bc']
        


        # generate original predictors list
        # predictors = day_weather + rain + subsidy + ['stat_date', 'city_id']
        predictors = day_weather + rain + subsidy + weather_cond + ['stat_date', 'city_id']
        # generate response list

        """
        data['calls_online_time_ratio'] = data['total_yes_call_need_ord_cnt'] / data['online_time']
        response = ['total_yes_call_need_ord_cnt', 'online_time', 'total_finish_order_cnt',
                    'calls_online_time_ratio', 'total_gmv', 'strive_order_cnt', 'total_no_call_order_cnt']
        """


        #data['calls_online_time_ratio'] = data['total_yes_call_need_ord_cnt'] / data['online_time']
        response = ['total_yes_call_need_ord_cnt', 'online_time', 'total_finish_order_cnt',
                    'total_gmv', 'strive_order_cnt', 'total_no_call_order_cnt']


        # sort by time
        data = data.sort_values(['city_id', 'stat_date'])
        data.stat_date = [datetime.datetime.strftime(i,'%Y-%m-%d') for i in data.stat_date]
        label = data[response]
        data = data[predictors].copy()

        # month and weekday feature
        data.stat_date = pd.to_datetime(data.stat_date)
        data['weekday'] = [i.weekday() for i in data.stat_date]
        data = pd.get_dummies(data, columns=['weekday'])
        data['weekday'] = [i.weekday() for i in data.stat_date]
        
        # add cross feature for weekend and rain
        
        def weekend(x):
            if x >= 5:
                return 1
            else:
                return 0
        
        def cross(p, q):
            result = []
            for i in range(len(p)):
                if p[i]:
                    temp = copy.deepcopy(q[i])
                    temp.extend(len(q[i]) * [0])  
                else:
                    temp = len(q[i]) * [0]
                    temp.extend(q[i])
                result.append(temp)
            return result
            
        data['weather_weekend'] = [weekend(x) for x in data['weekday']]
        
        cross_weekend_rain = cross(data['weather_weekend'].values.tolist(), 
                                   data[rain].values.tolist())
        
        cross_weekend_rain = np.array(cross_weekend_rain)
        weather_weekend_cols = []
        for i in range(len(cross_weekend_rain[0])):
            data['cross_weekend_rain_' + str(i)] = cross_weekend_rain[:, i]
            weather_weekend_cols.append('cross_weekend_rain_' + str(i))
        
        # last four weeks' same weekday's label
        for i in [1, 2, 3, 4]:
            # to_lag = label.join(data[['weekday', 'city_id']])
            to_lag = np.log(label).join(data[['weekday', 'city_id']])
            to_lag = to_lag.groupby(['city_id', 'weekday']).shift(i)
            to_lag.columns = [j + '_last_' + str(i) + '_week' for j in response]
            data = data.join(to_lag)


        # last 2,3 days' label
        # for i in [2, 3, 4, 5, 6]:
        for i in [1, 2, 3, 4, 5, 6]:
            # to_lag = label.join(data[['city_id']])
            to_lag = np.log(label).join(data[['city_id']])
            to_lag = to_lag.groupby(['city_id']).shift(i)
            to_lag.columns = [j + '_last_' + str(i) + '_day' 
                                    for j in response]
            data = data.join(to_lag)

        """
        # second order trend
        begin_train_day = pd.to_datetime(self.begin_train_day)
        data['trend_1'] = [(i - begin_train_day).days for i in data.stat_date]
        data['trend_2'] = [(i - begin_train_day).days ** 2 for i in data.stat_date]

        #  local trend
        local_length = 30
        data['stage'] = data['trend_1'] // local_length
        stage_num = max(data['stage']) + 1
        data = pd.get_dummies(data, columns=['stage'])
        var_name = ['stage_' + str(i) for i in range(stage_num)]
        newvar_name = ['stage_k_' + str(i) for i in range(stage_num)]
        data[newvar_name] = data[var_name].apply(lambda x: x * data['trend_1'])
        """
        
        # add city_id into label
        label = label.join(data[['city_id']])
        
        # forward and backward fill na
        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)
        data.stat_date = [datetime.datetime.strftime(i,'%Y-%m-%d') for i in data.stat_date]
         
        return data, label, rain, weather_weekend_cols

    def _normalize(self, to_normalize, city_data):
        train = city_data.loc[(pd.to_datetime(city_data.stat_date) >= pd.to_datetime(self.begin_train_day)) &
                              (pd.to_datetime(city_data.stat_date) <= pd.to_datetime(self.end_train_day))].copy()

        #train_val = city_data.loc[(city_data.stat_date >= self.begin_train_day) &
        #                     (city_data.stat_date <= self.end_val_day)].copy()
        
        train_mean = train[to_normalize].mean()
        #train_std = train[to_normalize].var() ** 0.5 + 1e-5
        train_std = train[to_normalize].std() + 1e-5

        #train_val_mean = train_val[to_normalize].mean()
        #train_val_std = train_std[to_normalize].var() ** 0.5 + 1e-5

        # data1 is used for test dataset.
        city_data = city_data.copy()
        #city_data1 = city_data.copy()

        city_data.loc[:,to_normalize] = (city_data.loc[:,to_normalize] - train_mean) / train_std
        
        #city_data1.loc[:,to_normalize] = (city_data1.loc[:,to_normalize] - train_val_mean)/(train_val_std)

        #return city_data, city_data1
        return city_data


    def _get_mean_std(self, infer_replace, city_data):
        train = city_data.loc[(pd.to_datetime(city_data.stat_date) >= pd.to_datetime(self.begin_train_day)) &
                              (pd.to_datetime(city_data.stat_date) <= pd.to_datetime(self.end_train_day))]
        train_mean = train[infer_replace].mean()
        train_std = train[infer_replace].std() + 1e-5

        return train_mean, train_std

       

    """
    normalize_type:
    0: no normalize
    1: normalize all
    2: normalize per city
    """
    def gen_whole_data(self, normalize_type):
        

        # infer_replace = ['per_subsidy_b', 'per_subsidy_c', 'per_subsidy_b_square', 'per_subsidy_c_square']

        
        infer_replace = ['per_subsidy_b', 'per_subsidy_c',
                         'per_subsidy_b_square', 'per_subsidy_c_square',
                         'per_subsidy_b_sqrt', 'per_subsidy_c_sqrt',
                         'logistic_b', 'logistic_c',
                         'sum_bc']
        

        idx = 0
        city_map = {}
        for city_id in set(self.city):
            city_map[city_id] = idx
            idx += 1

        #data, label = self._gen_features()
        data, label, rain, weather_weekend_cols = self._gen_features()


        ignore_list = ['stat_date', 'shujia', 'hanjia', 'city_id',
                       'weekday_0', 'weekday_1',
                       'weekday_2', 'weekday_3',
                       'weekday_4', 'weekday_5',
                       'weekday_6', 'stage_0',
                       'stage_1', 'stage_2',
                       'stage_3', 'stage_4',
                       'stage_5', 'stage_6',
                       'stage_7', 'stage_8',
                       'stage_9', 'stage_10',
                       'stage_11', 'stage_12',
                       'stage_13', 'stage_14',
                       'temperature_low', 'temperature_high', 'weather_weekend']
        ignore_list = ignore_list + rain + weather_weekend_cols
        ignore_set = set(ignore_list)

        to_normalize = list(set(data.columns).difference(ignore_set))

        
        """
        to_normalize = list(set(data.columns).difference({
            'stat_date', 'shujia', 'hanjia', 'city_id',
            'weekday_0', 'weekday_1',
            'weekday_2', 'weekday_3',
            'weekday_4', 'weekday_5',
            'weekday_6', 'stage_0',
            'stage_1', 'stage_2',
            'stage_3', 'stage_4',
            'stage_5', 'stage_6',
            'stage_7', 'stage_8',
            'stage_9', 'stage_10',
            'stage_11', 'stage_12',
            'stage_13', 'stage_14',
            'temperature_low', 'temperature_high',
            'weather_weekend'}))
        """


        the_end = pd.to_datetime(self.begin_infer_day)
        the_end = the_end - datetime.timedelta(2)
        the_begin = the_end - datetime.timedelta(370)
        the_all = pd.concat([data[['stat_date']], label], axis=1)
        the_all = the_all.loc[(pd.to_datetime(the_all.stat_date) >= the_begin) & (pd.to_datetime(the_all.stat_date) <= the_end)]
        city_max = the_all.groupby('city_id').max()
        city_min = the_all.groupby('city_id').min()

        train_mean = 0
        train_std = 0

        if normalize_type == 2:

            grouped = data.groupby('city_id', as_index = False)
            # need self.city to be sorted!
            split_data = [grouped.get_group(city_id) for city_id in self.city]

            mean_std_list = [self._get_mean_std(infer_replace, city_data) for city_data in split_data]

            train_mean, train_std = zip(*mean_std_list)

            train_mean = pd.DataFrame(list(train_mean))
            train_std = pd.DataFrame(list(train_std))
            train_mean['city_id'] = self.city
            train_std['city_id'] = self.city

            normalize_data = [self._normalize(to_normalize, city_data) for city_data in split_data]
            #data = pd.concat([data[0] for data in normalize_data])
            data = pd.concat([data for data in normalize_data])
        if normalize_type == 1:
            train_mean, train_std = self._get_mean_std(infer_replace, data)
            data = self._normalize(to_normalize, data)
            

        date = data.stat_date.copy()

        #date = pd.to_datetime(data.stat_date)

        data = data.drop('stat_date', axis=1)
        train_index = (pd.to_datetime(date) >= pd.to_datetime(self.begin_train_day)) & \
                      (pd.to_datetime(date) <= pd.to_datetime(self.end_train_day))
        val_index = (pd.to_datetime(date) >= pd.to_datetime(self.begin_val_day)) & \
                    (pd.to_datetime(date) <= pd.to_datetime(self.end_val_day))

    
        data.drop(['weekday_0', 'weekday_1',
                    'weekday_2', 'weekday_3',
                    'weekday_4', 'weekday_5',
                    'weekday_6', 'weekday'], axis=1, inplace=True)

        # train
        train_x = data.loc[train_index].copy()
        train_y = label.loc[train_index].copy()

        train_embed_weekday = np.array([pd.to_datetime(i).weekday() for i in date.loc[train_index]])
        train_embed_month = np.array([pd.to_datetime(i).month for i in date.loc[train_index]])
        train_embed_city = np.array([city_map[i] for i in train_x.city_id])
        train_real_city = np.array(train_x.city_id)

        train_x.drop(['city_id'], axis=1, inplace=True)

        # val
        val_x = data.loc[val_index].copy()
        val_y = label.loc[val_index].copy()

        val_embed_weekday = np.array([pd.to_datetime(i).weekday() for i in date.loc[val_index]])
        val_embed_month = np.array([pd.to_datetime(i).month for i in date.loc[val_index]])
        val_embed_city = np.array([city_map[i] for i in val_x.city_id])
        val_real_city = np.array(val_x.city_id)

        val_x.drop(['city_id'], axis=1, inplace=True)
        
        infer_index = (pd.to_datetime(date) >= pd.to_datetime(self.begin_infer_day) - pd.DateOffset(7)) & \
                      (pd.to_datetime(date) <= pd.to_datetime(self.begin_infer_day))

        infer_x = data.loc[infer_index].copy()
        infer_y = label.loc[infer_index].copy()

        infer_embed_weekday = np.array([pd.to_datetime(i).weekday() for i in date.loc[infer_index]])
        infer_embed_month = np.array([pd.to_datetime(i).month for i in date.loc[infer_index]])
        infer_embed_city = np.array([city_map[i] for i in infer_x.city_id])
        infer_real_city = np.array(infer_x.city_id)

        infer_city_map = {}
        for i in range(len(infer_real_city)):
            infer_city_map[infer_real_city[i]] = i

        #infer_x.drop(['city_id'], axis=1, inplace=True)

        return [train_x, train_embed_weekday, train_embed_month, 
                train_embed_city, train_real_city, train_y],\
                [val_x, val_embed_weekday, val_embed_month,
                val_embed_city, val_real_city, val_y],\
                [infer_x, infer_embed_weekday, infer_embed_month,
                infer_embed_city, infer_city_map, infer_y], city_max, city_min, train_mean, train_std


