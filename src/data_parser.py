import os
import natsort
import pandas as pd
import numpy as np
from utils.path_utils import PathUtils
import logging


# fetch the data directory and construct filenames
data_directory_path = PathUtils().get_data_directory()
raw_file_name = os.path.realpath(os.path.join(PathUtils().get_rawdata_directory(), '200_ds.txt'))
fsplit_path = os.path.realpath(os.path.join(data_directory_path, 'file_split'))
fout_path = os.path.realpath(os.path.join(data_directory_path, 'file_output'))
ext = ('.txt')
lambda_val = []

log = logging.getLogger(__name__)


class DataParser():
    
    # Split the data into small chunks based on a literal '#Parameters' as split criteria
    # log.info("File Path : ", fsplit_path)
    def parse_freq_data(self):
        with open(raw_file_name) as f:
            wsplit = ''
            f_out = None
            id = 1
            for line in f:
                # when a match is found , create a new output file
                if line.startswith('#Parameters'):
                    wsplit = line.split(';')[2]             # gets 'lambda=7' from this line: Parameters = {d=0.175; l2=3.5; lambda=7; r=0.023583}
                    wsplit = wsplit.split('=')[1]           # gets '7' from 'lambda=7'
                    lambda_val.append(float(wsplit))
                    title = 'file_' + str(id)
                    id = id + 1
                    # log.info(title)
                    if f_out:
                        f_out.close()
                    split_file = os.path.join(fsplit_path, f'{title}.txt')
                    f_out = open(os.path.realpath(split_file), 'w')
                if f_out:
                    f_out.write(line)
            if f_out:
                f_out.close()
        # end of file split, convert file output
        self.create_file_output()
        
        
    def create_file_output(self):
        # drop the lines with text (this is done for creating the dataframe just with the values)
        id = 1
        flist = os.listdir(fsplit_path)
        flist = natsort.natsorted(flist)

        for file in flist:
            if file.endswith(ext):
                # log.info(id, " : ", file)
                
                split_file_path = os.path.join(fsplit_path, file)
                with open(os.path.realpath(split_file_path), 'r') as fin:
                    data = fin.read().splitlines(True)
                    
                out_file = os.path.join(fout_path, f'file{id}_out.txt')
                with open(os.path.realpath(out_file), 'w') as fout:
                    fout.writelines(data[3:])  # drop the first 3 lines of text inside the file
                    id = id + 1
                if fout:
                    fout.close()
            else:
                continue
            
            
    def process_data(self):
        columns_header = ['Frequency', 'S1_Real[RE]',
                          'S1_Imaginary[Im]', 'Ref.Imp. [Re]', 'Ref.Imp. [Im]']
        self.df_dict = {}
        i = 0

        files = os.listdir(fout_path)
        files = natsort.natsorted(files)

        for f in files:
            if f.endswith(ext) and i < len(lambda_val):
                out_file = os.path.join(fout_path, f)
                val = pd.read_csv(os.path.realpath(out_file), delim_whitespace=True, names=columns_header)
                self.df_dict[lambda_val[i]] = val
                i += 1
            else:
                continue
            
    
    def create_data_frame(self):
        df_list = []
        for k, val in self.df_dict.items():
            val.drop(val.iloc[:, 3:], inplace=True, axis='columns')
            val['Lambda'] = k
            # update the data frame dictionary with lambda param(as key)
            # and corresponding data as value
            self.df_dict.update({k: val})
            # list of dataframes containing data corresponding to each lambda param
            df_list.append(val)

        # log.info(len(df_list))

        # calculate the magnitude using real and imaginary values from data
        for dframe in df_list:
            for idx in dframe.index:
                # dframe.at[idx, 'Magnitude'] = sqrt(pow(dframe['S1_Real[RE]'][idx], 2) + pow(dframe['S1_Imaginary[Im]'][idx], 2))
                cn = complex(dframe['S1_Real[RE]'][idx],
                             dframe['S1_Imaginary[Im]'][idx])
                dframe.at[idx, 'Magnitude'] = abs(cn)
                dframe.at[idx, 'Phase(Deg)'] = np.angle(cn, deg=True)
                dframe.at[idx, 'Phase(Rad)'] = np.angle(cn)

        self.frf_df = pd.concat(df_list, ignore_index=True)
        
        final_frf_csv_file = os.path.join(data_directory_path, 'final_frf_data.csv')
        self.frf_df.to_csv(os.path.realpath(final_frf_csv_file), float_format='%.14f')
        
        final_frf_txt_file = os.path.join(data_directory_path, 'final_frf_data.txt')
        self.frf_df.to_csv(os.path.realpath(final_frf_txt_file), sep='\t', index=False, float_format='%.14f')
        return self.frf_df
    
    
    def get_freq_data(self) -> pd.DataFrame:
        final_frf_data_file = os.path.join(data_directory_path, 'final_frf_data.csv')
        f_name = os.path.realpath(final_frf_data_file)
        if os.path.isfile(f_name):
            self.frf_df = pd.read_csv(f_name)
        else:
            # doesn't exist
            log.info("DataParser():File not found.")
        return self.frf_df