import json
import numpy as np
import pandas as pd
from src.data_processing import data_processing


with open('test/synthetic_data.json', 'rt') as synth_data:
    test_data = json.load(synth_data)


def test_config():
    assert isinstance(data_processing.config, dict)


def test_hc_map():
    assert isinstance(data_processing.hc_map, dict)


def test_clean_text_column():
    df = pd.DataFrame(test_data['data_process_test'])
    df = df.replace('', np.nan)
    
    title_df = data_processing.clean_text_column(df.copy(), 'AwardTitle', 'title')
    assert title_df['AwardTitle'].to_list() == test_data['clean_titles']
    
    abstract_df = data_processing.clean_text_column(df.copy(), 'AwardAbstract', 'abstract')
    assert abstract_df['AwardAbstract'].to_list() == test_data['clean_abstracts']


def test_process_tests():
    df = pd.DataFrame(test_data['data_process_test'])
    df = df.replace('', np.nan)

    processed_df = data_processing.process_texts(df.copy(), min_char_len=15)

    assert processed_df['AllText'].to_list() == test_data['clean_alltext']
