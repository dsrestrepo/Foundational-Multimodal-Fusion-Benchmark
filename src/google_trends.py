import os
import matplotlib.pyplot as plt
import pandas as pd
from pytrends.request import TrendReq as UTrendReq

GET_METHOD = "get"


class TrendReq(UTrendReq):
    """
    Child class of pytrends' TrendReq
    This allows to change the header to avoid 429 errors

    Source: https://stackoverflow.com/questions/50571317/pytrends-the-request-failed-google-returned-a-response-with-code-429

    Args:
        UTrendReq (TrendReq): native TrendReq object from pytrends
    """

    def _get_data(self, url, method=GET_METHOD, trim_chars=0, **kwargs):
        return super()._get_data(
            url, method=GET_METHOD, trim_chars=trim_chars, headers=headers,
            **kwargs
        )


def plot_top_queries(variable_data, variable_name, top_n=10):
    """
    Plots the top N queries for a given variable.

    Parameters:
    - variable_data (DataFrame): DataFrame containing 'query' and 'value' columns.
    - variable_name (str): Name of the variable for labeling the plot.
    - top_n (int): Number of top queries to plot (default is 5).

    Returns:
    - None: Displays the bar plot using matplotlib.

    Example:
    >>> plot_top_queries(data['top'], 'Top')
    """
    # Extract top N queries and values
    top_queries = variable_data.head(top_n)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.barh(top_queries['query'], top_queries['value'], color='skyblue')
    plt.xlabel('Value')
    plt.ylabel('Query')
    plt.title(f'Top {top_n} {variable_name} Queries')
    plt.gca().invert_yaxis()  # To have the highest value at the top
    plt.show()


def get_related_queries(query, iso='US', lang='en-US', plot=True, top_n=10):
    """
    Given a query, returns related queries as shown on Google Trends' Related Queries section.

    Parameters:
    - query (str or list): The query or set of queries for which related queries are to be retrieved.
    - iso (str): The ISO code of the location. Default is 'US'.
    - lang (str): The language to use. Default is 'en-US'.

    Returns:
    - dict: A dictionary containing related queries for each input query.
    """
    
    
    pytrends = TrendReq(hl=lang, tz=360, geo=iso)
    pytrends.build_payload([query], cat=0, timeframe='today 5-y', geo=iso, gprop='')

    related_queries_dict = pytrends.related_queries()
    related_topics_dict = pytrends.related_topics()
    
    if plot:
        plot_top_queries(related_queries_dict[query]['top'], 'Top', top_n)
        plot_top_queries(related_queries_dict[query]['rising'], 'Rising', top_n)
        
    
    return {'queries': related_queries_dict, 'topics': related_topics_dict}


def get_interest_over_time(iso_code, keywords, time_range='2016-01-01 2023-09-01', lang='en-US', plot=True):
    """
    Given an iso code of a location, keywords, and a time range (default 5 years),
    searches for the interest over time, generates a DataFrame, and optionally plots the data.

    Parameters:
    - iso_code (str): The ISO code of the location (e.g., 'US' for United States).
    - keywords (str or list): The keyword or list of keywords for which interest over time is to be retrieved.
    - time_range (str): The time range for the interest data. Default is 'today 5-y'.
    - lang (str): The language to use. Default is 'en-US'.
    - plot (bool): Whether to plot the interest over time. Default is False.

    Returns:
    - pd.DataFrame: A DataFrame containing the interest over time data.
    """
    pytrends = TrendReq(hl=lang, tz=360, geo=iso_code)
    
    # Convert single keyword to list for consistency
    if not isinstance(keywords, list):
        keywords = [keywords]

    pytrends.build_payload(keywords, cat=0, timeframe=time_range, geo=iso_code, gprop='')

    # Get interest over time data
    interest_over_time_df = pytrends.interest_over_time()

    if plot:
        # Plot interest over time
        interest_over_time_df.plot(title='Interest Over Time')
        plt.xlabel('Date')
        plt.ylabel('Interest Index')
        plt.legend(loc='upper left')
        plt.show()

    return interest_over_time_df




def set_URL(search_term, GEO, initial_date='2016-01-01', final_date='2023-09-01', HL="en"):
    """
    Generates a Google Trends URL for exploring the popularity of a search term over time.

    Parameters:
    - search_term (str): The search term or topic to explore on Google Trends.
    - GEO (str): The geographical location for which the trends should be analyzed. This should be a valid country code or region.
    - initial_date (str, optional): The start date for the trend analysis in the format 'YYYY-MM-DD'. Default is '2016-01-01'.
    - final_date (str, optional): The end date for the trend analysis in the format 'YYYY-MM-DD'. Default is '2023-09-01'.
    - HL (str, optional): The language parameter for the Trends page. Default is 'en' (English).

    Returns:
    - str: The generated Google Trends URL based on the provided parameters.
    
    Example:
    >>> set_URL("python programming", "US", initial_date='2020-01-01', final_date='2023-01-01', HL="en")
    'https://trends.google.com/trends/explore?date=2020-01-01%202023-01-01&q=python%20programming&geo=US&hl=en'
    """
    
    URL = f"https://trends.google.com/trends/explore?date={initial_date}%20{final_date}&q={search_term}&geo={GEO}&hl={HL}"
    
    print(f'Your URL is: {URL}')
    
    return URL


"""
import time, json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from parsel import Selector
from bs4 import BeautifulSoup

def Google_Trend_Data(url,timeframe=60, userdatadir='Path to /AppData/Local/Google/Chrome/User Data'):

    service = Service(ChromeDriverManager().install())
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--lang=en")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36")
    options.add_argument(f"--user-data-dir={userdatadir}")
    driver = webdriver.Chrome(service=service,options=options)
    driver.get(url)
    WebDriverWait(driver, 10000).until(EC.visibility_of_element_located((By.TAG_NAME, 'body')))
    time.sleep(5)
    selector = driver.page_source
    soup=BeautifulSoup(selector,'html.parser')
    columns=[]
    df=pd.DataFrame(columns=columns)
    table=soup.select('td')
        
    odd_i = []
    even_i = []
    for i in range(0, len(table)):
        if i % 2:
            even_i.append(table[i])
        else :
            odd_i.append(table[i])
 
   
    date_table=odd_i
    value_table=even_i
    date_table=date_table[::-1]
    int_value_table=[]
    for item in value_table:
        item=str(item)
        item=item.replace("<td>","")
        item=item.replace("</td>","")
        item=int(item)
        int_value_table.append(item)
    int_value_table.reverse()
    df['Date']=date_table[0:timeframe]
    df['Interest over Time']=int_value_table[0:timeframe]
    mean=df['Interest over Time'].mean()

    print(df.to_string())
    print(mean)
    driver.quit()

    return df,mean
"""
