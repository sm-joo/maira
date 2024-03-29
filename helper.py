import os
import re
import csv
import math
import time
import json
import random
import finnhub
import datasets
import pandas as pd
import yfinance as yf
from datetime import date, datetime, timedelta
from collections import defaultdict
from datasets import Dataset
from openai import OpenAI



def get_company_prompt(symbol):

    profile = finnhub_client.company_profile2(symbol=symbol)

    company_template = "[기업소개]:\n\n{name}은 {finnhubIndustry}섹터의 기업입니다. {ipo}에 상장하였으며, 오늘날 주가총액은 {currency} {marketCapitalization:.2f}입니다. "

    formatted_str = company_template.format(**profile)

    return formatted_str


def get_prompt_by_row(symbol, row):

    start_date = row['Start Date'] if isinstance(row['Start Date'], str) else row['Start Date'].strftime('%Y-%m-%d')
    end_date = row['End Date'] if isinstance(row['End Date'], str) else row['End Date'].strftime('%Y-%m-%d')
    term = '상승하였습니다' if row['End Price'] > row['Start Price'] else '하락하였습니다'
    head = "{}부터 {}까지, {}의 주식가격은 {:.2f}에서 {:.2f}으로 {}. 관련된 뉴스 리스트는 아래와 같습니다 (해외 주식은 영문 기사 기반으로 작성됩니다.):\n\n".format(
        start_date, end_date, symbol, row['Start Price'], row['End Price'], term)

    news = json.loads(row["News"])
    news = ["[headline]: {}\n[summary]: {}\n".format(
        n['headline'], n['summary']) for n in news if n['date'][:8] <= end_date.replace('-', '') and \
        not n['summary'].startswith("Looking for stock market analysis and research with proves results?")]

    basics = json.loads(row['Basics'])
    if basics:
        basics = "{} 관련하여 최근 {}에 보고된 재무정보는 아래와 같습니다.:\n\n[기본재무정보]:\n\n".format(
            symbol, basics['period']) + "\n".join(f"{k}: {v}" for k, v in basics.items() if k != 'period')
    else:
        basics = "[기본재무정보]:\n\n 관련 정보가 없습니다."

    return head, news, basics


def sample_news(news, k=5):

    return [news[i] for i in sorted(random.sample(range(len(news)), k))]


def map_bin_label(bin_lb):

    lb = bin_lb.replace('U', 'up by ')
    lb = lb.replace('D', 'down by ')
    lb = lb.replace('1', '0-1%')
    lb = lb.replace('2', '1-2%')
    lb = lb.replace('3', '2-3%')
    lb = lb.replace('4', '3-4%')
    if lb.endswith('+'):
        lb = lb.replace('5+', 'more than 5%')
#         lb = lb.replace('5+', '5+%')
    else:
        lb = lb.replace('5', '4-5%')

    return lb


#오늘 날짜
def get_curday():
    return date.today().strftime("%Y-%m-%d")

# 오늘 날짜 기준, n week 이전 날짜
def n_weeks_before(date_string, n):
    date = datetime.strptime(date_string, "%Y-%m-%d") - timedelta(days=7*n)
    return date.strftime("%Y-%m-%d")


# steps = [n_weeks_before(curday, n) for n in range(n_weeks + 1)][::-1]
# -> 오늘부터 (과거로) 일주일마다 n_week 개만큼 날짜를 찍어서 -> 역수로 전환 (과거부터 현재까지)

def get_stock_data(stock_symbol, steps):

    stock_data = yf.download(stock_symbol, steps[0], steps[-1])

    dates, prices = [], []
    available_dates = stock_data.index.format()

    for date in steps[:-1]:
        for i in range(len(stock_data)):
            if available_dates[i] >= date:
                prices.append(stock_data['Close'][i])
                dates.append(datetime.strptime(available_dates[i], "%Y-%m-%d"))
                break

    dates.append(datetime.strptime(available_dates[-1], "%Y-%m-%d"))
    prices.append(stock_data['Close'][-1])

    return pd.DataFrame({
        "Start Date": dates[:-1], "End Date": dates[1:],
        "Start Price": prices[:-1], "End Price": prices[1:]
    })



def get_news(symbol, data):

    news_list = []

    for end_date, row in data.iterrows():
        start_date = row['Start Date'].strftime('%Y-%m-%d')
        end_date = row['End Date'].strftime('%Y-%m-%d')
        print(symbol, ': ', start_date, ' - ', end_date)
        time.sleep(1) # control qpm
        weekly_news = finnhub_client.company_news(symbol, _from=start_date, to=end_date)
        weekly_news = [
            {
                "date": datetime.fromtimestamp(n['datetime']).strftime('%Y%m%d%H%M%S'),
                "headline": n['headline'],
                "summary": n['summary'],
            } for n in weekly_news
        ]
        weekly_news.sort(key=lambda x: x['date'])
        news_list.append(json.dumps(weekly_news))

    data['News'] = news_list

    return data


def get_basics(symbol, data, always=True):

    basic_financials = finnhub_client.company_basic_financials(symbol, 'all')

    final_basics, basic_list, basic_dict = [], [], defaultdict(dict)

    for metric, value_list in basic_financials['series']['quarterly'].items():
        for value in value_list:
            basic_dict[value['period']].update({metric: value['v']})

    for k, v in basic_dict.items():
        v.update({'period': k})
        basic_list.append(v)

    basic_list.sort(key=lambda x: x['period'])

    for i, row in data.iterrows():

        start_date = row['End Date'].strftime('%Y-%m-%d')
        last_start_date = START_DATE if i < 2 else data.loc[i-2, 'Start Date'].strftime('%Y-%m-%d')

        used_basic = {}
        for basic in basic_list[::-1]:
            if (always and basic['period'] < start_date) or (last_start_date <= basic['period'] < start_date):
                used_basic = basic
                break
        final_basics.append(json.dumps(used_basic))

    data['Basics'] = final_basics

    return data

def get_current_basics(symbol, curday):

    basic_financials = finnhub_client.company_basic_financials(symbol, 'all')

    final_basics, basic_list, basic_dict = [], [], defaultdict(dict)

    for metric, value_list in basic_financials['series']['quarterly'].items():
        for value in value_list:
            basic_dict[value['period']].update({metric: value['v']})

    for k, v in basic_dict.items():
        v.update({'period': k})
        basic_list.append(v)

    basic_list.sort(key=lambda x: x['period'])

    for basic in basic_list[::-1]:
        if basic['period'] <= curday:
            break

    return basic



def get_all_prompts_online(symbol, data, curday, with_basics=True):

    company_prompt = get_company_prompt(symbol)

    prev_rows = []

    for row_idx, row in data.iterrows():
        head, news, _ = get_prompt_by_row(symbol, row)
        prev_rows.append((head, news, None))

    prompt = ""
    for i in range(-len(prev_rows), 0):
        prompt += "\n" + prev_rows[i][0]
        sampled_news = sample_news(
            prev_rows[i][1],
            min(5, len(prev_rows[i][1]))
        )
        if sampled_news:
            prompt += "\n".join(sampled_news)
        else:
            prompt += "No relative news reported."



    period = "{} to {}".format(curday, n_weeks_before(curday, -1))

    if with_basics:
        basics = get_current_basics(symbol, curday)
        basics = "최근 {}에 보고된 {} 관련 재무정보는, 다음과 같습니다:\n\n[기본 재무정보]:\n\n".format(
            basics['period'], symbol) + "\n".join(f"{k}: {v}" for k, v in basics.items() if k != 'period')
    else:
        basics = "[기본 재무정보]:\n\n재무정보가 보고되지 않았습니다."

    info = company_prompt + '\n' + prompt + '\n' + basics
    prompt = info + f"\n\nBased on all the information before {curday}, let's first analyze the positive developments and potential concerns for {symbol}. Come up with 2-4 most important factors respectively and keep them concise. Most factors should be inferred from company related news. " \
        f"Then make your prediction of the {symbol} stock price movement for next week ({period}). Provide a summary analysis to support your prediction."

    return info, prompt

SYSTEM_PROMPT = "You are a seasoned stock market analyst working in South Korea. Your task is to list the positive developments and potential concerns for companies based on relevant news and basic financials from the past weeks, then provide an analysis and prediction for the companies' stock price movement for the upcoming week. " \
"Your answer format should be as follows:\n\n[Positive Developments]:\n1. ...\n\n[Potential Concerns]:\n1. ...\n\n[Prediction & Analysis]:\n...\n\n  Because you are working in South Korea, all responses should be done in Korean not in English. \n "



def prepare_data_for_company(symbol, past_weeks = 3, with_basics=True):
    curday = get_curday()
    n_weeks_before(curday, past_weeks )
    steps = [n_weeks_before(curday, n) for n in range(past_weeks + 1)][::-1]
    globals()['START_DATE'] = steps[0]
    globals()['END_DATE'] = steps[-1]

    data = get_stock_data(symbol, steps)
    data = get_news(symbol, data)

    if with_basics:
        data = get_basics(symbol, data)
    else:
        data['Basics'] = [json.dumps({})] * len(data)
    return data


def query_gpt4(symbol, past_weeks=3, with_basics=True):
    curday = get_curday()
    data= prepare_data_for_company(symbol, past_weeks, with_basics)
    prompts = get_all_prompts_online(symbol, data, curday, with_basics)

    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompts[0]}
            ]
    )
    return prompts, completion


