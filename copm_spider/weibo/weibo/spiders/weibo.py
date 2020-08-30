# coding:utf-8
from scrapy.spiders import Spider
from weibo.items import WeiboItem
import urllib
import scrapy
import json
import re
import pytz
import time, random
import os
import datetime

class WeiboSpider(Spider):
    name = "weibo"
    mediaName = '微博'
    filt_re = re.compile(r'<[^>]+>',re.S)
    # 对微博正文去除HTML标签

    def __init__(self, page='20', *args, **kwargs):
        super(WeiboSpider, self).__init__(*args, **kwargs)
        self.page = int(page)

    def start_requests(self):
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'
        }
        url_p1 = 'https://m.weibo.cn/api/container/getIndex?type=wb&queryVal='
        url_p2 = '&featurecode=20000320&luicode=10000011&lfid=106003type%3D1&title='
        url_p3 = '&containerid=100103type%3D2%26q%3D'
        url_p4 = '&page='
        urls = []
        self.tz = pytz.timezone('Asia/Shanghai')
        
        for key in self.keyword:
            print('Crawl KeyWord: %s' %key)
            for i in range(self.page, 0, -1):
                print("page: "+str(i))

                url = url_p1 + key + url_p2 + key + url_p3 + key + url_p4 + str(i)
                yield scrapy.Request(url=url, meta={'keyword':key}, callback=self.parse_search, headers=headers, dont_filter=True)
                sleep_time = random.random()
                # print sleep_time
                time.sleep( sleep_time )
            print("sleep 10s")
            time.sleep(10)

    def parse_search(self, response):
        res = json.loads(response.text)
        if (str(res['ok']) == '1'):
            for key in res['data']['cards'][0]['card_group']:
                item = WeiboItem()
                # weibo info
                item['weibo_id'] = key['mblog']['id']
                item['weibo_content'] = self.filt_re.sub('',key['mblog']['text'])           
                item['weibo_repost'] = key['mblog']['reposts_count']
                item['weibo_comment'] = key['mblog']['comments_count']
                item['weibo_like'] = key['mblog']['attitudes_count']

                timestr = str(key['mblog']['created_at'])
                ctime = self.get_ctime(timestr)
                item['weibo_ctime'] = ctime
                
                url_pos = key['scheme'].index("?mblogid")
                item['weibo_url'] = key['scheme'][0:url_pos]
                
                # user info
                item['user_id'] = str(key['mblog']['user']['id'])
                item['user_name'] = key['mblog']['user']['screen_name']
                item['user_weibo'] = key['mblog']['user']['statuses_count']
                item['user_follow'] = key['mblog']['user']['follow_count']
                item['user_fan'] = key['mblog']['user']['followers_count']
                item['user_verified'] = key['mblog']['user']['verified']
                
                user_url = key['mblog']['user']['profile_url']
                uurl_pos = user_url.index('?uid=')
                item['user_url'] = user_url[0:uurl_pos]
                item['mediaName'] = self.mediaName
                item['keyword'] = response.meta['keyword']
                
                # print('Weibo crawled at %s' %ctime)
                if item['weibo_content'].strip() == '':
                    return

                for word in self.filterword:
                    if item['weibo_content'].find(word) != -1:
                        return
                  
                yield item
                #yield scrapy.Request(url=key['scheme'],callback=self.parse)
        else:
            # i=i-1
            if ('msg' in res.keys()):
                print(res['msg'])

    def get_ctime(self,timestr):
        # 刚刚
        d = datetime.datetime.now(self.tz)
        # N分钟前
        if '分钟' in timestr:
            minute = int(timestr[0:timestr.index('分钟')])
            delta = datetime.timedelta(minutes=minute)
            d = d - delta
        # N小时前
        elif '小时' in timestr:
            hour = int(timestr[0:timestr.index('小时')])
            delta = datetime.timedelta(hours=hour)
            d = d - delta
        # 昨天 HH:MM
        elif '昨天' in timestr:
            pos = timestr.index(':')
            hour = int(timestr[pos - 2:pos])
            minute = int(timestr[pos + 1:pos + 3])
            delta = datetime.timedelta(days=1, hours=d.hour - hour, minutes=d.minute - minute)
            d = d - delta
        # mm-dd
        elif '-' in timestr:
            time_list = timestr.split('-')
            if len(time_list)==2:
                month = int(time_list[0])
                day = int(time_list[1])
                d = d.replace(month=month, day=day)
            else:
                year = int(time_list[0])
                month = int(time_list[1])
                day = int(time_list[2])
                d = d.replace(year=year, month=month, day=day)
        ctime = d.strftime('%Y-%m-%d %H:%M:%S')
        return ctime
        
        
