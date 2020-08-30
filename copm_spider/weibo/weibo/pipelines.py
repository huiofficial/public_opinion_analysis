# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
import pymysql
from aip import AipNlp
import time
import random
import hashlib
import json
import re

class WeiboPipeline(object):

    def __init__(self, host, port, user, passwd):
        self.host = host
        self.port = port
        self.user = user
        self.passwd = passwd

    @classmethod
    def from_crawler(cls, crawler):
        return cls(
            host = crawler.settings.get('MYSQL_HOST'),
            port = crawler.settings.get('MYSQL_PORT'),
            user = crawler.settings.get('MYSQL_USER'),
            passwd = crawler.settings.get('MYSQL_PASSWD')
        )

    def open_spider(self, spider):
        self.conn = pymysql.connect(host=self.host, port=self.port, user=self.user, passwd=self.passwd, db='copm', charset='utf8')
        self.cursor = self.conn.cursor()
        
        sql = 'select keyword_main, activate from copm.config'
        self.cursor.execute(sql)
        result = self.cursor.fetchall()
        spider.keyword = set()
        for item in result:
            if item[1] != 0:
                spider.keyword.add(item[0])
                
        sql = 'select word, id from copm.filterword'
        self.cursor.execute(sql)
        result = self.cursor.fetchall()
        spider.filterword = set()
        for item in result:
            spider.filterword.add(item[0])
        sql = 'select name,code from copm.city'
        self.cursor.execute(sql)
        result = self.cursor.fetchall()
        city_names = []
        city_code = {}
        for item in result:
            city_names.append(item[0])
            city_code[item[0]] = item[1]
        self.city_pattern = '|'.join(city_names)
        self.city_code = city_code
        
        APP_ID = '11576120'
        API_KEY = 'WOSriMMnS9eSKGftEOtx5rb6'
        SECRET_KEY = '0Q6NaE89wbhRRjSVspjiDiiT6ZnGVllE'
        self.client = AipNlp(APP_ID, API_KEY, SECRET_KEY)

    def close_spider(self, spider):
        self.cursor.close()
        self.conn.close()

    def process_item(self, item, spider):
        city = re.search(self.city_pattern, item['weibo_content'])
        if city is None:
            city = '不限'
        else:
            city = city.group()
        
        text = item['weibo_content']
        if text.strip() == '':
            return item
 
        sql = '''INSERT IGNORE INTO weibo (weibo_id, weibo_url, weibo_content, weibo_repost, weibo_like, weibo_comment, weibo_datetime,
        user_id, user_name, user_url, user_verified, user_follow, user_fan, user_weibo) VALUES('{weibo_id}', '{weibo_url}',
        '{weibo_content}', {weibo_repost}, {weibo_like}, {weibo_comment}, '{weibo_ctime}', '{user_id}', '{user_name}', '{user_url}',
        {user_verified}, {user_follow}, {user_fan}, {user_weibo})'''.format(**item)
        self.cursor.execute(sql)
        text = text.encode('gbk', 'ignore').decode('gbk')
        while True:
            result = self.client.sentimentClassify(text)
            if 'error_code' in result.keys() and result['error_code'] == 18:
                time.sleep(random.randint(0, 3))
            else:
                break
        if 'items' not in result.keys():
            emotion = 0
        else:
            emotion = result['items'][0]['sentiment']
            if emotion == 0:
                emotion = 2
            elif emotion == 1:
                emotion = 0
            elif emotion == 2:
                emotion = 1
        md5 = hashlib.md5()
        uniqueStr = 'weibo-{}'.format(item['weibo_id'])
        md5.update(uniqueStr.encode(encoding='utf-8'))
        uniqueCode = md5.hexdigest()
        writeDate = time.strftime('%Y-%m-%d',time.localtime(time.time()))
        sql = '''INSERT IGNORE INTO content (address, title, date, abstract, keyword, mediaName, emotion, uniqueCode, writeDate, city) VALUES('{address}','{title}',
                '{datetime}', '{abstract}', '{keyword}', '{mediaName}', {emotion}, '{uniqueCode}', '{writeDate}', '{city}')'''.format(address=item['weibo_url'], title=item['weibo_content'][:20],
                datetime=item['weibo_ctime'], abstract=item['weibo_content'], keyword=item['keyword'], mediaName=item['mediaName'], emotion=emotion, uniqueCode=uniqueCode, writeDate=writeDate, city=self.city_code[city])
        self.cursor.execute(sql)
        self.conn.commit()
        return item
