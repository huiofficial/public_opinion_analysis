# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
import pymysql
from aip import AipNlp
import time
import random

class SohuPipeline(object):

    def __init__(self, host, port, user, passwd):
        self.host = host
        self.port = port
        self.user = user
        self.passwd = passwd

    @classmethod
    def from_crawler(cls, crawler):
        return cls(
            host=crawler.settings.get('MYSQL_HOST'),
            port=crawler.settings.get('MYSQL_PORT'),
            user=crawler.settings.get('MYSQL_USER'),
            passwd=crawler.settings.get('MYSQL_PASSWD')
        )

    def open_spider(self, spider):
        self.conn = pymysql.connect(host=self.host, port=self.port, user=self.user, passwd=self.passwd, db='copm', charset='utf8')
        self.cursor = self.conn.cursor()
        APP_ID = '11576120'
        API_KEY = 'WOSriMMnS9eSKGftEOtx5rb6'
        SECRET_KEY = '0Q6NaE89wbhRRjSVspjiDiiT6ZnGVllE'
        self.client = AipNlp(APP_ID, API_KEY, SECRET_KEY)

    def close_spider(self, spider):
        self.cursor.close()
        self.conn.close()

    def process_item(self, item, spider):
        if spider.name == 'zixun': 
            text = item['content']
        elif spider.name == 'qa':
            text = item['question_detail']
            if text == '':
                text = item['title']
        elif spider.name == 'bbs':
            text = item['post_detail']
            if text == '':
                text = item['title']
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
        if spider.name == 'zixun':
            sql = '''INSERT IGNORE INTO zixun (url, origin, publisher, time, title, abstraction, content) VALUES('{url}', '{origin}',
                     '{publisher}', '{time}', '{title}', '{abstraction}', '{content}')'''.format(**item)
            sql2 = '''INSERT IGNORE INTO content (address, title, date, abstract, keyword, mediaName, emotion) VALUES('{address}', '{title}',
                      '{datetime}', '{abstract}', '{keyword}', '{mediaName}', {emotion})'''.format(address=item['url'], title=item['title'],
                      datetime=item['time'], abstract=item['abstraction'], keyword=item['keyword'], mediaName=item['mediaName'], emotion=emotion)
        elif spider.name == 'qa':
            sql = '''INSERT IGNORE INTO qa (url, origin, title, question_time, question_detail, question_tags, question_follow_num,
                     question_answer_num) VALUES('{url}', '{origin}',
                     '{title}', '{question_time}', '{question_detail}', '{question_tags}', {question_follow_num}, {question_answer_num})'''.format(**item)
            sql2 = '''INSERT IGNORE INTO content (address, title, date, abstract, keyword, mediaName, emotion) VALUES('{address}', '{title}',
                    '{datetime}', '{abstract}', '{keyword}', '{mediaName}', {emotion})'''.format(address=item['url'], title=item['title'],
                    datetime=item['question_time'], abstract=item['question_detail'], keyword=item['keyword'], mediaName=item['mediaName'], emotion=emotion)
        elif spider.name == 'bbs':
            sql = '''INSERT IGNORE INTO bbs (url, origin, title, post_time, view_num, response_num, poster_id,
                     poster_level, post_detail) VALUES('{url}', '{origin}',
                     '{title}', '{post_time}', {view_num}, {response_num}, '{poster_id}', {poster_level}, '{post_detail}')'''.format(**item)
            sql2 = '''INSERT IGNORE INTO content (address, title, date, abstract, keyword, mediaName, emotion) VALUES('{address}', '{title}',
                    '{datetime}', '{abstract}', '{keyword}', '{mediaName}', {emotion})'''.format(address=item['url'], title=item['title'],
                    datetime=item['post_time'], abstract=item['post_detail'], keyword=item['keyword'], mediaName=item['mediaName'], emotion=emotion)
        self.cursor.execute(sql)
        self.cursor.execute(sql2) 
        self.conn.commit()
        return item
