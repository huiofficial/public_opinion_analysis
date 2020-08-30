# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class ZixunItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    url = scrapy.Field()
    origin = scrapy.Field()
    publisher = scrapy.Field()
    time = scrapy.Field()
    title = scrapy.Field()
    abstraction = scrapy.Field()
    content = scrapy.Field()
    keyword = scrapy.Field()
    mediaName = scrapy.Field()


class QAItem(scrapy.Item):
    url = scrapy.Field()
    origin = scrapy.Field()
    title = scrapy.Field()
    question_time = scrapy.Field()
    question_detail = scrapy.Field()
    question_tags = scrapy.Field()
    question_follow_num = scrapy.Field()
    question_answer_num = scrapy.Field()
    keyword = scrapy.Field()
    mediaName = scrapy.Field()


class BBSItem(scrapy.Item):
    url = scrapy.Field()
    origin = scrapy.Field()
    title = scrapy.Field()
    post_time = scrapy.Field()
    view_num = scrapy.Field()
    response_num = scrapy.Field()
    poster_id = scrapy.Field()
    poster_level = scrapy.Field()
    post_detail = scrapy.Field()
    keyword = scrapy.Field()
    mediaName = scrapy.Field()

