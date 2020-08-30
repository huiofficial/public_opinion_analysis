# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class WechatItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    title = scrapy.Field()
    url = scrapy.Field()
    abstract = scrapy.Field()
    date = scrapy.Field()
    mediaName = scrapy.Field()
    keyword = scrapy.Field()
    emotion = scrapy.Field()

