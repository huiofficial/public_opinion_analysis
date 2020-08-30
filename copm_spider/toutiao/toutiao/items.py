# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class ToutiaoItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    url = scrapy.Field()
    date = scrapy.Field()
    title = scrapy.Field()
    abstract = scrapy.Field()
    content = scrapy.Field()
    keyword = scrapy.Field()
    mediaName = scrapy.Field()
